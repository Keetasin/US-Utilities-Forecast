from airflow import DAG
from airflow.operators.dummy import DummyOperator 
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
import pendulum
import pyarrow.parquet as pq
import builtins as _bi
import ta 
import traceback


PG_CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(PG_CONN)

TICKERS = ["AEP", "DUK", "SO", "ED", "EXC"]
MODELS = ["ARIMA", "SARIMA", "SARIMAX", "LSTM"]

CALENDAR_TO_BDAYS = {7: 5, 180: 126, 365: 252}

PARQUET_PATH = "/opt/airflow/src/spark/data/spark_out/market.parquet"

tz_th = pendulum.timezone("Asia/Bangkok")
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MODEL_PARAMS = {
    "AEP": {
        "arima":   {"order": (1, 1, 0)},
        "sarima":  {"order": (0, 1, 0), "seasonal_order": (1, 1, 0, 20)},
        "sarimax": {"order": (2, 0, 2), "seasonal_order": (0, 0, 1, 20)}
    },
    "DUK": {
        "arima":   {"order": (1, 1, 1)},
        "sarima":  {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, 63)},
        "sarimax": {"order": (0, 1, 2), "seasonal_order": (1, 1, 1, 63)}
    },
    "SO": {
        "arima":   {"order": (0, 1, 1)},
        "sarima":  {"order": (1, 0, 0), "seasonal_order": (1, 1, 0, 20)},
        "sarimax": {"order": (0, 1, 2), "seasonal_order": (1, 0, 1, 20)}
    },
    "ED": {
        "arima":   {"order": (2, 1, 2)},
        "sarima":  {"order": (2, 0, 2), "seasonal_order": (1, 1, 0, 63)},
        "sarimax": {"order": (2, 1, 1), "seasonal_order": (1, 1, 1, 63)}
    },
    "EXC": {
        "arima":   {"order": (2, 1, 2)},
        "sarima":  {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 20)},
        "sarimax": {"order": (1, 0, 0), "seasonal_order": (0, 1, 1, 63)}
    }
}

def get_params(symbol: str, model_name: str):
    symbol_cfg = MODEL_PARAMS.get(symbol, {})
    model_cfg = symbol_cfg.get(model_name, {})
    order = model_cfg.get("order", (2, 1, 0))
    seasonal_order = model_cfg.get("seasonal_order", (1, 1, 0, 20))
    return order, seasonal_order

def steps_to_bdays(steps: int) -> int:
    return CALENDAR_TO_BDAYS.get(int(steps), int(steps))

def to_bday_future_index(last_dt: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    bdays = steps_to_bdays(steps)
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=bdays)

def to_scalar(x) -> float:
    return _bi.float(np.asarray(x).reshape(-1)[0])

def ensure_datetime_freq(series: pd.Series | pd.DataFrame, use_bdays=True) -> pd.Series | pd.DataFrame:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    if len(s) == 0:
        return s
    inferred = pd.infer_freq(s.index)
    if inferred is None:
        full_idx = pd.bdate_range(s.index.min(), s.index.max()) if use_bdays else pd.date_range(s.index.min(), s.index.max())
        s = s.reindex(full_idx).ffill()
    return s

def series_to_chart_pairs_safe(series: pd.Series):
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(s))
        s.index = idx
    return [{"date": d.strftime('%Y-%m-%d'), "price": round(to_scalar(p), 2)} for d, p in zip(s.index, s.values)]

def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if "Close" in x.columns:
            s = x["Close"]
        else:
            num_cols = [c for c in x.columns if np.issubdtype(x[c].dtype, np.number)]
            s = x[num_cols[0]] if num_cols else x.iloc[:, 0]
        return pd.Series(s.values, index=s.index, name=getattr(s, "name", "val"))
    return pd.Series(x)

def _prep_exog_rets(exog, target_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    if exog is None:
        return None
    if isinstance(exog, pd.Series):
        exog = exog.to_frame(name="ex0")
    if not isinstance(exog, pd.DataFrame) or exog.empty:
        return None
    ex_clean = ensure_datetime_freq(exog)
    ex_rets = ex_clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    ex_rets = ex_rets.reindex(target_index).ffill().dropna(how="all")
    if ex_rets is None or ex_rets.empty:
        return None
    ex_rets.columns = [f"ex_{c}" for c in ex_rets.columns]
    return ex_rets

def forecast_exog_series(exog_series: pd.Series, steps: int):
    try:
        returns = exog_series.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last_val = _bi.float(exog_series.iloc[-1])
        future_vals = []
        for _ in range(steps_to_bdays(steps)):
            shock = np.random.normal(mu, sigma)
            last_val *= (1 + shock)
            future_vals.append(last_val)
        return pd.Series(future_vals, index=to_bday_future_index(exog_series.index[-1], steps))
    except Exception:
        last_val = _bi.float(exog_series.iloc[-1])
        return pd.Series([last_val] * steps_to_bdays(steps), index=to_bday_future_index(exog_series.index[-1], steps))

def build_indicators(close: pd.Series) -> pd.DataFrame:
    s = ensure_datetime_freq(close).astype(_bi.float)
    df = pd.DataFrame(index=s.index)

    df["ind_ema10"] = ta.trend.ema_indicator(s, window=10, fillna=True)
    df["ind_ema50"] = ta.trend.ema_indicator(s, window=50, fillna=True)

    df["ind_rsi14"] = ta.momentum.rsi(s, window=14, fillna=True)
 
    macd = ta.trend.MACD(s, window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["ind_macd"] = macd.macd()
    df["ind_macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(s, window=20, window_dev=2, fillna=True)
    df["ind_bbp"] = (s - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
   
    df["ind_vol20"] = s.pct_change().rolling(20, min_periods=1).std().fillna(0.0)
  
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df

def lstm_forecast_v2(
    series: pd.Series,
    exog: pd.DataFrame | None = None,
    steps: int = 7,
    lookback: int = 120,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 10,
    horizon_mode: str = "auto",  
    agg_step: int = 5,           
    clip_ret: float | None = 0.08,
    blend_drift: float = 0.2,     
) -> np.ndarray:
    _Sequential = tf.keras.models.Sequential
    _LSTM = tf.keras.layers.LSTM
    _Bidirectional = tf.keras.layers.Bidirectional
    _Dropout = tf.keras.layers.Dropout
    _Dense = tf.keras.layers.Dense
    _Huber = tf.keras.losses.Huber
    _EarlyStopping = tf.keras.callbacks.EarlyStopping
    _ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

    try:
        s = ensure_datetime_freq(_as_series(series)).astype(np.float32)
        rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if isinstance(rets, pd.DataFrame):
            rets = rets.iloc[:, 0]
        feats = [rets.rename("target_ret")]

        ex_rets = _prep_exog_rets(exog, rets.index)
        if ex_rets is not None:
            feats.append(ex_rets)

        df = pd.concat(feats, axis=1).dropna(how="any")
        if len(df) == 0:
            return np.asarray([_bi.float(s.iloc[-1])] * steps_to_bdays(steps), dtype=np.float32)

        H = steps_to_bdays(int(steps))
        drift = _bi.float(df["target_ret"].mean()) if "target_ret" in df.columns else 0.0

        if horizon_mode == "auto":
            horizon_mode = "direct" if H > 63 else "recursive"

        data = df.values.astype(np.float32)
        nfeat = data.shape[1]
        scaler_X = SK_StandardScaler()
        data_z = scaler_X.fit_transform(data)
        L = int(lookback)

        if horizon_mode == "recursive":
            if len(df) < (L + 10):
                last_price = _bi.float(s.iloc[-1])
                out = []
                for t in range(H):
                    r = drift
                    if blend_drift > 0:
                        w = blend_drift * (t + 1) / H
                        r = (1 - w) * r + w * drift
                    if clip_ret is not None:
                        r = _bi.float(np.clip(r, -clip_ret, clip_ret))
                    last_price *= (1.0 + r)
                    out.append(last_price)
                return np.asarray(out, dtype=np.float32)

            X, y = [], []
            for i in range(L, len(data_z)):
                X.append(data_z[i - L:i])
                y.append(data_z[i, 0])  
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            split = int(len(X) * 0.8)
            X_tr, y_tr = X[:split], y[:split]
            X_va, y_va = X[split:], y[split:]

            tf.keras.backend.clear_session()
            model = _Sequential([
                _Bidirectional(_LSTM(64, return_sequences=True)),
                _Dropout(0.2),
                _Bidirectional(_LSTM(32)),
                _Dense(1),
            ])
            model.compile(optimizer="adam", loss=_Huber())
            cbs = [
                _EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True),
                _ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, int(patience)//3), min_lr=1e-5),
            ]
            model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=int(epochs), batch_size=int(batch_size), verbose=0, callbacks=cbs)

            last_price = _bi.float(s.iloc[-1])
            last_win = data_z[-L:].copy()
            out = []
            for t in range(H):
                z_scaled = _bi.float(model.predict(last_win.reshape(1, L, nfeat), verbose=0)[0, 0])
                inv_ret = _bi.float(
                    scaler_X.inverse_transform(
                        np.array([[z_scaled] + [0]*(nfeat-1)], dtype=np.float32)
                    )[0, 0]
                )
                if blend_drift > 0:
                    w = blend_drift * (t + 1) / H
                    inv_ret = (1 - w) * inv_ret + w * drift
                if clip_ret is not None:
                    inv_ret = _bi.float(np.clip(inv_ret, -clip_ret, clip_ret))

                last_price *= (1.0 + inv_ret)
                out.append(last_price)

                last_win = np.roll(last_win, -1, axis=0)
                last_win[-1, 0] = z_scaled
                if nfeat > 1:
                    last_win[-1, 1:] = last_win[-2, 1:]
            return np.asarray(out, dtype=np.float32)

        else:
            import math
            K = int(math.ceil(H / int(agg_step)))
            target_lr = np.log1p(df.iloc[:, 0].astype(np.float32)).values
            max_shift = K * int(agg_step)
            if len(df) < (L + max_shift + 10):
                return lstm_forecast_v2(series, exog, steps, lookback, epochs, batch_size, patience,
                                        horizon_mode="recursive", clip_ret=clip_ret, blend_drift=blend_drift)

            X_list, Y_list = [], []
            for i in range(L, len(data_z) - max_shift):
                X_list.append(data_z[i - L:i])
                yk = []
                for k in range(K):
                    start = i + k * int(agg_step)
                    end   = start + int(agg_step)
                    yk.append(target_lr[start:end].sum())
                Y_list.append(yk)
            X = np.asarray(X_list, dtype=np.float32)
            Y = np.asarray(Y_list, dtype=np.float32)

            split = int(len(X) * 0.8)
            X_tr, X_va = X[:split], X[split:]
            Y_tr, Y_va = Y[:split], Y[split:]

            y_scaler = SK_StandardScaler()
            Y_tr_z = y_scaler.fit_transform(Y_tr)
            Y_va_z = y_scaler.transform(Y_va)

            tf.keras.backend.clear_session()
            model = _Sequential([
                _Bidirectional(_LSTM(64, return_sequences=True)),
                _Dropout(0.2),
                _Bidirectional(_LSTM(32)),
                _Dense(K),
            ])
            model.compile(optimizer="adam", loss=_Huber())
            cbs = [
                _EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True),
                _ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, int(patience)//3), min_lr=1e-5),
            ]
            model.fit(X_tr, Y_tr_z, validation_data=(X_va, Y_va_z),
                      epochs=int(epochs), batch_size=int(batch_size), verbose=0, callbacks=cbs)

            last_price = _bi.float(s.iloc[-1])
            last_win = data_z[-L:].copy()
            y_pred_z = model.predict(last_win.reshape(1, L, nfeat), verbose=0)[0]
            y_pred_lr_blocks = y_scaler.inverse_transform(y_pred_z.reshape(1, -1))[0]

            out = []
            for k in range(K):
                block_lr = _bi.float(y_pred_lr_blocks[k])
                if blend_drift > 0:
                    w = blend_drift * (k + 1) / K
                    block_lr = (1 - w) * block_lr + w * np.log1p(drift)

                days_in_block = int(agg_step) if k < K - 1 else (H - int(agg_step) * (K - 1))
                per_day_lr = block_lr / max(1, days_in_block)
                per_day_ret = np.expm1(per_day_lr)
                for _ in range(days_in_block):
                    r = per_day_ret
                    if clip_ret is not None:
                        r = _bi.float(np.clip(r, -clip_ret, clip_ret))
                    last_price *= (1.0 + r)
                    out.append(last_price)
            return np.asarray(out[:H], dtype=np.float32)

    except Exception:
        print("[LSTM DEBUG] traceback:\n", traceback.format_exc())
        raise

def lstm_forecast_better(series: pd.Series,
                         exog: pd.DataFrame | None = None,
                         steps: int = 7,
                         lookback: int = 120,
                         epochs: int = 30,
                         batch_size: int = 64,
                         patience: int = 10) -> np.ndarray:
    H = steps_to_bdays(int(steps))
    if H > 63:
        return lstm_forecast_v2(
            series, exog, steps, lookback, epochs, batch_size, patience,
            horizon_mode="direct", agg_step=5, clip_ret=0.08, blend_drift=0.2
        )
    else:
        return lstm_forecast_v2(
            series, exog, steps, lookback, epochs, batch_size, patience,
            horizon_mode="recursive", agg_step=1, clip_ret=0.07, blend_drift=0.0
        )

def backtest_last_n_days(series: pd.Series, model_name: str, steps=7, exog=None, symbol="GENERIC"):
    s = ensure_datetime_freq(series)
    steps_b = steps_to_bdays(steps)
    if len(s) <= steps_b:
        raise ValueError("Series too short for backtest window.")

    train_data = s.iloc[:-steps_b]
    true_future = s.iloc[-steps_b:]

    order, seasonal_order = get_params(symbol, model_name)

    if model_name == "arima":
        m = ARIMA(train_data, order=order).fit()
        fc = m.forecast(steps=steps_b)

    elif model_name == "sarima":
        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(train_data, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, method="powell", maxiter=50)
        fc = m.forecast(steps=steps_b)

    elif model_name == "sarimax":
        if (exog is None) or (isinstance(exog, pd.DataFrame) and exog.empty):
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exg = ensure_datetime_freq(exog).reindex(s.index)
        exg = exg.ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)
        ex_tr = exg.iloc[:-steps_b]
        ex_fu = exg.iloc[-steps_b:]

        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(train_data, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    exog=ex_tr,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, method="powell", maxiter=50)
        fc = m.forecast(steps=steps_b, exog=ex_fu)

    elif model_name == "lstm":
        fc_vals = lstm_forecast_better(
            train_data,
            exog=exog.iloc[:-steps_b] if isinstance(exog, pd.DataFrame) else None,
            steps=steps_b
        )
        fc = pd.Series(fc_vals, index=true_future.index)

    else:
        raise ValueError("Unknown model")

    mae = mean_absolute_error(true_future.values, pd.Series(fc, index=true_future.index).values)
    return pd.Series(fc, index=true_future.index), mae

def future_forecast(series: pd.Series, model_name: str, steps=7, exog=None, symbol="GENERIC"):
    s = ensure_datetime_freq(series)
    last_dt = s.index[-1]
    steps_b = steps_to_bdays(steps)

    order, seasonal_order = get_params(symbol, model_name)

    if model_name == "arima":
        m = ARIMA(s, order=order).fit()
        fc = m.forecast(steps=steps_b)

    elif model_name == "sarima":
        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(s, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b)

    elif model_name == "sarimax":
        if (exog is None) or (isinstance(exog, pd.DataFrame) and exog.empty):
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        ex_hist = ensure_datetime_freq(exog).reindex(s.index)
        ex_hist = ex_hist.ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)

        fut_idx = to_bday_future_index(last_dt, steps_b)
        ex_future = pd.DataFrame(index=fut_idx)
        for col in ex_hist.columns:
            ex_future[col] = forecast_exog_series(ex_hist[col], steps_b)
        ex_future = ex_future.replace([np.inf, -np.inf], np.nan).fillna(0)

        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(s, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    exog=ex_hist,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b, exog=ex_future)
        return pd.Series(np.asarray(fc).ravel(), index=fut_idx)

    elif model_name == "lstm":
        vals = lstm_forecast_better(s, exog=exog, steps=steps_b)
        return pd.Series(vals, index=to_bday_future_index(last_dt, steps_b))

    else:
        raise ValueError("Unknown model")

    return pd.Series(fc.values, index=to_bday_future_index(last_dt, steps_b))

def choose_seasonal_m(steps: int) -> int:
    if steps <= 30: return 5
    elif steps <= 180: return 20
    # elif steps <= 365: return 63
    elif steps <= 365: return 20
    else: return 252

def run_model(symbol, model_name, horizon, ti):
    try:
        steps_b = steps_to_bdays(horizon)
        mname = model_name.lower()
        print(f"🔹 Running {symbol}-{mname}-{horizon}d | steps={steps_b}")

        pdf = pq.read_table(PARQUET_PATH).to_pandas()
        pdf.columns = [c.lower() for c in pdf.columns]
        df = pdf[pdf["symbol"].str.lower() == symbol.lower()]
        if df.empty:
            raise ValueError(f"No data for {symbol}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date").asfreq("B").ffill().bfill()
        close = df["close"].astype(float)

        ind = build_indicators(close) 

        external_cols = [c for c in df.columns if c not in ["symbol", "close"] and not c.startswith("ind_")]
        exog_external = df[external_cols] if external_cols else None

        exog_lstm = None
        if exog_external is not None and not exog_external.empty:
            exog_lstm = pd.concat([exog_external, ind], axis=1)
        else:
            exog_lstm = ind.copy()

        if mname == "sarimax":
            exog_for_bt = exog_external
            exog_for_fut = exog_external
        elif mname == "lstm":
            exog_for_bt = exog_lstm
            exog_for_fut = exog_lstm
        else:
            exog_for_bt = None
            exog_for_fut = None

        bt_series, bt_mae = backtest_last_n_days(close, model_name=mname, steps=horizon, exog=exog_for_bt, symbol=symbol)
        fut_series = future_forecast(close, model_name=mname, steps=horizon, exog=exog_for_fut, symbol=symbol)

        backtest_json = series_to_chart_pairs_safe(bt_series)
        forecast_json = series_to_chart_pairs_safe(fut_series)
        last_price = to_scalar(close.iloc[-1])

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_forecasts (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    steps INT NOT NULL,
                    forecast_json JSONB,
                    backtest_json JSONB,
                    backtest_mae DOUBLE PRECISION,
                    last_price DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(symbol, model, steps)
                );
            """))
            conn.execute(text("""
                INSERT INTO stock_forecasts (symbol, model, steps, forecast_json, backtest_json, backtest_mae, last_price, updated_at)
                VALUES (:symbol, :model, :steps, :forecast_json, :backtest_json, :mae, :last_price, NOW())
                ON CONFLICT (symbol, model, steps)
                DO UPDATE SET
                    forecast_json = EXCLUDED.forecast_json,
                    backtest_json = EXCLUDED.backtest_json,
                    backtest_mae = EXCLUDED.backtest_mae,
                    last_price = EXCLUDED.last_price,
                    updated_at = NOW();
            """), {
                "symbol": symbol,
                "model": mname,
                "steps": horizon,
                "forecast_json": json.dumps(forecast_json),
                "backtest_json": json.dumps(backtest_json),
                "mae": float(bt_mae) if bt_mae is not None else None,
                "last_price": float(last_price),
            })

        print(f"✅ Updated {symbol}-{mname}-{horizon}d | Backtest MAE: {bt_mae:.4f}")
        return {"symbol": symbol, "model": mname, "horizon": horizon, "status": "ok"}

    except Exception as e:
        print(f"[{model_name.upper()} Error] {e}\n{traceback.format_exc()}")
        return {"symbol": symbol, "model": model_name, "horizon": horizon, "status": "error"}

def read_latest(ti):
    pdf = pq.read_table(PARQUET_PATH).to_pandas()
    pdf.columns = [c.lower() for c in pdf.columns]
    df = pdf[pdf["symbol"].str.lower() == TICKERS[0].lower()].sort_values("date")
    last_close = float(df.iloc[-1]["close"])
    ti.xcom_push(key="last_close", value=last_close)
    print(f"[read_latest] last_close={last_close}")

# def decide_branch(**ctx):
#     return [f"forecast_group.{s}_{m}_{h}" for s in TICKERS for m in MODELS for h in CALENDAR_TO_BDAYS.keys()]

def decide_branch(**kwargs):
    branch_ids = []
    for sym in TICKERS:
        for model in MODELS:
            for h in CALENDAR_TO_BDAYS.keys():
                branch_ids.append(f"forecast_group_{sym}.{sym}_{model}_{h}")
    return branch_ids


# def merge_results(ti):
#     with engine.begin() as conn:
#         total = conn.execute(text("SELECT COUNT(*) FROM stock_forecasts")).scalar()
#         last = conn.execute(text("SELECT MAX(updated_at) FROM stock_forecasts")).scalar()
#         print(f"📊 Total records: {total}, Last updated: {last}")

with DAG(
    "forecast_stock_pipeline",
    default_args=default_args,
    description="Forecast DAG (Parquet source) synced with Flask (LSTM v2 + Indicators)",
    schedule_interval="30 19 * * 1-5",
    start_date=datetime(2025, 1, 1, tzinfo=tz_th),
    catchup=False,
) as dag:

    start = DummyOperator(task_id="start")

    spark_transform = SparkSubmitOperator(
        task_id="spark_transform",
        application="/opt/airflow/src/spark/applications/spark_jobs/pull_yf.py",
        conn_id="spark_default",
        name="pull-yfinance-to-parquet",
        verbose=True,
        conf={"spark.master": "local[*]"},
        application_args=["--period", "5y", "--tickers", ",".join(TICKERS), "--exog", "CL=F,NG=F,XLU"],
    )

    read_latest_task = PythonOperator(task_id="read_latest", python_callable=read_latest)
    branch = BranchPythonOperator(task_id="branching", python_callable=decide_branch, provide_context=True)

    prev_group = branch
    for sym in TICKERS:
        with TaskGroup(group_id=f"forecast_group_{sym}") as forecast_group:
            for model in MODELS:
                for h in CALENDAR_TO_BDAYS.keys():
                    PythonOperator(
                        task_id=f"{sym}_{model}_{h}",
                        python_callable=run_model,
                        op_kwargs={"symbol": sym, "model_name": model, "horizon": h},  
                    )

        prev_group >> forecast_group
        prev_group = forecast_group

    # merge = PythonOperator(
    #     task_id="merge_results",
    #     python_callable=merge_results,
    #     trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    # )

    end = DummyOperator(task_id="end")


    start >> spark_transform >> read_latest_task >> branch
    # prev_group >> merge >> end
    prev_group >> end