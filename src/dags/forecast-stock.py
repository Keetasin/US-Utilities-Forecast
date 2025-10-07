# ==========================================================
# src/dags/forecast-stock.py  (SYNC 100% WITH Flask forecast.py; uses Parquet)
# ==========================================================
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pendulum
import pyarrow.parquet as pq

# ==========================================================
# CONFIG
# ==========================================================
PG_CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(PG_CONN)

# เลือกหุ้น/โมเดล/ฮอไรซอน
TICKERS = ["AEP"]
MODELS = ["ARIMA", "SARIMA", "SARIMAX", "LSTM"]
CALENDAR_TO_BDAYS = {7: 5, 180: 126, 365: 252}

# Path parquet ที่ pull_yf.py เขียนไว้
PARQUET_PATH = "/opt/airflow/src/spark/data/spark_out/market.parquet"

tz_th = pendulum.timezone("Asia/Bangkok")
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ==========================================================
# MODEL PARAMS (เหมือน Flask ทุกตัว)
# ==========================================================
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

# ==========================================================
# Calendar days -> Business days
# ==========================================================
def steps_to_bdays(steps: int) -> int:
    return CALENDAR_TO_BDAYS.get(steps, steps)

def to_bday_future_index(last_dt: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    bdays = steps_to_bdays(steps)
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=bdays)

# ==========================================================
# Utils (เหมือน Flask)
# ==========================================================
def to_scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])

def ensure_datetime_freq(series: pd.Series, use_bdays=True) -> pd.Series:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
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

def choose_seasonal_m(steps: int) -> int:
    if steps <= 30: return 5
    elif steps <= 180: return 20
    elif steps <= 365: return 63
    else: return 252

# ==========================================================
# Exogenous (เหมือน Flask)
# ==========================================================
def forecast_exog_series(exog_series: pd.Series, steps: int):
    """Random-walk จาก historical returns (เหมือน Flask)"""
    try:
        returns = exog_series.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last_val = exog_series.iloc[-1]
        future_vals = []
        for _ in range(steps):
            shock = np.random.normal(mu, sigma)
            last_val *= (1 + shock)
            future_vals.append(last_val)
        return pd.Series(future_vals, index=to_bday_future_index(exog_series.index[-1], steps))
    except Exception:
        last_val = exog_series.iloc[-1]
        return pd.Series([last_val] * steps, index=to_bday_future_index(exog_series.index[-1], steps))

# ==========================================================
# LSTM (better version — เหมือน Flask)
# ==========================================================
def lstm_forecast_better(series: pd.Series,
                         exog: pd.DataFrame | None = None,
                         steps: int = 7,
                         lookback: int = 120,
                         epochs: int = 1,
                         batch_size: int = 64,
                         patience: int = 10) -> np.ndarray:
    s = ensure_datetime_freq(series).astype(np.float32)
    rets = s.pct_change().dropna()

    if exog is not None:
        exog_clean = ensure_datetime_freq(exog)
        exog_rets = exog_clean.pct_change().dropna()
        combined_df = pd.concat([rets, exog_rets], axis=1).dropna()
        data = combined_df.values.astype(np.float32)
        num_features = data.shape[1]
    else:
        combined_df = rets.to_frame()
        data = combined_df.values.astype(np.float32)
        num_features = 1

    if len(combined_df) < lookback + 10:
        return np.array([float(s.iloc[-1])] * steps, dtype=np.float32)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    split = int(len(X) * 0.8)
    X_tr, y_tr, X_va, y_va = X[:split], y[:split], X[split:], y[split:]

    tf.keras.backend.clear_session()
    model = Sequential([
        tf.keras.Input(shape=(lookback, num_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    cbs = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 3), min_lr=1e-5),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cbs)

    last_window = scaled_data[-lookback:].copy()
    pred_prices, last_price = [], float(s.iloc[-1].item())

    for _ in range(steps):
        x = last_window.reshape(1, lookback, num_features)
        pred_z = float(model.predict(x, verbose=0)[0, 0])
        pred_ret = float(scaler.inverse_transform([[pred_z] + [0] * (num_features - 1)])[0, 0])
        last_price *= (1.0 + pred_ret)
        pred_prices.append(last_price)

        last_window = np.roll(last_window, -1, axis=0)
        last_window[-1, 0] = pred_z
        if num_features > 1:
            last_window[-1, 1:] = scaled_data[-1, 1:]

    return np.array(pred_prices, dtype=np.float32)

# ==========================================================
# Helpers (เหมือน Flask)
# ==========================================================
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
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b)
    elif model_name == "sarimax":
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog = ensure_datetime_freq(exog).reindex(s.index)
        exog = exog.ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)
        exog_train = exog.iloc[:-steps_b]
        exog_future = exog.iloc[-steps_b:]  # เหมือน Flask (ใช้อนาคตจริงสำหรับ backtest)
        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(train_data, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    exog=exog_train,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b, exog=exog_future)
    elif model_name == "lstm":
        fc_vals = lstm_forecast_better(train_data, exog=exog.iloc[:-steps_b] if exog is not None else None, steps=steps_b)
        fc = pd.Series(fc_vals, index=true_future.index)
    else:
        raise ValueError("Unknown model")

    mae = mean_absolute_error(true_future.values, fc.values)
    return pd.Series(fc.values, index=true_future.index), mae

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
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog_hist = ensure_datetime_freq(exog).reindex(s.index)
        exog_hist = exog_hist.ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)
        exog_future = pd.DataFrame(index=to_bday_future_index(last_dt, steps_b))
        for col in exog_hist.columns:
            exog_future[col] = forecast_exog_series(exog_hist[col], steps_b)
        exog_future = exog_future.replace([np.inf, -np.inf], np.nan).fillna(0)
        m_val = choose_seasonal_m(steps_b)
        m = SARIMAX(s, order=order,
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], m_val),
                    exog=exog_hist,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b, exog=exog_future)
    elif model_name == "lstm":
        vals = lstm_forecast_better(s, exog=exog, steps=steps_b)
        fc = pd.Series(vals, index=to_bday_future_index(last_dt, steps_b))
        return fc
    else:
        raise ValueError("Unknown model")

    return pd.Series(fc.values, index=to_bday_future_index(last_dt, steps_b))

# ==========================================================
# CORE: run model (อ่านจาก Parquet แทน yfinance)
# ==========================================================
def run_model(symbol, model_name, horizon, ti):
    try:
        steps_b = steps_to_bdays(horizon)
        mname = model_name.lower()
        print(f"🔹 Running {symbol}-{mname}-{horizon}d | steps={steps_b}")

        # --- Load parquet
        pdf = pq.read_table(PARQUET_PATH).to_pandas()
        pdf.columns = [c.lower() for c in pdf.columns]
        df = pdf[pdf["symbol"].str.lower() == symbol.lower()]
        if df.empty:
            raise ValueError(f"No data for {symbol}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date").asfreq("B").ffill().bfill()

        # close series
        data = df["close"].astype(float)

        # exog columns: pull_yf.py เซฟชื่อเป็น cl_f, ng_f, xlu (lowercase)
        exog_cols = [c for c in df.columns if c not in ["symbol", "close"]]
        exog = df[exog_cols] if exog_cols else None

        # --- Backtest & Forecast (ตรรกะเดียวกับ Flask)
        bt_series, bt_mae = backtest_last_n_days(data, model_name=mname, steps=horizon, exog=exog, symbol=symbol)
        fut_series = future_forecast(data, model_name=mname, steps=horizon, exog=exog, symbol=symbol)

        backtest_json = series_to_chart_pairs_safe(bt_series)
        forecast_json = series_to_chart_pairs_safe(fut_series)
        last_price = to_scalar(data.iloc[-1])

        # --- Save to Postgres
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
        print(f"[{model_name.upper()} Error] {e}")
        return {"symbol": symbol, "model": model_name, "horizon": horizon, "status": "error"}

# ==========================================================
# DAG helpers (XCom/Branch/Merge)
# ==========================================================
def read_latest(ti):
    pdf = pq.read_table(PARQUET_PATH).to_pandas()
    pdf.columns = [c.lower() for c in pdf.columns]
    df = pdf[pdf["symbol"].str.lower() == TICKERS[0].lower()].sort_values("date")
    last_close = float(df.iloc[-1]["close"])
    ti.xcom_push(key="last_close", value=last_close)
    print(f"[read_latest] last_close={last_close}")

def decide_branch(**ctx):
    return [f"forecast_group.{s}_{m}_{h}" for s in TICKERS for m in MODELS for h in CALENDAR_TO_BDAYS.keys()]

def merge_results(ti):
    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM stock_forecasts")).scalar()
        last = conn.execute(text("SELECT MAX(updated_at) FROM stock_forecasts")).scalar()
        print(f"📊 Total records: {total}, Last updated: {last}")

# ==========================================================
# DAG DEFINITION
# ==========================================================
with DAG(
    "forecast_stock_pipeline",
    default_args=default_args,
    description="Forecast DAG (Parquet source) synced 100% with Flask forecast.py",
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

    with TaskGroup("forecast_group") as forecast_group:
        for sym in TICKERS:
            for model in MODELS:
                for h in CALENDAR_TO_BDAYS.keys():
                    PythonOperator(
                        task_id=f"{sym}_{model}_{h}",
                        python_callable=run_model,
                        op_kwargs={"symbol": sym, "model_name": model, "horizon": h},
                    )

    merge = PythonOperator(
        task_id="merge_results",
        python_callable=merge_results,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    end = DummyOperator(task_id="end")

    start >> spark_transform >> read_latest_task >> branch >> forecast_group >> merge >> end