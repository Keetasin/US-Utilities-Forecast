import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
from ..models import StockForecast
from .. import db
from datetime import datetime

# ==========================
# Custom parameters for each stock
# ==========================
MODEL_PARAMS = {
    "AEP": {
        "arima":   {"order": (2,1,2)},
        "sarima":  {"order": (1,1,1), "seasonal_order": (0,1,1,20)},
        "sarimax": {"order": (2,1,1), "seasonal_order": (1,0,1,20)}
    },
    "DUK": {
        "arima":   {"order": (1,1,1)},
        "sarima":  {"order": (2,1,0), "seasonal_order": (0,1,1,20)},
        "sarimax": {"order": (1,1,2), "seasonal_order": (1,0,1,20)}
    },
    "SO": {
        "arima":   {"order": (0,1,2)},
        "sarima":  {"order": (1,1,1), "seasonal_order": (1,1,0,20)},
        "sarimax": {"order": (2,1,2), "seasonal_order": (0,0,1,20)}
    },
    "ED": {
        "arima":   {"order": (2,1,0)},
        "sarima":  {"order": (1,1,2), "seasonal_order": (0,1,1,20)},
        "sarimax": {"order": (1,1,1), "seasonal_order": (1,0,0,20)}
    },
    "EXC": {
        "arima":   {"order": (1,1,2)},
        "sarima":  {"order": (0,1,1), "seasonal_order": (1,1,1,20)},
        "sarimax": {"order": (2,1,1), "seasonal_order": (0,0,1,20)}
    }
}

def get_params(symbol: str, model_name: str):
    symbol_cfg = MODEL_PARAMS.get(symbol, {})
    model_cfg = symbol_cfg.get(model_name, {})
    order = model_cfg.get("order", (2,1,0))
    seasonal_order = model_cfg.get("seasonal_order", (1,1,0,20))
    return order, seasonal_order

# ==========================
# Exogenous variables
# ==========================
EXOG_TICKERS = {"oil": "CL=F", "gas": "NG=F", "xlu": "XLU"}

def get_exogenous(period="5y"):
    exog_df = pd.DataFrame()
    for name, tkr in EXOG_TICKERS.items():
        try:
            s = yf.download(tkr, period=period, progress=False, auto_adjust=True)["Close"].dropna()
            s = ensure_datetime_freq(s)
            exog_df[name] = s
        except Exception as e:
            print(f"[Exog] Failed to fetch {tkr}: {e}")
    return exog_df

def forecast_exog_series(exog_series: pd.Series, steps: int, order=(1,1,1)):
    try:
        model = ARIMA(exog_series, order=order).fit()
        return model.forecast(steps=steps)
    except Exception as e:
        print(f"[Exog Forecast Error] {e}")
        last_val = exog_series.iloc[-1]
        return pd.Series([last_val]*steps, index=to_bday_future_index(exog_series.index[-1], steps))

# ==========================
# Mapping calendar horizon → BDays
# ==========================
CALENDAR_TO_BDAYS = {7: 5, 90: 63, 180: 126, 365: 252}

def steps_to_bdays(steps: int) -> int:
    return CALENDAR_TO_BDAYS.get(steps, steps)

def to_bday_future_index(last_dt: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    bdays = steps_to_bdays(steps)
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=bdays)

# ==========================
# Utils
# ==========================
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
    return [{"date": d.strftime('%Y-%m-%d'), "price": round(to_scalar(p),2)} for d,p in zip(s.index, s.values)]

def get_period_by_model(model_name: str, steps: int) -> str:
    model_name = (model_name or "arima").lower()
    if model_name == "lstm":
        return "10y"
    if steps <= 7:
        return "6mo"
    elif steps <= 180:
        return "2y"
    elif steps <= 365:
        return "5y"
    else:
        return "10y"

# ==========================
# Seasonal logic
# ==========================
def choose_seasonal_m(steps: int) -> int:
    if steps <= 30: return 5
    elif steps <= 180: return 20
    elif steps <= 365: return 63
    else: return 252

# ==========================
# NEW: Downsample forecast series
# ==========================
def downsample_forecast_series(fc: pd.Series, steps: int) -> pd.Series:
    if fc is None or fc.empty:
        return fc
    if steps == 180:   # 6 เดือน → 6 จุด
        return fc.resample("M").last().iloc[:6]
    elif steps == 365: # 1 ปี → 12 จุด
        return fc.resample("M").last().iloc[:12]
    return fc

# ==========================
# LSTM (better version)
# ==========================
def lstm_forecast_better(series: pd.Series,
                         exog: pd.DataFrame | None = None,
                         steps: int = 7,
                         lookback: int = 120,
                         epochs: int = 3,
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
        print("⚠️ Not enough data for LSTM training → fallback to naive forecast")
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
    pred_prices, last_price = [], float(s.iloc[-1])

    for _ in range(steps):
        x = last_window.reshape(1, lookback, num_features)
        pred_z = float(model.predict(x, verbose=0)[0, 0])
        pred_ret = float(scaler.inverse_transform([[pred_z] + [0]*(num_features-1)])[0, 0])

        last_price *= (1.0 + pred_ret)
        pred_prices.append(last_price)

        last_window = np.roll(last_window, -1, axis=0)
        last_window[-1, 0] = pred_z
        if num_features > 1:
            last_window[-1, 1:] = scaled_data[-1, 1:]

    return np.array(pred_prices, dtype=np.float32)

# ==========================
# Forecasting helpers
# ==========================
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
        exog = exog.ffill().bfill()
        exog = exog.replace([np.inf, -np.inf], np.nan).fillna(0)

        exog_train = exog.iloc[:-steps_b]
        exog_future = exog.iloc[-steps_b:]
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
    fc = pd.Series(fc.values, index=true_future.index)
    fc = downsample_forecast_series(fc, steps)
    return fc, mae

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
        exog_hist = exog_hist.ffill().bfill()
        exog_hist = exog_hist.replace([np.inf, -np.inf], np.nan).fillna(0)

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
        fc = downsample_forecast_series(fc, steps)
        return fc
    else:
        raise ValueError("Unknown model")

    fc = pd.Series(fc.values, index=to_bday_future_index(last_dt, steps_b))
    fc = downsample_forecast_series(fc, steps)
    return fc

# ==========================
# Update forecast
# ==========================
def update_forecast(app, tickers, models=["arima","sarima","sarimax","lstm"], steps_list=[7,90,365]):
    with app.app_context():
        for symbol in tickers:
            for m in models:
                for steps in steps_list:
                    try:
                        period = get_period_by_model(m, steps)
                        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
                        data = ensure_datetime_freq(data)
                        if len(data) < max(70, steps+10):
                            print(f"[Forecast] Skip {symbol}-{m}-{steps}d (not enough data)")
                            continue

                        exog = None
                        if m in ["sarimax","lstm"]:
                            exog_all = get_exogenous(period=period)
                            exog_all = exog_all.reindex(data.index)
                            exog_all = exog_all.ffill().bfill()
                            exog_all = exog_all.replace([np.inf, -np.inf], np.nan).fillna(0)
                            exog = exog_all

                        back_fc, back_mae = backtest_last_n_days(data, model_name=m, steps=steps, exog=exog, symbol=symbol)
                        backtest_json = series_to_chart_pairs_safe(back_fc)

                        fut_fc = future_forecast(data, model_name=m, steps=steps, exog=exog, symbol=symbol)
                        forecast_json = series_to_chart_pairs_safe(fut_fc)

                        last_price = to_scalar(data.iloc[-1])

                        fc = StockForecast.query.filter_by(symbol=symbol, model=m, steps=steps).first()
                        if not fc:
                            fc = StockForecast(symbol=symbol,
                                               model=m,
                                               steps=steps,
                                               forecast_json=forecast_json,
                                               backtest_json=backtest_json,
                                               backtest_mae=back_mae,
                                               last_price=last_price,
                                               updated_at=datetime.utcnow())
                            db.session.add(fc)
                        else:
                            fc.forecast_json = forecast_json
                            fc.backtest_json = backtest_json
                            fc.backtest_mae = back_mae
                            fc.last_price = last_price
                            fc.updated_at = datetime.utcnow()
                        db.session.commit()
                        print(f"[Forecast] Updated {symbol}-{m}-{steps}d | Backtest MAE: {back_mae:.4f}")
                    except Exception as e:
                        print(f"[Forecast Error] {symbol}-{m}-{steps}d: {e}")
