import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from ..models import StockForecast
from .. import db
from datetime import datetime

# ==========================
# Custom parameters for each stock
# ==========================
MODEL_PARAMS = {
    "AEP": {
        "arima": {"order": (2,1,2)},
        "sarima": {"order": (2,1,1), "seasonal_order": (1,1,0,12)},
        "sarimax": {"order": (1,1,1), "seasonal_order": (1,1,1,12)},
    },
    "DUK": {
        "arima": {"order": (1,1,1)},
        "sarima": {"order": (3,1,0), "seasonal_order": (0,1,1,12)},
        "sarimax": {"order": (2,1,1), "seasonal_order": (1,0,1,12)},
    },
    "SO": {
        "arima": {"order": (5,1,0)},
        "sarima": {"order": (2,1,2), "seasonal_order": (1,1,0,12)},
        "sarimax": {"order": (2,1,0), "seasonal_order": (1,1,1,12)},
    },
    "ED": {
        "arima": {"order": (2,1,3)},
        "sarima": {"order": (1,1,1), "seasonal_order": (1,0,0,12)},
        "sarimax": {"order": (3,1,0), "seasonal_order": (0,1,1,12)},
    },
    "EXC": {
        "arima": {"order": (3,1,1)},
        "sarima": {"order": (2,1,1), "seasonal_order": (1,1,0,12)},
        "sarimax": {"order": (1,1,2), "seasonal_order": (1,0,1,12)},
    },
}

def get_params(symbol: str, model_name: str):
    """Return order and seasonal_order for given stock and model"""
    symbol_cfg = MODEL_PARAMS.get(symbol, {})
    model_cfg = symbol_cfg.get(model_name, {})
    order = model_cfg.get("order", (2,1,0))
    seasonal_order = model_cfg.get("seasonal_order", (1,1,0,12))
    return order, seasonal_order

# ==========================
# Exogenous variables
# ==========================
def get_exogenous(period="5y"):
    """ดึง exogenous variables หลายตัวแล้วรวมเป็น DataFrame"""
    tickers = {
        "oil": "CL=F",
        "gas": "NG=F",
        "xlu": "XLU"
    }
    exog_df = pd.DataFrame()
    for name, tkr in tickers.items():
        try:
            s = yf.download(tkr, period=period, progress=False, auto_adjust=True)["Close"].dropna()
            s = ensure_datetime_freq(s)
            exog_df[name] = s
        except Exception as e:
            print(f"[Exog] Failed to fetch {tkr}: {e}")
    return exog_df

# ==========================
# Mapping calendar horizon → BDays
# ==========================
CALENDAR_TO_BDAYS = {
    7: 5,
    90: 63,
    180: 126,
    365: 252
}

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
        m = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b)
    elif model_name == "sarimax":
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog = ensure_datetime_freq(exog)
        exog = exog.reindex(s.index).ffill()
        exog_train = exog.iloc[:-steps_b]
        exog_future = exog.iloc[-steps_b:]
        m = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                    exog=exog_train,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b, exog=exog_future)
    elif model_name == "lstm":
        fc_vals = lstm_forecast_simple(train_data, steps_b)
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
        m = SARIMAX(s, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b)
    elif model_name == "sarimax":
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog = ensure_datetime_freq(exog)
        exog_hist = exog.reindex(s.index).ffill()
        exog_future = _build_exog_future_like_last(exog_hist, last_dt, steps_b)
        m = SARIMAX(s, order=order, seasonal_order=seasonal_order,
                    exog=exog_hist,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps_b, exog=exog_future)
    elif model_name == "lstm":
        vals = lstm_forecast_simple(s, steps_b)
        fc = pd.Series(vals, index=to_bday_future_index(last_dt, steps_b))
        return fc
    else:
        raise ValueError("Unknown model")

    fc = pd.Series(fc.values, index=to_bday_future_index(last_dt, steps_b))
    return fc

def _build_exog_future_like_last(exog: pd.DataFrame, last_dt: pd.Timestamp, steps_b: int) -> pd.DataFrame:
    future_idx = to_bday_future_index(last_dt, steps_b)
    last_vals = exog.iloc[-1].values
    future_vals = np.tile(last_vals, (len(future_idx), 1))
    return pd.DataFrame(future_vals, index=future_idx, columns=exog.columns)

# ==========================
# LSTM
# ==========================
def lstm_forecast_simple(series: pd.Series, steps=7, lookback=60, epochs=1, batch_size=64):
    values = series.values.reshape(-1,1).astype(np.float32)
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(values)
    X,y=[],[]
    for i in range(lookback,len(scaled)):
        X.append(scaled[i-lookback:i,0])
        y.append(scaled[i,0])
    X=np.array(X)
    y=np.array(y)
    if len(X)<10:
        return np.array([float(values[-1,0])]*steps)
    split = int(len(X)*0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1],1))
    tf.keras.backend.clear_session()
    model = Sequential([
        tf.keras.Input(shape=(lookback,1)),
        LSTM(16, return_sequences=True),
        Dropout(0.1),
        LSTM(8),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_data=(X_val,y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)
    last_window = scaled[-lookback:,0].copy()
    preds=[]
    for _ in range(steps):
        x = last_window.reshape(1, lookback,1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        last_window = np.roll(last_window,-1)
        last_window[-1]=p
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()

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
                        if m == "sarimax":
                            exog_all = get_exogenous(period=period)
                            exog_all = exog_all.reindex(data.index).ffill()
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
