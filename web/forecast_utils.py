import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

def to_bday_future_index(last_dt: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=steps)

def series_to_chart_pairs_safe(series: pd.Series):
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(s))
        s.index = idx
    return [{"date": d.strftime('%Y-%m-%d'), "price": round(to_scalar(p),2)} for d,p in zip(s.index, s.values)]

def get_period_by_model(model_name: str) -> str:
    model_name = (model_name or "arima").lower()
    return {"arima":"6mo", "sarima":"3y", "lstm":"10y"}.get(model_name,"6mo")

# Forecasting helpers
def backtest_last_n_days(series: pd.Series, model_name: str, steps=7):
    s = ensure_datetime_freq(series)
    train_data = s.iloc[:-steps]
    true_future = s.iloc[-steps:]
    if model_name=="arima":
        m = ARIMA(train_data, order=(2,1,0)).fit()
        fc = m.forecast(steps=steps)
    elif model_name=="sarima":
        m = SARIMAX(train_data, order=(2,1,0), seasonal_order=(1,1,0,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)
    elif model_name=="lstm":
        fc = lstm_forecast_simple(train_data, steps)
        fc = pd.Series(fc, index=true_future.index)
    else:
        raise ValueError("Unknown model")
    mae = mean_absolute_error(true_future.values, fc.values)
    return pd.Series(fc.values, index=true_future.index), mae

def future_forecast(series: pd.Series, model_name: str, steps=7):
    s = ensure_datetime_freq(series)
    last_dt = s.index[-1]
    if model_name=="arima":
        m = ARIMA(s, order=(2,1,0)).fit()
        fc = m.forecast(steps=steps)
    elif model_name=="sarima":
        m = SARIMAX(s, order=(2,1,0), seasonal_order=(1,1,0,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)
    elif model_name=="lstm":
        vals = lstm_forecast_simple(s, steps)
        fc = pd.Series(vals, index=to_bday_future_index(last_dt, steps))
        return fc
    else:
        raise ValueError("Unknown model")
    fc = pd.Series(fc.values, index=to_bday_future_index(last_dt, steps))
    return fc

def lstm_forecast_simple(series: pd.Series, steps=7, lookback=60, epochs=15, batch_size=32):
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
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.1),
        LSTM(32),
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
