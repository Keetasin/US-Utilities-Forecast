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


def get_period_by_model(model_name: str, steps: int) -> str:
    """
    กำหนด period ของ yfinance ตาม model และ horizon (steps)
    """
    model_name = (model_name or "arima").lower()

    if model_name == "lstm":
        return "10y"

    # non-LSTM → ปรับตาม horizon
    if steps <= 7:
        return "6mo"   # 6 เดือนสำหรับ historical ของ 1 week
    elif steps <= 180:
        return "2y"    # 2 ปีเพื่อให้มีข้อมูล historical สำหรับ 6 เดือน
    elif steps <= 365:
        return "5y"    # 5 ปีเพื่อให้มีข้อมูล historical สำหรับ 1 ปี
    else:
        return "10y"


# ==========================
# Forecasting helpers
# ==========================
def _build_exog_future_like_last(exog: pd.Series, last_dt: pd.Timestamp, steps: int) -> pd.Series:
    """สร้าง exog อนาคตแบบ naive: ใช้ค่าล่าสุดคงที่ แล้วยืด index ออกไปตามวันทำการ"""
    future_idx = to_bday_future_index(last_dt, steps)
    return pd.Series([to_scalar(exog.iloc[-1])] * steps, index=future_idx)

def backtest_last_n_days(series: pd.Series, model_name: str, steps=7, exog=None):
    s = ensure_datetime_freq(series)
    if len(s) <= steps:
        raise ValueError("Series too short for backtest window.")

    train_data = s.iloc[:-steps]
    true_future = s.iloc[-steps:]

    if model_name == "arima":
        # ใช้ค่าพารามิเตอร์เดียวกันทั้ง backtest/future เพื่อลดความเพี้ยน
        m = ARIMA(train_data, order=(2,1,0)).fit()
        fc = m.forecast(steps=steps)

    elif model_name == "sarima":
        m = SARIMAX(train_data, order=(2,1,0), seasonal_order=(1,1,0,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)

    elif model_name == "sarimax":
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog = ensure_datetime_freq(exog)
        exog = exog.reindex(s.index).ffill()
        exog_train = exog.iloc[:-steps]
        exog_future = exog.iloc[-steps:]
        m = SARIMAX(train_data, order=(2,1,0), seasonal_order=(1,1,0,12),
                    exog=exog_train,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps, exog=exog_future)

    elif model_name == "lstm":
        fc_vals = lstm_forecast_simple(train_data, steps)
        fc = pd.Series(fc_vals, index=true_future.index)

    else:
        raise ValueError("Unknown model")

    mae = mean_absolute_error(true_future.values, fc.values)
    return pd.Series(fc.values, index=true_future.index), mae


def future_forecast(series: pd.Series, model_name: str, steps=7, exog=None):
    s = ensure_datetime_freq(series)
    last_dt = s.index[-1]

    if model_name == "arima":
        m = ARIMA(s, order=(2,1,0)).fit()
        fc = m.forecast(steps=steps)

    elif model_name == "sarima":
        m = SARIMAX(s, order=(2,1,0), seasonal_order=(1,1,0,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)

    elif model_name == "sarimax":
        if exog is None:
            raise ValueError("SARIMAX requires exogenous variable (exog).")
        exog = ensure_datetime_freq(exog)
        # จัด exog ให้ align กับส่วนนำ และสร้างอนาคตแบบ naive ให้ยาวเท่า steps
        exog_hist = exog.reindex(s.index).ffill()
        exog_future = _build_exog_future_like_last(exog_hist, last_dt, steps)
        m = SARIMAX(s, order=(2,1,0), seasonal_order=(1,1,0,12),
                    exog=exog_hist,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps, exog=exog_future)

    elif model_name == "lstm":
        vals = lstm_forecast_simple(s, steps)
        fc = pd.Series(vals, index=to_bday_future_index(last_dt, steps))
        return fc

    else:
        raise ValueError("Unknown model")

    fc = pd.Series(fc.values, index=to_bday_future_index(last_dt, steps))
    return fc


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
    """
    สร้าง/อัปเดต forecast สำหรับแต่ละสัญลักษณ์-โมเดล-ช่วงเวลา (steps)
    """
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

                        # --- Exog สำหรับ SARIMAX ---
                        exog = None
                        if m == "sarimax":
                            oil = yf.download("CL=F", period=period, progress=False, auto_adjust=True)['Close'].dropna()
                            oil = ensure_datetime_freq(oil)
                            oil = oil.reindex(data.index).ffill()
                            exog = oil

                        # Backtest + forecast
                        back_fc, back_mae = backtest_last_n_days(data, model_name=m, steps=steps, exog=exog) # backtest ใช้ 7 วันคงที่
                        backtest_json = series_to_chart_pairs_safe(back_fc)

                        fut_fc = future_forecast(data, model_name=m, steps=steps, exog=exog)
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
