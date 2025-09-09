from flask import Blueprint, render_template, request, redirect, url_for
from . import db
from .models import Stock
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time, timedelta
import pytz
import numpy as np
import pandas as pd

# Statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# Keras (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

views = Blueprint('views', __name__)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

# ==========================================================
# Utils: Data fetch & scheduler (ของเดิม)
# ==========================================================

def fetch_and_update_stock(t):
    stock = yf.Ticker(t)
    info = stock.history(period="1d", interval="1m", auto_adjust=True)
    if len(info) >= 1:
        close = info["Close"].iloc[-1]
        prev_close = stock.info.get("previousClose", None)
        if prev_close:
            change_pct = ((close - prev_close) / prev_close) * 100
        else:
            open_price = info["Open"].iloc[0]
            change_pct = ((close - open_price) / open_price) * 100
        cap = stock.info.get("marketCap", 0)
        intensity = min(abs(change_pct), 3) / 3
        lightness = 50 - intensity * 30
        hue = 120 if change_pct >= 0 else 0
        bg_color = f"hsl({hue}, 80%, {lightness}%)"
        print(f"[{t}] Price={close:.2f}, Change={change_pct:.2f}%")
        return round(close, 2), round(change_pct, 2), cap, bg_color
    return None, None, None, None


def update_stock_data(app, force=False):
    with app.app_context():
        tz = pytz.timezone("US/Eastern")
        now = datetime.now(tz)
        market_open = time(9, 30)
        market_close = time(16, 0)
        update_allowed = force or (market_open <= now.time() <= market_close)

        for t in TICKERS:
            s = Stock.query.filter_by(symbol=t).first()
            if not s:
                s = Stock(symbol=t, price=0.0, change=0.0, marketCap=0, bg_color="hsl(0,0%,50%)")
                db.session.add(s)
                db.session.commit()
            if update_allowed:
                price, change, cap, bg_color = fetch_and_update_stock(t)
                if price is not None:
                    s.price = price
                    s.change = change
                    s.marketCap = cap
                    s.bg_color = bg_color

        db.session.commit()
        print(f"[{now}] Stock data updated (force={force})")


def initialize_stocks(app):
    with app.app_context():
        if Stock.query.count() == 0:
            print("DB empty. Fetching initial stock data...")
            update_stock_data(app, force=True)
        else:
            print("DB already has stock data.")

scheduler = BackgroundScheduler()

def start_scheduler(app):
    scheduler.add_job(func=lambda: update_stock_data(app, force=True), trigger="interval", minutes=1)
    scheduler.start()
    print("Scheduler started ✅")

# ==========================================================
# New: Model helpers
# ==========================================================

def get_period_by_model(model_name: str) -> str:
    """
    กำหนดช่วงเวลา (period) ที่เหมาะกับแต่ละโมเดล
    ARIMA: short-term → 6 เดือน
    SARIMA: medium-term → 3 ปี
    LSTM: long-term → 10 ปี
    """
    model_name = (model_name or "arima").lower()
    if model_name == "arima":
        return "6mo"
    if model_name == "sarima":
        return "3y"
    if model_name == "lstm":
        return "10y"
    return "6mo"

def backtest_last_n_days(series: pd.Series, model_name: str, steps: int = 7):
    s = ensure_datetime_freq(series)
    train_data = s.iloc[:-steps]
    true_future = s.iloc[-steps:]

    if model_name == "arima":
        m = ARIMA(train_data, order=(5,1,0)).fit()
        fc = m.forecast(steps=steps)
    elif model_name == "sarima":
        m = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,20),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)
    elif model_name == "lstm":
        fc = lstm_forecast_simple(train_data, steps=steps)
        fc = pd.Series(fc, index=true_future.index)
    else:
        raise ValueError("Unknown model for backtest")

    # บังคับ index ของ forecast ให้ตรงกับจริง (กัน integer index)
    fc = pd.Series(np.asarray(fc).ravel(), index=true_future.index)

    mae = mean_absolute_error(true_future.values, fc.values)
    return fc, mae


def future_forecast(series: pd.Series, model_name: str, steps: int = 7):
    s = ensure_datetime_freq(series)
    last_dt = s.index[-1]

    if model_name == "arima":
        m = ARIMA(s, order=(5,1,0)).fit()
        fc = m.forecast(steps=steps)
    elif model_name == "sarima":
        m = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,20),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.forecast(steps=steps)
    elif model_name == "lstm":
        vals = lstm_forecast_simple(s, steps=steps)
        future_idx = to_bday_future_index(last_dt, steps)  # สร้าง index วันทำการ
        fc = pd.Series(vals, index=future_idx)
        return fc
    else:
        raise ValueError("Unknown model for future forecast")

    # ถึงแม้ statsmodels อาจให้ datetime index ถ้า freq หาย เราบังคับเองให้ชัวร์
    future_idx = to_bday_future_index(last_dt, steps)
    fc = pd.Series(np.asarray(fc).ravel(), index=future_idx)
    return fc


def lstm_forecast_simple(series: pd.Series, steps: int = 7, lookback: int = 60, epochs: int = 15, batch_size: int = 32):
    """
    LSTM แบบเบา ๆ สำหรับหน้าเว็บ:
    - ใช้ Close อย่างเดียว
    - scale 0-1
    - ทำ sequence ตาม lookback
    - train/val split 80/20
    - forecast แบบ recursive steps วัน
    *หมายเหตุ: ใช้งานในเว็บ อาจช้าบนเครื่องเล็ก — แนะนำเปิด GPU ถ้ามี
    """
    values = series.values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # สร้างลำดับ
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:  # สั้นเกินไป
        # fallback: ทำนายคงที่
        last_val = float(values[-1, 0])
        return np.array([last_val] * steps, dtype=np.float32)

    # split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # reshape
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # model
    tf.keras.backend.clear_session()
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.1),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)

    # recursive forecast
    last_window = scaled[-lookback:, 0].copy()
    preds = []
    for _ in range(steps):
        x = last_window.reshape(1, lookback, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        last_window = np.roll(last_window, -1)
        last_window[-1] = p

    preds = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(preds).ravel()
    return inv
def to_bday_future_index(last_dt: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """สร้าง index อนาคตแบบวันทำการต่อจากวันสุดท้าย"""
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=steps)

def series_to_chart_pairs_safe(series: pd.Series):
    """
    ปลอดภัยกว่าของเดิม: ถ้า index ไม่ใช่ datetime จะ fallback เป็นลำดับวันทำการย้อนหลัง
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        # fallback: สร้าง index วันทำการย้อนหลังเท่าความยาว
        end = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(end=end, periods=len(s))
        s.index = idx

    return [{"date": d.strftime('%Y-%m-%d'), "price": round(float(p), 2)} for d, p in zip(s.index, s.values)]

# ==========================================================
# Routes
# ==========================================================

@views.route('/')
def home():
    return render_template('home.html')

@views.route('/heatmap')
def heatmap():
    data = Stock.query.order_by(Stock.marketCap.desc()).all()
    return render_template("heatmap.html", data=data)

@views.route('/stock/<symbol>')
def stock_detail(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock:
        return "Stock not found", 404
    return render_template("stock_detail.html", stock=stock)

@views.route('/forecasting/<symbol>')
def forecasting(symbol):
    model = (request.args.get("model") or "arima").lower()
    try:
        period = get_period_by_model(model)
        full_close = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close']
        full_close = full_close.dropna()
        full_close = ensure_datetime_freq(full_close)

        if len(full_close) < 70:
            # แจ้ง error บนหน้าเดียวกัน (ไม่มี tickers/drpdown)
            return render_template(
                "forecasting.html",
                has_data=False,
                symbol=symbol,
                model=model.upper(),
                error=f"Not enough data ({len(full_close)} rows) for {model.upper()} with period={period}"
            )

        # Backtest และ Forecast เหมือนเดิม
        back_fc, back_mae = backtest_last_n_days(full_close, model_name=model, steps=7)
        future_fc = future_forecast(full_close, model_name=model, steps=7)
        hist = full_close.tail(90)

        # payload สำหรับกราฟ (ใช้ safe pair ป้องกัน index ไม่ใช่ datetime)
        historical_data = series_to_chart_pairs_safe(hist)
        backtest_data   = series_to_chart_pairs_safe(back_fc)
        future_data     = series_to_chart_pairs_safe(future_fc)

        last_hist      = float(np.asarray(hist.iloc[-1]).ravel()[0])
        last_forecast  = float(np.asarray(future_fc.iloc[-1]).ravel()[0])

        trend_direction = "Up" if last_forecast > last_hist else "Down"
        trend_icon      = "🔼" if trend_direction == "Up" else "🔽"

        return render_template(
            "forecasting.html",
            has_data=True,
            symbol=symbol,
            model=model.upper(),
            forecast=future_data,
            historical=historical_data,
            backtest=backtest_data,
            backtest_mae=round(float(back_mae), 4),
            trend={"direction": trend_direction, "icon": trend_icon},
            last_price=round(last_hist, 2),
        )

    except Exception as e:
        print(f"Forecast error for {symbol}: {e}")
        return render_template(
            "forecasting.html",
            has_data=False,
            symbol=symbol,
            model=model.upper(),
            error=str(e)
        )

@views.route('/compare/<symbol>')
def compare_models(symbol):
    results = {}
    historical = None
    last_price = None
    models = ["arima", "sarima", "lstm"]

    try:
        long_close = yf.download(symbol, period="10y", progress=False, auto_adjust=True)['Close'].dropna()
        long_close = ensure_datetime_freq(long_close)
        historical = series_to_chart_pairs_safe(long_close.tail(120))
        last_price = round(float(long_close.iloc[-1]), 2)
    except Exception as e:
        print("Error fetching long history:", e)

    for m in models:
        try:
            period = get_period_by_model(m)
            s = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
            s = ensure_datetime_freq(s)

            if len(s) < 70:
                results[m] = {"ok": False, "error": f"Not enough data ({len(s)}) for {m.upper()} ({period})"}
                continue

            back_fc, back_mae = backtest_last_n_days(s, model_name=m, steps=7)
            fut_fc = future_forecast(s, model_name=m, steps=7)

            results[m] = {
                "ok": True,
                "model": m.upper(),
                "period": period,
                "backtest_mae": round(float(back_mae), 4),
                "forecast": series_to_chart_pairs_safe(fut_fc),
                "forecast_last": round(float(np.asarray(fut_fc.iloc[-1]).ravel()[0]), 2)
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            results[m] = {"ok": False, "error": str(e)}

    best_model, best_mae = None, None
    for m in models:
        if results.get(m, {}).get("ok"):
            mae = results[m]["backtest_mae"]
            if (best_mae is None) or (mae < best_mae):
                best_mae, best_model = mae, m.upper()

    return render_template(
        "compare.html",
        symbol=symbol,
        historical=historical,
        last_price=last_price,
        results=results,
        best_model=best_model,
        best_mae=best_mae
    )

def ensure_datetime_freq(series: pd.Series, use_bdays: bool = True) -> pd.Series:
    """
    ทำให้ series:
    - มี DatetimeIndex (ถ้าไม่ใช่ จะพยายามแปลง)
    - มี freq เสมอ (พยายาม infer; ถ้าไม่ได้ จะ reindex เป็น BusinessDay และ ffill)
    """
    s = series.copy()

    # ให้แน่ใจว่าเป็น DatetimeIndex
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")

    # ลบ NaT index ที่แปลงไม่ได้
    s = s[~s.index.isna()].sort_index()

    # พยายามเดา freq
    inferred = pd.infer_freq(s.index)
    if inferred is None:
        if use_bdays:
            # Reindex เป็นวันทำการ
            full_idx = pd.bdate_range(start=s.index.min(), end=s.index.max())
        else:
            # ใช้ daily ธรรมดา
            full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")

        s = s.reindex(full_idx).ffill()

    return s
