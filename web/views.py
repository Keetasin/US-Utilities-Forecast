from flask import Blueprint, render_template, request
from . import db
from .models import Stock
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time
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
# Utils
# ==========================================================

def to_scalar(x) -> float:
    """Force any Series/ndarray/np scalar into a Python float scalar safely."""
    a = np.asarray(x)
    return float(a.reshape(-1)[0])

def fetch_and_update_stock(t):
    stock = yf.Ticker(t)
    info = stock.history(period="1d", interval="1m", auto_adjust=True)
    if len(info) >= 1:
        close = info["Close"].iloc[-1]
        prev_close = stock.info.get("previousClose", None)
        if prev_close is not None and not pd.isna(prev_close) and prev_close != 0:
            change_pct = ((close - prev_close) / prev_close) * 100
        else:
            open_price = info["Open"].iloc[0]
            if open_price and not pd.isna(open_price) and open_price != 0:
                change_pct = ((close - open_price) / open_price) * 100
            else:
                change_pct = 0.0
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
                s = Stock(symbol=t, price=0.0, change=0.0,
                          marketCap=0, bg_color="hsl(0,0%,50%)")
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
    scheduler.add_job(func=lambda: update_stock_data(app, force=True),
                      trigger="interval", minutes=1)
    scheduler.start()
    print("Scheduler started ✅")

# ==========================================================
# Model helpers
# ==========================================================

def get_period_by_model(model_name: str) -> str:
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
        future_idx = to_bday_future_index(last_dt, steps)
        fc = pd.Series(vals, index=future_idx)
        return fc
    else:
        raise ValueError("Unknown model for future forecast")

    future_idx = to_bday_future_index(last_dt, steps)
    fc = pd.Series(np.asarray(fc).ravel(), index=future_idx)
    return fc

def lstm_forecast_simple(series: pd.Series, steps: int = 7, lookback: int = 60,
                         epochs: int = 15, batch_size: int = 32):
    values = series.values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        last_val = float(values[-1, 0])
        return np.array([last_val] * steps, dtype=np.float32)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

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
    start = last_dt + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=steps)

def series_to_chart_pairs_safe(series: pd.Series):
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        end = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(end=end, periods=len(s))
        s.index = idx
    return [{"date": d.strftime('%Y-%m-%d'), "price": round(to_scalar(p), 2)}
            for d, p in zip(s.index, s.values)]

def ensure_datetime_freq(series: pd.Series, use_bdays: bool = True) -> pd.Series:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    inferred = pd.infer_freq(s.index)
    if inferred is None:
        if use_bdays:
            full_idx = pd.bdate_range(start=s.index.min(), end=s.index.max())
        else:
            full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
        s = s.reindex(full_idx).ffill()
    return s

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
            return render_template(
                "forecasting.html",
                has_data=False,
                symbol=symbol,
                model=model.upper(),
                error=f"Not enough data ({len(full_close)} rows) for {model.upper()} with period={period}"
            )

        back_fc, back_mae = backtest_last_n_days(full_close, model_name=model, steps=7)
        future_fc = future_forecast(full_close, model_name=model, steps=7)
        hist = full_close.tail(90)

        last_price_val = to_scalar(hist.iloc[-1])
        back_mae_pct = (to_scalar(back_mae) / last_price_val * 100.0) if last_price_val else 0.0

        historical_data = series_to_chart_pairs_safe(hist)
        backtest_data = series_to_chart_pairs_safe(back_fc)
        future_data = series_to_chart_pairs_safe(future_fc)

        last_future = to_scalar(future_fc.iloc[-1])
        last_hist = to_scalar(hist.iloc[-1])
        trend_direction = "Up" if last_future > last_hist else "Down"
        trend_icon = "🔼" if trend_direction == "Up" else "🔽"

        return render_template(
            "forecasting.html",
            symbol=symbol,
            model=model.upper(),
            forecast=future_data,
            historical=historical_data,
            backtest=backtest_data,
            backtest_mae=to_scalar(back_mae),
            backtest_mae_pct=back_mae_pct,
            has_data=True,
            trend={"direction": trend_direction, "icon": trend_icon},
            last_price=round(last_price_val, 2)
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
        last_price = round(to_scalar(long_close.iloc[-1]), 2)
    except Exception as e:
        print("Error fetching long history:", e)

    for m in models:
        try:
            period = get_period_by_model(m)
            s = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
            s = ensure_datetime_freq(s)
            if len(s) < 70:
                results[m] = {"ok": False, "error": f"Not enough data ({len(s)} rows) for {m.upper()} period={period}"}
                continue

            back_fc, back_mae = backtest_last_n_days(s, model_name=m, steps=7)
            fut_fc = future_forecast(s, model_name=m, steps=7)

            last_p = to_scalar(s.iloc[-1]) if len(s) else 0.0
            mae_pct = (to_scalar(back_mae) / last_p * 100.0) if last_p else 0.0

            results[m] = {
                "ok": True,
                "period": period,
                "backtest_mae": to_scalar(back_mae),
                "backtest_mae_pct": mae_pct,
                "forecast": series_to_chart_pairs_safe(fut_fc),
                "forecast_last": round(to_scalar(fut_fc.iloc[-1]), 2)
            }
        except Exception as e:
            results[m] = {"ok": False, "error": str(e)}

    best_model, best_mae = None, None
    for m in models:
        if results.get(m, {}).get("ok"):
            mae = to_scalar(results[m]["backtest_mae"])
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
