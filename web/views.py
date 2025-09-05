from flask import Blueprint, render_template, current_app
from . import db
from .models import Stock
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time, timedelta
import pytz
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

views = Blueprint('views', __name__)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

# ... (ส่วน fetch_and_update_stock, update_stock_data, initialize_stocks ไม่มีการเปลี่ยนแปลง) ...
def fetch_and_update_stock(t):
    """ดึงข้อมูลหุ้นจาก yfinance และอัปเดต Stock object"""
    stock = yf.Ticker(t)
    info = stock.history(period="5d")
    if len(info) >= 2:
        close = info["Close"].iloc[-1]
        prev_close = info["Close"].iloc[-2]
        change_pct = ((close - prev_close) / prev_close) * 100
        cap = stock.info.get("marketCap", 0)

        intensity = min(abs(change_pct), 3) / 3
        lightness = 50 - intensity * 30
        hue = 120 if change_pct >= 0 else 0
        bg_color = f"hsl({hue}, 80%, {lightness}%)"

        return round(close, 2), round(change_pct, 2), cap, bg_color
    return None, None, None, None


def update_stock_data(force=False):
    """อัปเดตข้อมูลหุ้น"""
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    market_open = time(9, 30)
    market_close = time(16, 0)

    update_allowed = market_open <= now.time() <= market_close or force

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
    """ตรวจสอบ DB ตอนรันครั้งแรก ถ้าว่าง ให้ fetch ข้อมูลทันที"""
    with app.app_context():
        if Stock.query.count() == 0:
            print("DB empty. Fetching initial stock data...")
            update_stock_data(force=True)
        else:
            print("DB already has stock data.")

# -----------------------------
# Start scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_stock_data, trigger="interval", minutes=1)
scheduler.start()

# -----------------------------
# Routes
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


@views.route('/forecasting')
@views.route('/forecasting/<symbol>')
def forecasting(symbol=None):
    if not symbol:
        return render_template("forecasting.html", has_data=False, tickers=TICKERS)

    try:
        print(f"Starting forecast for {symbol}...")
        
        full_hist_data = yf.download(symbol, period="2y", progress=False)['Close']
        
        # --- START: Backtesting Logic ---
        print("Performing backtest...")
        train_data = full_hist_data[:-7]
        
        backtest_model = ARIMA(train_data, order=(5, 1, 0))
        backtest_model_fit = backtest_model.fit()
        backtest_forecast_result = backtest_model_fit.forecast(steps=7)
        actual_data_for_backtest = full_hist_data[-7:]
        
        # 2. คำนวณค่า Mean Absolute Error
        backtest_mae = mean_absolute_error(actual_data_for_backtest.values, backtest_forecast_result.values)
        
        # --- START OF FIX ---
        # สร้างวันที่สำหรับช่วง Backtest (คือวันที่ 7 วันสุดท้ายของข้อมูลทั้งหมด)
        backtest_dates = full_hist_data.index[-7:]
        
        backtest_data = [
            {"date": date.strftime('%Y-%m-%d'), "price": round(float(price), 2)}
            for date, price in zip(backtest_dates, backtest_forecast_result.values)
        ]
        # --- END OF FIX ---
        # --- END: Backtesting Logic ---

        # --- START: Future Forecast Logic ---
        print(f"Training final model for future forecast...")
        final_model = ARIMA(full_hist_data, order=(5, 1, 0))
        final_model_fit = final_model.fit()
        future_forecast_result = final_model_fit.forecast(steps=7)
        
        last_date = full_hist_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        future_forecast_data = [
            {"date": date.strftime('%Y-%m-%d'), "price": round(float(price), 2)}
            for date, price in zip(future_dates, future_forecast_result.values)
        ]
        # --- END: Future Forecast Logic ---

        hist_data_for_chart = full_hist_data.tail(90)
        historical_data = [
            {"date": date.strftime('%Y-%m-%d'), "price": round(float(price), 2)}
            for date, price in zip(hist_data_for_chart.index, hist_data_for_chart.values)
        ]
        
        trend_direction = "Up" if future_forecast_data[-1]['price'] > historical_data[-1]['price'] else "Down"
        trend_icon = "🔼" if trend_direction == "Up" else "🔽"
        trend_info = {"direction": trend_direction, "icon": trend_icon}

        print(f"Forecast for {symbol} successful.")
        return render_template("forecasting.html",
                               symbol=symbol,
                               forecast=future_forecast_data,
                               historical=historical_data,
                               backtest=backtest_data,
                               backtest_mae=backtest_mae,
                               has_data=True,
                               tickers=TICKERS,
                               trend=trend_info)
                               
    except Exception as e:
        print(f"An error occurred during forecast for {symbol}: {e}")
        return str(e), 500