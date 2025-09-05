from flask import Blueprint, render_template, current_app
from . import db
from .models import Stock
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time
import pytz

views = Blueprint('views', __name__)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

# -----------------------------
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
            # สร้าง object ถ้ายังไม่มี พร้อมค่า default ป้องกัน NOT NULL
            s = Stock(
                symbol=t,
                price=0.0,
                change=0.0,
                marketCap=0,
                bg_color="hsl(0,0%,50%)"
            )
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
        count = Stock.query.count()
        if count == 0:
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

@views.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')

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
