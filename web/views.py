from flask import Blueprint, render_template
from . import db
from .models import Stock
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time
import pytz

views = Blueprint('views', __name__)

# -----------------------------
# หุ้นที่จะดึง
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

# -----------------------------
# ฟังก์ชันอัปเดตข้อมูล
def update_stock_data():
    # timezone US Eastern (ตลาดหุ้น NYSE/NASDAQ)
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    market_open = time(9, 30)
    market_close = time(16, 0)

    # อัปเดตเฉพาะช่วงตลาดเปิด
    if market_open <= now.time() <= market_close:
        for t in TICKERS:
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

                s = Stock.query.filter_by(symbol=t).first()
                if not s:
                    s = Stock(symbol=t)
                    db.session.add(s)

                s.price = round(close, 2)
                s.change = round(change_pct, 2)
                s.marketCap = cap
                s.bg_color = bg_color

        db.session.commit()
        print(f"[{now}] Stock data updated.")
    else:
        print(f"[{now}] Market closed. Using latest data.")

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
    # อ่านข้อมูลล่าสุดจาก DB (ไม่ดึง yfinance ใน route)
    data = Stock.query.order_by(Stock.marketCap.desc()).all()
    return render_template("heatmap.html", data=data)


@views.route('/stock/<symbol>')
def stock_detail(symbol):
    # ดึงข้อมูลล่าสุดของหุ้นตัวนั้นจาก DB
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock:
        return "Stock not found", 404

    return render_template("stock_detail.html", stock=stock)
