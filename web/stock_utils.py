import yfinance as yf
from datetime import datetime, time, timezone
import pytz
from .models import Stock
from . import db

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

def fetch_and_update_stock(ticker):
    stock = yf.Ticker(ticker)
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
        print(f"[{ticker}] Price={close:.2f}, Change={change_pct:.2f}%")
        return round(close,2), round(change_pct,2), cap, bg_color
    return None, None, None, None

def update_stock_data(app, force=False):
    with app.app_context():
        now_utc = datetime.now(timezone.utc)
        market_open = time(9,30)
        market_close = time(16,0)
        update_allowed = force or (market_open <= now_utc.time() <= market_close)

        for t in TICKERS:
            s = Stock.query.filter_by(symbol=t).first()
            if not s:
                s = Stock(symbol=t)
                db.session.add(s)
                db.session.commit()
            if update_allowed:
                price, change, cap, bg_color = fetch_and_update_stock(t)
                if price is not None:
                    s.price = price
                    s.change = change
                    s.marketCap = cap
                    s.bg_color = bg_color
                    s.last_updated = now_utc
        db.session.commit()
        tz_th = pytz.timezone("Asia/Bangkok")
        now_th = now_utc.astimezone(tz_th)
        print(f"[{now_th.strftime('%Y-%m-%d %H:%M:%S %Z')}] Stock data updated (force={force})")

def initialize_stocks(app):
    with app.app_context():
        if Stock.query.count() == 0:
            print("DB empty. Fetching initial stock data...")
            update_stock_data(app, force=True)
        else:
            print("DB already has stock data.")
