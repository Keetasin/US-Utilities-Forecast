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
        now_utc = datetime.now(pytz.UTC)
        market_open = time(9,30)
        market_close = time(16,0)
        update_allowed = force or (market_open <= now_utc.time() <= market_close)

        for t in TICKERS:
            # First, fetch the data
            price, change, cap, bg_color = fetch_and_update_stock(t)
            
            # Only proceed if data was successfully fetched
            if price is not None:
                # Check if the stock already exists in the database
                s = Stock.query.filter_by(symbol=t).first()
                if not s:
                    # If it doesn't exist, create a new entry with the fetched data
                    s = Stock(
                        symbol=t,
                        price=price,
                        change=change,
                        marketCap=cap,
                        bg_color=bg_color,
                        last_updated=now_utc
                    )
                    db.session.add(s)
                else:
                    # If it exists, update its attributes
                    s.price = price
                    s.change = change
                    s.marketCap = cap
                    s.bg_color = bg_color
                    s.last_updated = now_utc
        
        # Commit all changes to the database at once after the loop
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
