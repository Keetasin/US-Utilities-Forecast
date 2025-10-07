import yfinance as yf
from datetime import datetime, time, timezone
import pytz
from ..models import Stock
from .. import db


# TICKERS = ["AEP", "DUK", "SO", "ED", "EXC"]
TICKERS = ["AEP"]

tz_ny = pytz.timezone("America/New_York")
tz_th = pytz.timezone("Asia/Bangkok")

def fetch_and_update_stock(ticker):
    stock = yf.Ticker(ticker)
    info = stock.history(period="1d", interval="1m", auto_adjust=True)
    if len(info) >= 1:
        close = float(info["Close"].iloc[-1]) 
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
        return float(round(close, 2)), float(round(change_pct, 2)), int(cap) if cap else None, bg_color
    return None, None, None, None


def update_stock_data(app, force=False):
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    now_ny = now_utc.astimezone(tz_ny)
    
    is_weekday = now_ny.weekday() < 5
    in_market_hours = market_open <= now_ny.time() <= market_close
    
    if not force:
        if not is_weekday:
            print(f"[{now_ny}] Today is weekend. Market closed.")
            return
        if not in_market_hours:
            print(f"[{now_ny}] Outside market hours ({market_open}-{market_close} ET). Stock update skipped.")
            return

    with app.app_context():
        for t in TICKERS:
            price, change, cap, bg_color = fetch_and_update_stock(t)
            if price is not None:
                s = Stock.query.filter_by(symbol=t).first()
                if not s:
                    s = Stock(
                                symbol=t,
                                price=float(price),
                                change=float(change),
                                marketCap=int(cap),
                                bg_color=bg_color,
                                last_updated=now_utc
                            )
                    db.session.add(s)
                else:
                    s.price = float(price)
                    s.change = float(change)
                    s.marketCap = int(cap) if cap else None
                    s.bg_color = bg_color
                    s.last_updated = now_utc
        db.session.commit()
        print(f"[{datetime.now(tz_th)}] Stock data updated ✅")

        
def initialize_stocks(app):
    with app.app_context():
        if Stock.query.count() == 0:
            print("DB empty. Fetching initial stock data...")
            update_stock_data(app, force=True)
        else:
            print("DB already has stock data.")
