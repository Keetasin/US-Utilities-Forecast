import requests
from datetime import datetime
from ..models import StockNews
from .. import db
from .stock import TICKERS
from datetime import datetime, timedelta
import pytz

NEWS_API_KEY = "API_KEY" # Replace with your actual NewsAPI key

TICKERS = {
    "AEP": "American Electric Power",
    "DUK": "Duke Energy",
    "SO": "Southern Company",
    "ED": "Consolidated Edison",
    "EXC": "Exelon"
}

def fetch_news(ticker, api_key=NEWS_API_KEY):
    company_name = TICKERS[ticker]
    query = f'"{ticker}" AND "{company_name}"'  
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    
    try:
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        filtered = []
        for a in articles:
            text = (a.get("title") or "") + " " + (a.get("description") or "")
            if ticker in text or company_name.lower() in text.lower():
                filtered.append({"text": text, "published": a.get("publishedAt"), "url": a.get("url")})
        return filtered
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def update_stock_news(app, symbols):
    with app.app_context():
        for symbol in symbols:
            news_list = fetch_news(symbol)
            sn = StockNews.query.filter_by(symbol=symbol).first()
            if not sn:
                sn = StockNews(symbol=symbol, news_json=news_list, updated_at=datetime.utcnow()) 
                db.session.add(sn)
            else:
                sn.news_json = news_list
                sn.updated_at = datetime.utcnow()
        db.session.commit()
        print(f"[{datetime.utcnow()}] Stock news updated ✅")
        

def initialize_news(app):
    """Initialize news: fetch missing tickers OR update if last update before last cutoff (19:00 TH time)"""

    tz_th = pytz.timezone("Asia/Bangkok")
    now_th = datetime.now(tz_th)

    today_19 = now_th.replace(hour=19, minute=0, second=0, microsecond=0)
    if now_th < today_19:
        cutoff = today_19 - timedelta(days=1)
    else:
        cutoff = today_19

    with app.app_context():
        for t in TICKERS:
            sn = StockNews.query.filter_by(symbol=t).first()
            if not sn:
                print(f"DB missing news for {t}. Fetching initial stock news...")
                update_stock_news(app, [t])
            else:
                last_update_th = sn.updated_at.replace(tzinfo=pytz.UTC).astimezone(tz_th)
                if last_update_th < cutoff:
                    print(f"News for {t} is outdated (last updated {last_update_th}). Updating...")
                    update_stock_news(app, [t])
                else:
                    print(f"News for {t} already up-to-date.")