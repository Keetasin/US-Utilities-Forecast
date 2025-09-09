import requests
from datetime import datetime
from .models import StockNews
from . import db
from .stock_utils import TICKERS


NEWS_API_KEY = "bfa14549cdc84faf88c40e947da98d4d"

def fetch_news(query, api_key=NEWS_API_KEY):
    url=f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        return [{"text": (a.get("title") or "") + " " + (a.get("description") or ""),
                 "published": a.get("publishedAt"), "url": a.get("url")} for a in articles]
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []

def summarize_news_for_investor(news_list):
    if not news_list: return "No recent news."
    combined_news = " ".join([n["text"] for n in news_list[:5]])
    prompt=f"Summarize the following news for an investor in 2-3 sentences: {combined_news}"
    try:
        resp = requests.post("http://localhost:11434/v1/chat/completions",
                             json={"model":"llama3","messages":[{"role":"user","content":prompt}]})
        if resp.status_code != 200:
            return "Failed to generate summary."
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return "Failed to generate summary."

def update_stock_news(app, symbols):
    """Update news + summary daily"""
    from datetime import datetime
    with app.app_context():
        for symbol in symbols:
            news_list = fetch_news(symbol)
            summary = summarize_news_for_investor(news_list)
            
            sn = StockNews.query.filter_by(symbol=symbol).first()
            if not sn:
                sn = StockNews(symbol=symbol, news_json=news_list, summary=summary, updated_at=datetime.utcnow())
                db.session.add(sn)
            else:
                sn.news_json = news_list
                sn.summary = summary
                sn.updated_at = datetime.utcnow()
        db.session.commit()
        print(f"[{datetime.utcnow()}] Stock news updated ✅")

def initialize_news(app):
    with app.app_context():
        # ตรวจสอบว่ามีข่าวของ ticker ครบหรือไม่
        missing = [t for t in TICKERS if not StockNews.query.filter_by(symbol=t).first()]
        if missing:
            print(f"DB missing news for: {missing}. Fetching initial stock news...")
            update_stock_news(app, missing)
        else:
            print("DB already has stock news for all tickers.")
