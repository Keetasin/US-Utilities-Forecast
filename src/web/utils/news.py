import requests
from datetime import datetime
from ..models import StockNews
from .. import db
from .stock import TICKERS


NEWS_API_KEY = "bfa14549cdc84faf88c40e947da98d4d"

TICKERS = {
    "AEP": "American Electric Power",
    "DUK": "Duke Energy",
    "SO": "Southern Company",
    "ED": "Consolidated Edison",
    "EXC": "Exelon"
}

def fetch_news(ticker, api_key=NEWS_API_KEY):
    company_name = TICKERS[ticker]
    query = f'"{ticker}" OR "{company_name}"'  
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

# def fetch_news(query, api_key=NEWS_API_KEY):
#     url=f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
#     try:
#         resp = requests.get(url).json()
#         articles = resp.get("articles", [])
#         return [{"text": (a.get("title") or "") + " " + (a.get("description") or ""),
#                  "published": a.get("publishedAt"), "url": a.get("url")} for a in articles]
#     except Exception as e:
#         print(f"Error fetching news for {query}: {e}")
#         return []

# def summarize_news_for_investor(news_list):
#     if not news_list: return "No recent news."
#     combined_news = " ".join([n["text"] for n in news_list[:5]])
#     prompt=f"Summarize the following news for an investor in 2-3 sentences: {combined_news}"
#     try:
#         # resp = requests.post("http://localhost:11434/v1/chat/completions",
#         #                      json={"model":"llama3","messages":[{"role":"user","content":prompt}]})
#         resp = requests.post(
#             "http://ollama:11434/v1/chat/completions",
#             json={"model":"llama3","messages":[{"role":"user","content":prompt}]}
#         )
#         if resp.status_code != 200:
#             return "Failed to generate summary."
#         return resp.json()['choices'][0]['message']['content']
#     except Exception as e:
#         print(f"Error calling Ollama API: {e}")
#         return "Failed to generate summary."


# def summarize_news_for_investor(news_list):
#     if not news_list:
#         return "No recent news."
    
#     combined_news = " ".join([n["text"] for n in news_list[:5]])
#     prompt = f"Summarize the following news for an investor in 2-3 sentences: {combined_news}"
    
#     try:
#         # เรียก Ollama ผ่านชื่อ service ของ Docker Compose
#         resp = requests.post(
#             "http://ollama:11434/v1/chat/completions",
#             json={"model":"llama3","messages":[{"role":"user","content":prompt}]}
#         )
#         if resp.status_code != 200:
#             print(f"Ollama response error: {resp.status_code}, {resp.text}")
#             return "Failed to generate summary."
        
#         return resp.json()['choices'][0]['message']['content']
    
#     except Exception as e:
#         print(f"Error calling Ollama API: {e}")
#         return "Failed to generate summary."



def update_stock_news(app, symbols):
    with app.app_context():
        for symbol in symbols:
            news_list = fetch_news(symbol)
            # summary = summarize_news_for_investor(news_list)
            sn = StockNews.query.filter_by(symbol=symbol).first()
            if not sn:
                sn = StockNews(symbol=symbol, news_json=news_list, updated_at=datetime.utcnow()) #, summary=summary
                db.session.add(sn)
            else:
                sn.news_json = news_list
                # sn.summary = summary
                sn.updated_at = datetime.utcnow()
        db.session.commit()
        print(f"[{datetime.utcnow()}] Stock news updated ✅")
        

def initialize_news(app):
    """Initialize news: fetch missing tickers OR update if last update before last cutoff (19:00 TH time)"""
    from datetime import datetime, time, timedelta
    import pytz

    tz_th = pytz.timezone("Asia/Bangkok")
    now_th = datetime.now(tz_th)

    # กำหนด cutoff = 19:00 ของวันนี้ หรือถ้ายังไม่ถึง -> ใช้ 19:00 ของเมื่อวาน
    today_19 = now_th.replace(hour=19, minute=0, second=0, microsecond=0)
    if now_th < today_19:
        cutoff = today_19 - timedelta(days=1)
    else:
        cutoff = today_19

    with app.app_context():
        for t in TICKERS:
            sn = StockNews.query.filter_by(symbol=t).first()
            if not sn:
                # ไม่มีข่าวใน DB => fetch
                print(f"DB missing news for {t}. Fetching initial stock news...")
                update_stock_news(app, [t])
            else:
                # มีข่าวแล้ว แต่อัพเดทล่าสุดก่อน cutoff => update
                last_update_th = sn.updated_at.replace(tzinfo=pytz.UTC).astimezone(tz_th)
                if last_update_th < cutoff:
                    print(f"News for {t} is outdated (last updated {last_update_th}). Updating...")
                    update_stock_news(app, [t])
                else:
                    print(f"News for {t} already up-to-date.")

