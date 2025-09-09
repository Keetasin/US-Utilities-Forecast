# web/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from .news_utils import update_stock_news
from .stock_utils import update_stock_data, TICKERS
import pytz

# ---------------------------
# Stock price scheduler
# ---------------------------
stock_scheduler = BackgroundScheduler()

def start_stock_scheduler(app):
    # รันทุก 1 นาที
    stock_scheduler.add_job(
        func=lambda: update_stock_data(app, force=True),
        trigger="interval",
        minutes=1
    )
    stock_scheduler.start()
    print("Stock scheduler started ✅")

# ---------------------------
# News scheduler
# ---------------------------
news_scheduler = BackgroundScheduler()

def start_news_scheduler(app):
    # รันทุกวัน 20:00 Bangkok
    tz_th = pytz.timezone("Asia/Bangkok")
    news_scheduler.add_job(
        func=lambda: update_stock_news(app, TICKERS),
        trigger="cron",
        hour=20,
        minute=0,
        timezone=tz_th
    )
    news_scheduler.start()
    print("News scheduler started ✅")
