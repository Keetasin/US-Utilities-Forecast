from apscheduler.schedulers.background import BackgroundScheduler
from .utils.news import update_stock_news
from .utils.stock import update_stock_data, TICKERS
from .utils.forecast import update_forecast
from datetime import datetime, time, timedelta
import pytz

tz_th = pytz.timezone("Asia/Bangkok")

# ---------------------------
# Stock Scheduler
# ---------------------------
stock_scheduler = BackgroundScheduler()

def start_stock_scheduler(app):
    with app.app_context():
        # 1) ตรวจสอบ DB ว่างหรือไม่
        from .models import Stock
        now_th = datetime.now(tz_th)
        market_open = time(9,30)
        market_close = time(16,0)

        if Stock.query.count() == 0:
            print("Stock DB empty -> Fetch latest data")
            update_stock_data(app, force=True)
        else:
            latest = Stock.query.order_by(Stock.last_updated.desc()).first()
            if latest:
                last_update_th = latest.last_updated.replace(tzinfo=pytz.UTC).astimezone(tz_th)
                if last_update_th.date() < now_th.date():
                    print("Stock data outdated -> Fetch once")
                    update_stock_data(app, force=True)

    # 2) อยู่ในช่วงตลาดเปิด -> run ทุก 1 นาที
    stock_scheduler.add_job(
        func=lambda: update_stock_data(app),
        trigger="interval",
        minutes=1
    )
    stock_scheduler.start()
    print("Stock scheduler started ✅")


# ---------------------------
# News Scheduler
# ---------------------------
news_scheduler = BackgroundScheduler()

def start_news_scheduler(app):
    with app.app_context():
        from .models import StockNews
        now_th = datetime.now(tz_th)
        today_20 = now_th.replace(hour=20, minute=0, second=0, microsecond=0)
        cutoff = today_20 if now_th >= today_20 else today_20 - timedelta(days=1)

        for t in TICKERS:
            sn = StockNews.query.filter_by(symbol=t).first()
            if not sn:
                print(f"News DB empty -> Fetch latest news for {t}")
                update_stock_news(app, [t])
            else:
                last_update_th = sn.updated_at.replace(tzinfo=pytz.UTC).astimezone(tz_th)
                if last_update_th < cutoff:
                    print(f"News outdated -> Fetch once for {t}")
                    update_stock_news(app, [t])

    # Schedule daily at 20:00
    news_scheduler.add_job(
        func=lambda: update_stock_news(app, TICKERS),
        trigger="cron",
        hour=20,
        minute=0,
        timezone=tz_th
    )
    news_scheduler.start()
    print("News scheduler started ✅")


# ---------------------------
# Forecast Scheduler
# ---------------------------
forecast_scheduler = BackgroundScheduler()

def start_forecast_scheduler(app):
    from .models import StockForecast
    with app.app_context():
        now_th = datetime.now(tz_th)
        today_10 = now_th.replace(hour=10, minute=0, second=0, microsecond=0)
        cutoff = today_10 if now_th >= today_10 else today_10 - timedelta(days=1)

        # ✅ เพิ่ม SARIMAX เข้ามา
        models = ["arima", "sarima", "sarimax", "lstm"]

        # Fetch latest forecast if DB empty or outdated
        for t in TICKERS:
            for m in models:
                fc = StockForecast.query.filter_by(symbol=t, model=m).first()
                if not fc:
                    print(f"Forecast DB empty -> Update {t}-{m}")
                    update_forecast(app, [t], models=[m])   # ระบุโมเดลที่ต้องการชัดเจน
                else:
                    last_update_th = fc.updated_at.replace(tzinfo=pytz.UTC).astimezone(tz_th)
                    if last_update_th < cutoff:
                        print(f"Forecast outdated -> Update {t}-{m}")
                        update_forecast(app, [t], models=[m])

    # Schedule daily at 10:00 (อัปเดตครบทุกโมเดลรวม SARIMAX)
    forecast_scheduler.add_job(
        func=lambda: update_forecast(app, TICKERS, models=["arima", "sarima", "sarimax", "lstm"]),
        trigger="cron",
        hour=10,
        minute=0,
        timezone=tz_th
    )
    forecast_scheduler.start()
    print("Forecast scheduler started ✅")
