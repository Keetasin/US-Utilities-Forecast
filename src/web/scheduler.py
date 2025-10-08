from apscheduler.schedulers.background import BackgroundScheduler
from .utils.news import update_stock_news
from .utils.stock import update_stock_data, TICKERS
from .utils.forecast import update_forecast
from datetime import datetime, timedelta
from .models import StockNews, StockForecast
import pytz

tz_th = pytz.timezone("Asia/Bangkok")
tz_ny = pytz.timezone("America/New_York")


stock_scheduler = BackgroundScheduler()

def start_stock_scheduler(app):
    with app.app_context():
        from .models import Stock
        now_th = datetime.now(tz_th)

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

    stock_scheduler.add_job(
        func=lambda: update_stock_data(app),
        trigger='cron',
        day_of_week='mon-fri',
        hour=9,
        minute='30-59',
        timezone=tz_ny,
        max_instances=1,
        misfire_grace_time=30
    )

    stock_scheduler.add_job(
        func=lambda: update_stock_data(app),
        trigger='cron',
        day_of_week='mon-fri',
        hour='10-15',
        minute='*',
        timezone=tz_ny,
        max_instances=1,
        misfire_grace_time=30
    )

    stock_scheduler.start()
    print("Stock scheduler started ✅")


news_scheduler = BackgroundScheduler()

def start_news_scheduler(app):
    with app.app_context():
        now_th = datetime.now(tz_th)
        today_19 = now_th.replace(hour=19, minute=0, second=0, microsecond=0)
        cutoff = today_19 if now_th >= today_19 else today_19 - timedelta(days=1)

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

    news_scheduler.add_job(
        func=lambda: update_stock_news(app, TICKERS),
        trigger="cron",
        day_of_week='mon-fri',
        hour=19,
        minute=00,
        timezone=tz_th,
        max_instances=1,          
        misfire_grace_time=300 
    )
    news_scheduler.start()
    print("News scheduler started ✅")



def start_forecast_scheduler(app):
    with app.app_context():
        models = ["arima", "sarima", "sarimax", "lstm"]
        steps_list = [7, 180, 365]

        for t in TICKERS:
            for m in models:
                for steps in steps_list:
                    fc = StockForecast.query.filter_by(symbol=t, model=m, steps=steps).first()
                    if not fc:
                        print(f"Forecast DB empty -> Update {t}-{m}-{steps}d")
                        update_forecast(app, [t], models=[m], steps_list=[steps])
                    else:
                        print(f"Forecast already exists for {t}-{m}-{steps}d, skipping.")

                        
# forecast_scheduler = BackgroundScheduler()
# def start_forecast_scheduler(app):
#     from .models import StockForecast
#     with app.app_context():
#         now_th = datetime.now(tz_th)
#         today_19_30 = now_th.replace(hour=19, minute=30, second=0, microsecond=0)
#         cutoff = today_19_30 if now_th >= today_19_30 else today_19_30 - timedelta(days=1)

#         models = ["arima", "sarima", "sarimax", "lstm"]
#         steps_list = [7, 180, 365]

#         for t in TICKERS:
#             for m in models:
#                 for steps in steps_list:
#                     fc = StockForecast.query.filter_by(symbol=t, model=m, steps=steps).first()
#                     if not fc:
#                         print(f"Forecast DB empty -> Update {t}-{m}-{steps}d")
#                         update_forecast(app, [t], models=[m], steps_list=[steps])
#                     else:
#                         last_update_th = fc.updated_at.replace(tzinfo=pytz.UTC).astimezone(tz_th)
#                         if last_update_th < cutoff:
#                             print(f"Forecast outdated -> Update {t}-{m}-{steps}d")
#                             update_forecast(app, [t], models=[m], steps_list=[steps])

#     # Schedule daily
#     forecast_scheduler.add_job(
#         func=lambda: update_forecast(app, TICKERS, models=["arima", "sarima", "sarimax", "lstm"], steps_list=[7,180,365]),
#         trigger="cron",
#         day_of_week='mon-fri',
#         hour=19,
#         minute=30,
#         timezone=tz_th,
#         max_instances=1,         
#         misfire_grace_time=300 
#     )
#     forecast_scheduler.start()
#     print("Forecast scheduler started ✅")





