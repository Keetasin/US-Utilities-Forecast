# from web import create_app
# from web.utils.stock import initialize_stocks
# from web.utils.news import initialize_news
# from web.scheduler import start_stock_scheduler, start_news_scheduler, start_forecast_scheduler, update_forecast
# from web.utils.stock import TICKERS

# app = create_app()

# # initialize_stocks(app)
# # initialize_news(app)
# # update_forecast(app, TICKERS)

# # start_stock_scheduler(app) 
# # start_news_scheduler(app) 
# # start_forecast_scheduler(app, TICKERS)

# if __name__ == "__main__":
#     app.run(debug=True)


from web import create_app
from web.scheduler import start_stock_scheduler, start_news_scheduler, start_forecast_scheduler

app = create_app()

# --------------------------
# Initialize DB & Schedulers
# --------------------------
with app.app_context():
    # Scheduler จะเช็ค DB ว่างหรือ outdated เอง
    start_stock_scheduler(app)
    start_news_scheduler(app)
    start_forecast_scheduler(app)

# --------------------------
# Run Flask
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
