from web import create_app
from web.stock_utils import initialize_stocks
from web.news_utils import initialize_news
from web.scheduler import start_stock_scheduler, start_news_scheduler

app = create_app()

initialize_stocks(app)
initialize_news(app)

start_stock_scheduler(app)  # Update stock prices every 1 min
start_news_scheduler(app)   # Update news daily 20:00

if __name__ == "__main__":
    app.run(debug=True)
