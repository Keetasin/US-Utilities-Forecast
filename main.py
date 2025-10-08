from src.web import create_app, db
from src.web.scheduler import (start_stock_scheduler, start_news_scheduler, start_forecast_scheduler)

app = create_app()

with app.app_context():
    db.create_all()             

    start_stock_scheduler(app)
    start_news_scheduler(app)
    # start_forecast_scheduler(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
