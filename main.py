from src.web import create_app
from src.web.scheduler import start_stock_scheduler, start_news_scheduler, start_forecast_scheduler

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
