from apscheduler.schedulers.background import BackgroundScheduler
from .stock_utils import update_stock_data

scheduler = BackgroundScheduler()

def start_scheduler(app):
    scheduler.add_job(func=lambda: update_stock_data(app, force=True), trigger="interval", minutes=1)
    scheduler.start()
    print("Scheduler started")
