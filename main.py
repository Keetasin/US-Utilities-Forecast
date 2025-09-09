from web import create_app, db
from web.stock_utils import initialize_stocks
from web.scheduler import start_scheduler

app = create_app()
initialize_stocks(app)
start_scheduler(app) 

if __name__ == "__main__":
    app.run(debug=True)