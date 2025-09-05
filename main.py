from web import create_app, db
from web.views import initialize_stocks

app = create_app()
initialize_stocks(app)

if __name__ == "__main__":
    app.run(debug=True)
