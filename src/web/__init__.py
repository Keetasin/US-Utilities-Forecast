
# src/web/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # ✅ แนะนำให้ปิด

    db.init_app(app)
    migrate.init_app(app, db)

    from .views import views
    app.register_blueprint(views, url_prefix='/')

    return app
