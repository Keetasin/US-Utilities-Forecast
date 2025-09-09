from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_caching import Cache  # เพิ่มบรรทัดนี้

db = SQLAlchemy()
DB_NAME = "database.db"

# กำหนดค่าสำหรับแคช
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 3600}) # เพิ่มบรรทัดนี้

def create_app():
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['UPLOAD_FOLDER'] = 'website/static/uploads'
    
    db.init_app(app)

    # เริ่มต้นใช้งานแคชกับแอปพลิเคชัน
    cache.init_app(app)  # เพิ่มบรรทัดนี้

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    create_database(app)

    return app

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        with app.app_context():
            db.create_all()
        print('Created Database!')