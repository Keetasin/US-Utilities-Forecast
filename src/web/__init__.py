# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from os import path

# db = SQLAlchemy()
# DB_NAME = "database.db"

# def create_app():
#     app = Flask(__name__)
    
#     app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
#     app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
#     # app.config['UPLOAD_FOLDER'] = 'website/static/uploads'
    
#     db.init_app(app)

#     from .views import views

#     app.register_blueprint(views, url_prefix='/')

#     create_database(app)

#     return app

# def create_database(app):
#     if not path.exists('website/' + DB_NAME):
#         with app.app_context():
#             db.create_all()
#         print('Created Database!')


from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path, makedirs

db = SQLAlchemy()
DB_NAME = "database.db"
DB_PATH = path.abspath("src/instance")  

def create_app():
    app = Flask(__name__, instance_path=DB_PATH)
    
    # สร้าง folder instance ถ้ายังไม่มี
    if not path.exists(DB_PATH):
        makedirs(DB_PATH)

    # ตั้งค่า Flask
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{path.join(DB_PATH, DB_NAME)}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # Import Blueprint
    from .views import views
    app.register_blueprint(views, url_prefix='/')

    # สร้าง database ถ้ายังไม่มี
    create_database(app)

    return app

def create_database(app):
    db_file = path.join(DB_PATH, DB_NAME)
    if not path.exists(db_file):
        with app.app_context():
            db.create_all()
        print(f'Created Database at {db_file}!')
