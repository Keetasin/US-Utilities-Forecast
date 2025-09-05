from . import db

class Stock(db.Model):
    __tablename__ = "stocks"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    price = db.Column(db.Float, nullable=False)
    change = db.Column(db.Float, nullable=False)
    marketCap = db.Column(db.Float, nullable=False)
    bg_color = db.Column(db.String(50), nullable=False)
