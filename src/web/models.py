from . import db
from datetime import datetime

class Stock(db.Model):
    __tablename__ = "stocks"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    price = db.Column(db.Float, nullable=False)
    change = db.Column(db.Float, nullable=False)
    marketCap = db.Column(db.Float, nullable=False)
    bg_color = db.Column(db.String(50), nullable=False)
    last_updated = db.Column(db.DateTime, nullable=True)

class StockNews(db.Model):
    __tablename__ = "stock_news"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    news_json = db.Column(db.JSON, nullable=False)  
    # summary = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class StockForecast(db.Model):
    __tablename__ = "stock_forecasts"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    model = db.Column(db.String(10), nullable=False)
    steps = db.Column(db.Integer, nullable=False, default=7)  
    forecast_json = db.Column(db.JSON, nullable=False)
    backtest_json = db.Column(db.JSON, nullable=True) 
    backtest_mae = db.Column(db.Float, nullable=True)  
    last_price = db.Column(db.Float, nullable=True)    
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("symbol", "model", "steps", name="uq_symbol_model_steps"),  
    )
