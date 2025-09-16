from flask import Blueprint, render_template, request
from . import db
from .models import Stock, StockNews, StockForecast
from .utils.stock import TICKERS, fetch_and_update_stock, update_stock_data
from .utils.news import fetch_news, summarize_news_for_investor
from .utils.forecast import (ensure_datetime_freq, series_to_chart_pairs_safe,
                             get_period_by_model, backtest_last_n_days,
                             future_forecast, to_scalar)

import yfinance as yf
import pytz
import pandas as pd
import numpy as np
from datetime import datetime

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template('home.html')

@views.route('/heatmap')
def heatmap():
    data = Stock.query.order_by(Stock.marketCap.desc()).all()
    last_updated_str = None
    if data:
        last_updated_utc = max([s.last_updated for s in data if s.last_updated])
        if last_updated_utc:
            # แปลงให้ timezone-aware ก่อน
            if last_updated_utc.tzinfo is None:
                last_updated_utc = last_updated_utc.replace(tzinfo=pytz.UTC)
            tz_th = pytz.timezone("Asia/Bangkok")
            last_updated = last_updated_utc.astimezone(tz_th).replace(microsecond=0)
            last_updated_str = last_updated.strftime("%H:%M:%S %Y-%m-%d")

    return render_template("heatmap.html", data=data, last_updated=last_updated_str)

MARKET_MAPPING = {
    "AEP": "NASDAQ",
    "DUK": "NYSE",
    "SO": "NYSE",
    "ED": "NYSE",
    "EIX": "NYSE"
}

@views.route('/stock/<symbol>')
def stock_detail(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock: 
        return "Stock not found", 404
    market = MARKET_MAPPING.get(symbol, "NASDAQ")  # default NASDAQ
    return render_template("stock_detail.html", stock=stock, market=market)


@views.route('/news/<symbol>')
def news(symbol):
    sn = StockNews.query.filter_by(symbol=symbol).first()
    if sn:
        news_list = sn.news_json
        summary = sn.summary
    else:
        news_list = []
        summary = "No news yet. Will update at 20:00."
    return render_template("news.html", symbol=symbol, news=news_list[:5], summary=summary)

@views.route('/forecasting/<symbol>/<model>')
def forecasting(symbol, model):
    model = (model or "arima").lower()
    
    # โหลด forecast จาก DB
    fc = StockForecast.query.filter_by(symbol=symbol, model=model).first()
    
    if not fc:
        return render_template(
            "forecasting.html",
            has_data=False,
            symbol=symbol,
            model=model.lower(),
            error="No forecast yet. Will update at 20:00."
        )

    # Forecast (future)
    forecast_json = fc.forecast_json

    # Last price จาก historical หรือ forecast แรก
    last_price_val = getattr(fc, "last_price", None)
    if last_price_val is None:
        # fallback: ใช้ราคาของ forecast วันแรก
        last_price_val = forecast_json[0]["price"] if forecast_json else 0.0

    # Trend
    trend = {
        "direction": "Up" if forecast_json[-1]["price"] > last_price_val else "Down",
        "icon": "🔼" if forecast_json[-1]["price"] > last_price_val else "🔽"
    }

    # Historical data: fallback ถ้า DB ไม่มี ให้สร้าง dummy last 90 วัน
    historical = getattr(fc, "historical_json", None)
    if not historical or len(historical) == 0:
        # ดึงข้อมูลจาก yfinance
        try:
            period = get_period_by_model(model)
            full_close = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
            full_close = ensure_datetime_freq(full_close)
            historical = series_to_chart_pairs_safe(full_close.tail(30))
            last_price_val = to_scalar(full_close.iloc[-1])
        except:
            historical = []

    # Backtest: fallback ถ้า DB ไม่มี
    backtest = getattr(fc, "backtest_json", None)
    backtest_mae = getattr(fc, "backtest_mae", None)
    if not backtest or len(backtest) == 0:
        try:
            # สร้าง backtest 7 วันล่าสุด
            s = pd.Series([d["price"] for d in historical], index=pd.to_datetime([d["date"] for d in historical]))
            bt, mae = backtest_last_n_days(s, model_name=model, steps=7)
            backtest = series_to_chart_pairs_safe(bt)
            backtest_mae = mae
        except:
            backtest = []
            backtest_mae = 0.0

    backtest_mae_pct = (backtest_mae / last_price_val * 100.0) if last_price_val else 0.0

    return render_template(
        "forecasting.html",
        symbol=symbol,
        model=model.upper(),
        forecast=forecast_json,
        historical=historical,
        backtest=backtest,
        backtest_mae=backtest_mae,
        backtest_mae_pct=backtest_mae_pct,
        last_price=round(last_price_val, 2),
        trend=trend,
        has_data=True,
        last_updated=fc.updated_at.strftime("%Y-%m-%d %H:%M")
    )

@views.route('/compare/<symbol>')
def compare_models(symbol):
    results = {}
    historical = []
    last_price = None
    models = ["arima", "sarima", "lstm"]

    # พยายามดึง historical จาก DB ตัวแรก
    fc_records = StockForecast.query.filter_by(symbol=symbol).all()
    if fc_records:
        first_fc = fc_records[0]
        historical = getattr(first_fc, "historical_json", []) or []

    # ถ้า DB ไม่มี historical ให้ fallback ไปดึงจาก yfinance
    if not historical or len(historical) == 0:
        try:
            long_close = yf.download(symbol, period="10y", progress=False, auto_adjust=True)['Close'].dropna()
            long_close = ensure_datetime_freq(long_close)
            historical = series_to_chart_pairs_safe(long_close.tail(30))
            last_price = round(to_scalar(long_close.iloc[-1]),2)
        except Exception as e:
            print("Error fetching long history:", e)
            historical = []
            last_price = None
    else:
        last_price = round(historical[-1]["price"],2) if historical else None

    # ดึง forecast/backtest จาก DB
    for m in models:
        fc = StockForecast.query.filter_by(symbol=symbol, model=m).first()
        if not fc:
            results[m] = {"ok": False, "error": f"No forecast for {m.upper()}"}
            continue

        forecast_json = fc.forecast_json or []
        backtest_json = getattr(fc, "backtest_json", []) or []
        backtest_mae = getattr(fc, "backtest_mae", 0)
        backtest_mae_pct = (backtest_mae / last_price * 100.0) if last_price else 0.0

        results[m] = {
            "ok": True,
            "forecast": forecast_json,
            "historical": historical,
            "backtest": backtest_json,
            "backtest_mae": backtest_mae,
            "backtest_mae_pct": backtest_mae_pct,
            "forecast_last": round(forecast_json[-1]["price"],2) if forecast_json else None
        }

    # หา best model ตาม MAE
    best_model, best_mae = None, None
    for m in models:
        if results.get(m, {}).get("ok"):
            mae = results[m]["backtest_mae"]
            if best_mae is None or mae < best_mae:
                best_mae, best_model = mae, m.upper()

    return render_template(
        "compare.html",
        symbol=symbol,
        historical=historical,
        last_price=last_price,
        results=results,
        best_model=best_model,
        best_mae=best_mae
    )



@views.route('/stock/analytics/<symbol>')
def stock_analytics(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock:
        return "Stock not found", 404

    last_updated = stock.last_updated.strftime("%Y-%m-%d %H:%M") if stock.last_updated else "N/A"

    import yfinance as yf
    try:
        stk_yf = yf.Ticker(symbol)
        info = stk_yf.info

        # -------- Dividend --------
        dividends = stk_yf.dividends
        dividend_years = dividends.resample('Y').sum()
        dividend_years_dict = {d.strftime("%Y"): float(v) for d, v in dividend_years.items()}

        # -------- Revenue & Net Income --------
        fin = stk_yf.financials
        revenue = fin.loc["Total Revenue"].sort_index() if "Total Revenue" in fin.index else None
        net_income = fin.loc["Net Income"].sort_index() if "Net Income" in fin.index else None
        revenue_dict = {str(d): float(v) for d, v in revenue.items()} if revenue is not None else {}
        net_income_dict = {str(d): float(v) for d, v in net_income.items()} if net_income is not None else {}

        # -------- Moving Averages & Volatility --------
        hist_full = stk_yf.history(period="5y")
        hist_full['MA50'] = hist_full['Close'].rolling(50).mean()
        hist_full['MA200'] = hist_full['Close'].rolling(200).mean()
        hist_full['Returns'] = hist_full['Close'].pct_change()
        hist_full['Volatility'] = hist_full['Returns'].rolling(20).std() * 100

        # -------- Relative Perf vs S&P500 --------
        sp500 = yf.Ticker("^GSPC").history(period="5y")['Close']
        relative_perf = (hist_full['Close'] / hist_full['Close'].iloc[0] * 100) - (sp500 / sp500.iloc[0] * 100)

    except Exception as e:
        print("Error fetching yfinance:", e)
        info = {}
        dividend_years_dict, revenue_dict, net_income_dict = {}, {}, {}
        hist_full, relative_perf = {}, {}

    # Dividend Yield %
    div_yield_raw = info.get("dividendYield", 0) or 0
    div_yield = round(div_yield_raw*100,2) if div_yield_raw < 1 else round(div_yield_raw,2)

    overview_data = [{
        "ticker": symbol,
        "price": round(info.get("currentPrice", stock.price), 2),
        "div_yield": div_yield,
        "pe": round(info.get("trailingPE",0),2),
        "beta": round(info.get("beta",0),2),
        "market_cap": info.get("marketCap", stock.marketCap),
    }]

    return render_template(
        "stock_analytics.html",
        stock=stock,
        last_updated=last_updated,
        overview_data=overview_data,
        dividend_years=dividend_years_dict,
        net_income=net_income_dict,
        revenue=revenue_dict,
        hist_prices=hist_full.reset_index().to_dict(orient='list') if hist_full is not None else {},
        relative_perf=relative_perf.reset_index().to_dict(orient='list') if relative_perf is not None else {}
    )




@views.route('/dashboard')
def dashboard():
    stocks = Stock.query.all()
    
    symbols = []
    market_caps = []
    colors = []
    dividend_yields = []

    hist_prices_dict = {}

    # ปีนี้
    year_start = datetime(datetime.now().year, 1, 1)

    for s in stocks:
        symbols.append(s.symbol)
        market_caps.append(s.marketCap)
        colors.append(s.bg_color)

        # Dividend Yield
        try:
            info = yf.Ticker(s.symbol).info
            dy_raw = info.get("dividendYield", 0) or 0
            dy = round(dy_raw*100,2) if dy_raw < 1 else round(dy_raw,2)
        except:
            dy = 0
        dividend_yields.append(dy)

        # Historical prices ปีนี้
        try:
            hist = yf.Ticker(s.symbol).history(start=year_start)['Close']
            hist_prices_dict[s.symbol] = hist
        except:
            hist_prices_dict[s.symbol] = pd.Series([0])

    # สร้าง all_dates
    all_dates = sorted(set(date.date() for s in hist_prices_dict.values() for date in s.index))
    historical_dates = [str(d) for d in all_dates]

    # จัด historical_prices ให้ตรงกัน
    historical_prices = []
    for sym in symbols:
        series = hist_prices_dict[sym]
        prices_aligned = []
        last_val = series.iloc[0] if not series.empty else 0
        idx = 0
        for d in all_dates:
            if idx < len(series) and series.index[idx].date() == d:
                last_val = series.iloc[idx]
                idx += 1
            prices_aligned.append(last_val)
        historical_prices.append(prices_aligned)

    # สร้าง correlation matrix
    df_hist = pd.DataFrame({s: historical_prices[i] for i, s in enumerate(symbols)})
    correlation_matrix = df_hist.corr().values.tolist()

    return render_template(
        'dashboard.html',
        stocks=stocks,
        symbols=symbols,
        market_caps=market_caps,
        colors=colors,
        dividend_yields=dividend_yields,
        correlation_matrix=correlation_matrix,
        historical_prices=historical_prices,
        historical_dates=historical_dates
    )