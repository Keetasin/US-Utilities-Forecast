from flask import Blueprint, render_template, request, current_app
from . import db
from .models import Stock, StockNews, StockForecast
from .utils.stock import TICKERS, fetch_and_update_stock, update_stock_data
from .utils.news import fetch_news, summarize_news_for_investor
from .utils.forecast import (ensure_datetime_freq, series_to_chart_pairs_safe,
                             get_period_by_model, backtest_last_n_days,
                             future_forecast, to_scalar, update_forecast)

import yfinance as yf
import pytz
import pandas as pd
import numpy as np
from datetime import datetime

views = Blueprint('views', __name__)

# ---------------------------
# Helper: downsample historical
# ---------------------------
def downsample_historical(data, steps):
    """ลด historical resolution ให้สอดคล้องกับ horizon"""
    if not data or not isinstance(data, list):
        return data
    if steps <= 7:   # ❗ ไม่แตะ 1 week
        return data

    df = pd.DataFrame(data)
    if "date" not in df or "price" not in df:
        return data

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    if steps == 180:   # 6 เดือน → weekly
        df = df.resample("W").last()
    elif steps == 365: # 1 ปี → monthly
        df = df.resample("M").last()
    else:
        return data

    return [{"date": d.strftime("%Y-%m-%d"), "price": float(p)} for d, p in df["price"].items()]

# ---------------------------
# Home & Heatmap
# ---------------------------
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

# ---------------------------
# News
# ---------------------------
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

# ---------------------------
# Stock Analytics
# ---------------------------
@views.route('/stock/analytics/<symbol>')
def stock_analytics(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock:
        return "Stock not found", 404

    last_updated = stock.last_updated.strftime("%d/%m/%Y") if stock.last_updated else "N/A"

    try:
        stk_yf = yf.Ticker(symbol)
        info = stk_yf.info

        dividends = stk_yf.dividends
        dividend_years = dividends.resample('Y').sum()
        dividend_years_dict = {d.strftime("%Y"): float(v) for d, v in dividend_years.items()}

        fin = stk_yf.financials
        revenue = fin.loc["Total Revenue"].sort_index() if "Total Revenue" in fin.index else None
        net_income = fin.loc["Net Income"].sort_index() if "Net Income" in fin.index else None
        revenue_dict = {str(d): float(v) for d, v in revenue.items()} if revenue is not None else {}
        net_income_dict = {str(d): float(v) for d, v in net_income.items()} if net_income is not None else {}

        hist_full = stk_yf.history(period="5y")
        hist_full['MA50'] = hist_full['Close'].rolling(50).mean()
        hist_full['MA200'] = hist_full['Close'].rolling(200).mean()
        hist_full['Returns'] = hist_full['Close'].pct_change()
        hist_full['Volatility'] = hist_full['Returns'].rolling(20).std() * 100

        sp500 = yf.Ticker("^GSPC").history(period="5y")['Close']
        relative_perf = (hist_full['Close'] / hist_full['Close'].iloc[0] * 100) - (sp500 / sp500.iloc[0] * 100)

    except Exception as e:
        print("Error fetching yfinance:", e)
        info = {}
        dividend_years_dict, revenue_dict, net_income_dict = {}, {}, {}
        hist_full, relative_perf = {}, {}

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

# ---------------------------
# Forecasting (DB first)
# ---------------------------
@views.route('/forecasting/<symbol>/<model>')
def forecasting(symbol, model):
    model = (model or "arima").lower()
    steps = int(request.args.get("steps", 7))

    fc = StockForecast.query.filter_by(symbol=symbol, model=model, steps=steps).first()

    if not fc:
        update_forecast(current_app, [symbol], models=[model], steps_list=[steps])
        fc = StockForecast.query.filter_by(symbol=symbol, model=model, steps=steps).first()
        if not fc:
            return render_template(
                "forecasting.html",
                has_data=False,
                symbol=symbol,
                model=model.lower(),
                error="No forecast yet. Try again later.",
                steps=steps
            )

    forecast_json = fc.forecast_json

    # Historical (from DB or fallback)
    historical = getattr(fc, "historical_json", None)
    if not historical:
        try:
            if steps == 7:
                hist_period = "7d"
            elif steps == 180:
                hist_period = "6mo"
            elif steps == 365:
                hist_period = "1y"
            else:
                hist_period = get_period_by_model(model, steps)

            full_close = yf.download(symbol, period=hist_period, progress=False, auto_adjust=True)['Close'].dropna()
            full_close = ensure_datetime_freq(full_close)

            max_points = 300
            if len(full_close) > max_points:
                full_close = full_close.tail(max_points)

            historical = series_to_chart_pairs_safe(full_close)
            last_price_val = to_scalar(full_close.iloc[-1])
        except:
            historical = []

    # ✅ Downsample historical (only after we have it)
    historical = downsample_historical(historical, steps)

    backtest = getattr(fc, "backtest_json", None) or []
    backtest_mae = getattr(fc, "backtest_mae", None)

    last_price_val = getattr(fc, "last_price", None) or (forecast_json[0]["price"] if forecast_json else 0.0)
    trend = {
        "direction": "Up" if forecast_json and forecast_json[-1]["price"] > last_price_val else "Down",
        "icon": "🔼" if forecast_json and forecast_json[-1]["price"] > last_price_val else "🔽"
    }
    backtest_mae_pct = (backtest_mae / last_price_val * 100.0) if last_price_val and backtest_mae else 0.0

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
        steps=steps,
        last_updated=fc.updated_at.strftime("%Y-%m-%d %H:%M")
    )

# ---------------------------
# Compare
# ---------------------------
@views.route('/compare/<symbol>')
def compare_models(symbol):
    steps = int(request.args.get("steps", 7))
    results, historical, last_price = {}, [], None
    models = ["arima", "sarima", "sarimax", "lstm"]

    for m in models:
        fc = StockForecast.query.filter_by(symbol=symbol, model=m, steps=steps).first()
        if not fc:
            update_forecast(current_app, [symbol], models=[m], steps_list=[steps])
            fc = StockForecast.query.filter_by(symbol=symbol, model=m, steps=steps).first()
        if not fc:
            results[m] = {"ok": False, "error": f"No forecast for {m.upper()} ({steps}d)"}
            continue

        forecast_json = fc.forecast_json or []
        backtest_json = getattr(fc, "backtest_json", []) or []
        backtest_mae = getattr(fc, "backtest_mae", 0)

        # Historical (from DB or fallback)
        h = getattr(fc, "historical_json", None)
        if not h:
            try:
                if steps == 7:
                    hist_period = "7d"
                elif steps == 180:
                    hist_period = "6mo"
                elif steps == 365:
                    hist_period = "1y"
                else:
                    hist_period = get_period_by_model(m, steps)

                full_close = yf.download(symbol, period=hist_period, progress=False, auto_adjust=True)['Close'].dropna()
                full_close = ensure_datetime_freq(full_close)

                max_points = 300
                if len(full_close) > max_points:
                    full_close = full_close.tail(max_points)

                h = series_to_chart_pairs_safe(full_close)
            except:
                h = []

        # ✅ Downsample
        historical = downsample_historical(h, steps)
        last_price = round(to_scalar(h[-1]["price"]) if h else forecast_json[0]["price"], 2)

        backtest_mae_pct = (backtest_mae / last_price * 100.0) if last_price else 0.0
        results[m] = {
            "ok": True,
            "forecast": forecast_json,
            "historical": historical,
            "backtest": backtest_json,
            "backtest_mae": backtest_mae,
            "backtest_mae_pct": backtest_mae_pct,
            "forecast_last": round(forecast_json[-1]["price"], 2) if forecast_json else None
        }

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
        best_mae=best_mae,
        steps=steps
    )
