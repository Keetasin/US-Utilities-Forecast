from flask import Blueprint, render_template, request
from . import db, cache # เพิ่ม cache ที่นี่
from .models import Stock, StockNews
from .stock_utils import TICKERS, fetch_and_update_stock, update_stock_data
from .news_utils import fetch_news, summarize_news_for_investor
from .forecast_utils import (ensure_datetime_freq, series_to_chart_pairs_safe,
                             get_period_by_model, backtest_last_n_days,
                             future_forecast, to_scalar)

import yfinance as yf
import pytz

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

@views.route('/stock/<symbol>')
def stock_detail(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock: return "Stock not found", 404
    return render_template("stock_detail.html", stock=stock)


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


@views.route('/forecasting/<symbol>')
@cache.cached(query_string=True) # เพิ่มบรรทัดนี้
def forecasting(symbol):
    model = (request.args.get("model") or "arima").lower()
    try:
        period = get_period_by_model(model)
        full_close = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
        full_close = ensure_datetime_freq(full_close)
        if len(full_close) < 70:
            return render_template("forecasting.html", has_data=False, symbol=symbol,
                                   model=model.upper(), error=f"Not enough data ({len(full_close)} rows)")
        back_fc, back_mae = backtest_last_n_days(full_close, model_name=model, steps=7)
        future_fc = future_forecast(full_close, model_name=model, steps=7)
        hist = full_close.tail(90)
        last_price_val = to_scalar(hist.iloc[-1])
        back_mae_pct = (to_scalar(back_mae)/last_price_val*100.0) if last_price_val else 0.0
        return render_template("forecasting.html",
                               symbol=symbol, model=model.upper(),
                               forecast=series_to_chart_pairs_safe(future_fc),
                               historical=series_to_chart_pairs_safe(hist),
                               backtest=series_to_chart_pairs_safe(back_fc),
                               backtest_mae=to_scalar(back_mae),
                               backtest_mae_pct=back_mae_pct,
                               has_data=True,
                               trend={"direction":"Up" if to_scalar(future_fc.iloc[-1])>last_price_val else "Down",
                                      "icon":"🔼" if to_scalar(future_fc.iloc[-1]) > last_price_val else "🔽"},
                               last_price=round(last_price_val,2)
                               )
    except Exception as e:
        return render_template("forecasting.html", has_data=False, symbol=symbol, model=model.upper(), error=str(e))

@views.route('/compare/<symbol>')
@cache.cached()  # เพิ่มบรรทัดนี้
def compare_models(symbol):
    results = {}
    historical = None
    last_price = None
    models = ["arima", "sarima", "lstm"]
    try:
        long_close = yf.download(symbol, period="10y", progress=False, auto_adjust=True)['Close'].dropna()
        long_close = ensure_datetime_freq(long_close)
        historical = series_to_chart_pairs_safe(long_close.tail(120))
        last_price = round(to_scalar(long_close.iloc[-1]),2)
    except Exception as e:
        print("Error fetching long history:", e)
    for m in models:
        try:
            period = get_period_by_model(m)
            s = yf.download(symbol, period=period, progress=False, auto_adjust=True)['Close'].dropna()
            s = ensure_datetime_freq(s)
            if len(s) < 70:
                results[m] = {"ok": False, "error": f"Not enough data ({len(s)} rows) for {m.upper()} period={period}"}
                continue
            back_fc, back_mae = backtest_last_n_days(s, model_name=m, steps=7)
            fut_fc = future_forecast(s, model_name=m, steps=7)
            last_p = to_scalar(s.iloc[-1]) if len(s) else 0.0
            mae_pct = (to_scalar(back_mae) / last_p * 100.0) if last_p else 0.0
            results[m] = {
                "ok": True,
                "period": period,
                "backtest_mae": to_scalar(back_mae),
                "backtest_mae_pct": mae_pct,
                "forecast": series_to_chart_pairs_safe(fut_fc),
                "forecast_last": round(to_scalar(fut_fc.iloc[-1]),2)
            }
        except Exception as e:
            results[m] = {"ok": False, "error": str(e)}
    best_model, best_mae = None, None
    for m in models:
        if results.get(m, {}).get("ok"):
            mae = to_scalar(results[m]["backtest_mae"])
            if (best_mae is None) or (mae < best_mae):
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