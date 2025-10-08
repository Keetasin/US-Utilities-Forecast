from flask import Blueprint, render_template, request, current_app
from .models import Stock, StockNews, StockForecast
from .utils.forecast import (ensure_datetime_freq, series_to_chart_pairs_safe,
                             get_period_by_model, to_scalar, update_forecast)

import yfinance as yf
import pytz
import pandas as pd
from datetime import datetime


MARKET_MAPPING = {
    "AEP": "NASDAQ",
    "DUK": "NYSE",
    "SO": "NYSE",
    "ED": "NYSE",
    "EIX": "NYSE"
}


views = Blueprint('views', __name__)


def downsample_historical(data, steps):
    if not data or not isinstance(data, list):
        return data
    if steps <= 7:
        return data

    df = pd.DataFrame(data)
    if "date" not in df or "price" not in df:
        return data

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    last_day = df.index[-1]  
    if steps == 180:
        df_down = df.iloc[::7] 
    elif steps == 365:
        df_down = df.iloc[::30]  
    else:
        return data

    if last_day not in df_down.index:
        df_down.loc[last_day] = df.loc[last_day]

    df_down = df_down.sort_index()
    return [{"date": d.strftime("%Y-%m-%d"), "price": float(p)} for d, p in df_down["price"].items()]

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

@views.route('/stock/<symbol>')
def stock_detail(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock: 
        return "Stock not found", 404
    market = MARKET_MAPPING.get(symbol, "NASDAQ")  
    return render_template("stock_detail.html", stock=stock, market=market)

@views.route('/news/<symbol>')
def news(symbol):
    sn = StockNews.query.filter_by(symbol=symbol).first()
    if sn:
        news_list = sn.news_json
    else:
        news_list = []
    return render_template("news.html", symbol=symbol, news=news_list[:5])

@views.route('/stock/analytics/<symbol>')
def stock_analytics(symbol):
    stock = Stock.query.filter_by(symbol=symbol).first()
    if not stock:
        return "Stock not found", 404

    last_updated = stock.last_updated.strftime("%d/%m/%Y") if stock.last_updated else "N/A"

    info = {}
    hist_prices = {'Date': [], 'Close': [], 'MA50': [], 'MA200': []}
    sp500_prices = []
    financials = []

    try:
        stk_yf = yf.Ticker(symbol)
        info = stk_yf.info

        hist = stk_yf.history(period="5y")
        if not hist.empty:
            hist_full = hist.copy()
            hist_full['MA50'] = hist_full['Close'].rolling(50).mean()
            hist_full['MA200'] = hist_full['Close'].rolling(200).mean()
            hist_full = hist_full.reset_index()
            hist_full['Date'] = hist_full['Date'].dt.strftime('%Y-%m-%d')
            hist_prices = hist_full[['Date','Close','MA50','MA200']].to_dict(orient='list')

            sp500_hist = yf.Ticker("^GSPC").history(period="5y")['Close'].reset_index()
            sp500_hist['Date'] = sp500_hist['Date'].dt.strftime('%Y-%m-%d')
            sp500_prices = sp500_hist['Close'].tolist()

        if hasattr(stk_yf, 'financials') and not stk_yf.financials.empty:
            fin = stk_yf.financials.T.reset_index()
            fin.rename(columns={'index':'date','Total Revenue':'revenue','Net Income':'netIncome'}, inplace=True)
            financials = fin[['date','revenue','netIncome']].to_dict(orient='records')

    except Exception as e:
        print("Error fetching yfinance:", e)

    div_yield_raw = info.get("dividendYield")
    div_yield = round(div_yield_raw, 2) if div_yield_raw else 0
    pe_raw = info.get("trailingPE")
    pe = round(pe_raw, 2) if pe_raw else 0
    beta_raw = info.get("beta")
    beta = round(beta_raw, 2) if beta_raw else 0

    overview_data = [{
        "ticker": symbol,
        "price": round(info.get("currentPrice", stock.price), 2),
        "div_yield": div_yield,
        "pe": pe,
        "beta": beta,
        "market_cap": info.get("marketCap", stock.marketCap),
    }]

    return render_template(
        "stock_analytics.html",
        stock=stock,
        last_updated=last_updated,
        overview_data=overview_data,
        hist_prices=hist_prices,
        sp500_prices=sp500_prices,
        financials=financials
    )

@views.route('/dashboard')
def dashboard():
    stocks = Stock.query.all()
    
    symbols, market_caps, colors = [], [], []
    dividend_yields, pe_ratios, betas, pb_ratios = [], [], [], []
    high_52w, low_52w, ytd_changes, avg_volumes = [], [], [], []

    hist_prices_dict = {}
    year_start = datetime(datetime.now().year, 1, 1)

    for s in stocks:
        symbols.append(s.symbol)
        market_caps.append(s.marketCap)
        colors.append(s.bg_color)

        try:
            info = yf.Ticker(s.symbol).info
        except:
            info = {}

        dy_raw = info.get("dividendYield", 0) or 0
        dy = round(dy_raw,2) if dy_raw < 1 else round(dy_raw,2)
        dividend_yields.append(dy)

        pe_ratios.append(round(info.get("trailingPE", 0) or 0,2))
        betas.append(round(info.get("beta", 0) or 0,2))
        pb_ratios.append(round(info.get("priceToBook", 0) or 0,2))

        high_52w.append(round(info.get("fiftyTwoWeekHigh",0) or 0,2))
        low_52w.append(round(info.get("fiftyTwoWeekLow",0) or 0,2))
        avg_volumes.append(round(info.get("averageVolume",0) or 0))

        try:
            hist = yf.Ticker(s.symbol).history(start=year_start)['Close']
            hist_prices_dict[s.symbol] = hist
        except:
            hist_prices_dict[s.symbol] = pd.Series([0])

        if not hist_prices_dict[s.symbol].empty:
            ytd = round((hist_prices_dict[s.symbol].iloc[-1] - hist_prices_dict[s.symbol].iloc[0]) / hist_prices_dict[s.symbol].iloc[0] * 100,2)
        else:
            ytd = 0
        ytd_changes.append(ytd)

    all_dates = sorted(set(date.date() for s in hist_prices_dict.values() for date in s.index))
    historical_dates = [str(d) for d in all_dates]

    historical_prices = []
    for sym in symbols:
        series = hist_prices_dict[sym]
        prices_aligned, last_val, idx = [], (series.iloc[0] if not series.empty else 0), 0
        for d in all_dates:
            if idx < len(series) and series.index[idx].date() == d:
                last_val = series.iloc[idx]
                idx += 1
            prices_aligned.append(last_val)
        historical_prices.append(prices_aligned)

    return render_template(
        'dashboard.html',
        stocks=stocks,
        symbols=symbols,
        market_caps=market_caps,
        colors=colors,
        dividend_yields=dividend_yields,
        pe_ratios=pe_ratios,
        betas=betas,
        pb_ratios=pb_ratios,
        high_52w=high_52w,
        low_52w=low_52w,
        ytd_changes=ytd_changes,
        avg_volumes=avg_volumes,
        historical_prices=historical_prices,
        historical_dates=historical_dates
    )

@views.route('/forecasting/<symbol>/<model>')
def forecasting(symbol, model):
    model = (model or "arima").lower()
    steps = int(request.args.get("steps", 7))

    fc = StockForecast.query.filter_by(symbol=symbol, model=model, steps=steps).first()

    if not fc:
        update_forecast(current_app, [symbol], models=[model], steps_list=[steps])
        fc = StockForecast.query.filter_by(symbol=symbol, model=model, steps=steps).first()
        if not fc:
            return render_template("forecasting.html", has_data=False, symbol=symbol, model=model.lower(), error="No forecast yet.", steps=steps)

    forecast_full = fc.forecast_json or []
    forecast_json = downsample_historical(forecast_full, steps)

    backtest_full = getattr(fc, "backtest_json", None) or []
    backtest = downsample_historical(backtest_full, steps)

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
            if len(full_close) > 300:
                full_close = full_close.tail(300)
            historical = series_to_chart_pairs_safe(full_close)
            last_price_val = to_scalar(full_close.iloc[-1])
        except:
            historical = []

    historical = downsample_historical(historical, steps)

    backtest_mae = getattr(fc, "backtest_mae", None)
    last_price_val = getattr(fc, "last_price", None) or (forecast_json[0]["price"] if forecast_json else 0.0)
    backtest_mae_pct = (backtest_mae / last_price_val * 100.0) if last_price_val and backtest_mae else 0.0

    return render_template("forecasting.html", symbol=symbol, model=model.upper(),
                           forecast=forecast_json, historical=historical,
                           backtest=backtest, backtest_mae=backtest_mae,
                           backtest_mae_pct=backtest_mae_pct, last_price=round(last_price_val, 2),
                           has_data=True, steps=steps,
                           last_updated=fc.updated_at.strftime("%Y-%m-%d %H:%M"))

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

        forecast_full = fc.forecast_json or []
        forecast_json = downsample_historical(forecast_full, steps)

        backtest_full = getattr(fc, "backtest_json", []) or []
        backtest_json = downsample_historical(backtest_full, steps)

        backtest_mae = getattr(fc, "backtest_mae", 0)

        h = getattr(fc, "historical_json", None) or []
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
                if len(full_close) > 300:
                    full_close = full_close.tail(300)
                h = series_to_chart_pairs_safe(full_close)
            except:
                h = []

        historical = downsample_historical(h, steps)
        last_price = round(to_scalar(full_close.iloc[-1]) if h else forecast_json[0]["price"], 2)

        backtest_mae_pct = (backtest_mae / last_price * 100.0) if last_price else 0.0
        results[m] = {"ok": True, "forecast": forecast_json, "historical": historical,
                      "backtest": backtest_json, "backtest_mae": backtest_mae,
                      "backtest_mae_pct": backtest_mae_pct,
                      "forecast_last": round(forecast_json[-1]["price"], 2) if forecast_json else None}

    best_model, best_mae = None, None
    for m in models:
        if results.get(m, {}).get("ok"):
            mae = results[m]["backtest_mae"]
            if best_mae is None or mae < best_mae:
                best_mae, best_model = mae, m.upper()

    return render_template("compare.html", symbol=symbol, historical=historical,
                           last_price=last_price, results=results,
                           best_model=best_model, best_mae=best_mae, steps=steps)