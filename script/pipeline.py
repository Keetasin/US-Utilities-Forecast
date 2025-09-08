import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
from transformers import pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# STEP 1: Config
# -----------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]  # The Magnificent Seven
START_DATE = "2020-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")
NEWS_API_KEY = "bfa14549cdc84faf88c40e947da98d4d"

# -----------------------------
# STEP 2: Fetch News (NewsAPI)
# -----------------------------
def fetch_news(query, api_key=NEWS_API_KEY):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", [])
        news_list = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            published = a.get("publishedAt")
            news_list.append({"text": title + " " + desc, "published": published})
        return news_list
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []

# -----------------------------
# STEP 3: Sentiment Analysis (FinBERT)
# -----------------------------
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(news_list):
    if not news_list:
        return pd.DataFrame(columns=["Sentiment", "Confidence", "Published"])
    texts = [n["text"] for n in news_list[:5]]
    results = sentiment_pipeline(texts)
    df = pd.DataFrame(results).rename(columns={"label": "Sentiment", "score": "Confidence"})
    df["Published"] = [n["published"] for n in news_list[:5]]
    print(f"Sentiment Analysis Results:\n{df}")
    return df

# -----------------------------
# STEP 3B: Summarize News (LLaMA)
# -----------------------------
def summarize_news_for_investor(news_list):
    if not news_list:
        return "No recent news."
    combined_news = " ".join([n["text"] for n in news_list[:5]])
    prompt = f"Summarize the following news for an investor in 2-3 sentences: {combined_news}"
    try:
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={"model": "llama3", "messages": [{"role": "user", "content": prompt}]}
        )
        if response.status_code != 200:
            return "Failed to generate summary."
        summary = response.json()['choices'][0]['message']['content']
        return summary
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return "Failed to generate summary."

# -----------------------------
# STEP 4: Fetch Stock Data (OHLC + Volume)
# -----------------------------
def fetch_stock(ticker):
    try:
        data = yf.download(ticker, START_DATE, END_DATE, auto_adjust=True)
        return data[["Open","High","Low","Close","Volume"]] if not data.empty else pd.DataFrame()
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

# -----------------------------
# STEP 5: LSTM Forecast (OHLC + Volume, Dropout, lookback=90)
# -----------------------------
def build_lstm_forecast(data, test_size=60):
    if len(data) < 100:
        return data['Close'].iloc[-1], [], []

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)

    lookback = 90
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 3])  # Close price as target

    X, y = np.array(X), np.array(y)
    split = len(X) - test_size
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stop])

    # Predict test set
    y_pred = model.predict(X_test, verbose=0)
    y_pred_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred),3)), y_pred, np.zeros((len(y_pred),1))], axis=1))[:,3]
    y_test_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test),3)), y_test.reshape(-1,1), np.zeros((len(y_test),1))], axis=1))[:,3]

    # Predict next day
    last_seq = scaled[-lookback:].reshape(1, lookback, X.shape[2])
    next_pred = model.predict(last_seq, verbose=0)
    next_pred_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((1,3)), next_pred, np.zeros((1,1))], axis=1))[:,3]

    return next_pred_rescaled[0], y_test_rescaled, y_pred_rescaled

# -----------------------------
# STEP 6: Adaptive Sentiment Adjustment
# -----------------------------
def sentiment_factor_adaptive(sentiment_df, stock_volatility=0.02):
    """
    ปรับ sentiment factor อัตโนมัติตาม volatility ของหุ้น
    - stock_volatility: ค่าความผันผวน (เช่น std ของ returns) ใช้ scale factor
    """
    if sentiment_df.empty:
        return 0
    factor_sum, weight_sum = 0, 0
    now = datetime.datetime.now(datetime.timezone.utc)
    for _, row in sentiment_df.iterrows():
        conf, published = row['Confidence'], row['Published']
        try: published_dt = datetime.datetime.fromisoformat(published.replace("Z","+00:00"))
        except: published_dt = now
        days_diff = max((now - published_dt).days,1)
        weight = 1/(days_diff)
        score = conf if row['Sentiment']=='positive' else -conf
        factor_sum += weight * score
        weight_sum += weight

    raw_factor = (factor_sum/weight_sum) if weight_sum != 0 else 0
    # scale factor according to volatility to avoid bias
    adaptive_factor = np.tanh(raw_factor) * stock_volatility  # tanh limit factor between -stock_volatility ~ +stock_volatility
    return adaptive_factor


def adjust_forecast_with_sentiment(forecast, sentiment_df, stock_data=None):
    # ใช้ volatility ของหุ้นจาก historical returns
    if stock_data is not None and not stock_data.empty:
        returns = stock_data['Close'].pct_change().dropna()
        vol = returns.std()  # standard deviation of returns
    else:
        vol = 0.02  # default
    factor = sentiment_factor_adaptive(sentiment_df, stock_volatility=vol)
    return forecast*(1+factor), factor


# -----------------------------
# STEP 6C: Generate Signal (fixed)
# -----------------------------
def generate_signal(last_price, forecast_price, threshold=0.02):
    last_price = float(last_price)
    forecast_price = float(forecast_price)
    change = (forecast_price - last_price)/last_price
    if change > threshold:
        return "Buy"
    elif change < -threshold:
        return "Sell"
    else:
        return "Hold"

# -----------------------------
# STEP 6D: Evaluate Forecast (MAPE)
# -----------------------------
def evaluate_forecast(actual, predicted):
    return np.mean(np.abs((np.array(actual)-np.array(predicted))/np.array(actual)))*100

# -----------------------------
# STEP 6: Pipeline for 7 stocks
# -----------------------------
all_insights, mape_before_list, mape_after_list = [], [], []

for ticker in TICKERS:
    print(f"\n==============================")
    print(f"📌 Processing {ticker} ...")
    print(f"==============================")
    
    news = fetch_news(ticker)
    sentiment_df = analyze_sentiment(news)
    news_summary = summarize_news_for_investor(news)
    
    stock_data = fetch_stock(ticker)
    if stock_data.empty:
        print(f"No stock data for {ticker}, skipping.")
        continue

    last_price = float(stock_data['Close'].iloc[-1])
    forecast_price, y_test, y_pred = build_lstm_forecast(stock_data)
    forecast_adjusted, sentiment_factor = adjust_forecast_with_sentiment(forecast_price, sentiment_df)
    signal = generate_signal(last_price, forecast_adjusted)

    if len(y_test)>0:
        mape_before = evaluate_forecast(y_test, y_pred)
        y_pred_adj = y_pred*(1+sentiment_factor)
        mape_after = evaluate_forecast(y_test, y_pred_adj)
    else: 
        mape_before, mape_after = np.nan, np.nan

    mape_before_list.append(mape_before)
    mape_after_list.append(mape_after)
    sentiment_summary = sentiment_df['Sentiment'].value_counts().to_dict() if not sentiment_df.empty else {}
    
    all_insights.append({
        "Ticker": ticker,
        "Last Price": last_price,
        "Forecast Price": forecast_price,
        "Forecast Adjusted": forecast_adjusted,
        "Sentiment": sentiment_summary,
        "Signal": signal,
        "MAPE Test Set (%)": mape_before,
        "MAPE Adjusted Test Set (%)": mape_after
    })

    # Rolling visualization
    if len(y_test)>0:
        plt.figure(figsize=(12,6))
        plt.plot(y_test,label="Actual (Test)")
        plt.plot(y_pred,label="Forecast (LSTM)")
        plt.plot(y_pred_adj,label="Forecast Adjusted (Sentiment)")
        plt.axhline(forecast_adjusted, color='orange', linestyle='--', label="Next Day Adjusted")
        plt.title(f"{ticker} Stock Forecast vs Actual")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    # Print summary for each stock
    print(f"Ticker: {ticker}")
    print(f"Last Price: {last_price:.2f}")
    print(f"Forecast Price: {forecast_price:.2f}")
    print(f"Forecast Adjusted: {forecast_adjusted:.2f} (Sentiment factor: {sentiment_factor:.4f})")
    print(f"Signal: {signal}")
    print(f"Sentiment Summary: {sentiment_summary}")
    print(f"MAPE Test Set: {mape_before:.2f}%")
    print(f"MAPE Adjusted: {mape_after:.2f}%")
    print(f"News Summary: {news_summary}")
    print("\n------------------------------\n")

# -----------------------------
# STEP 7: Show Final Table
# -----------------------------
insight_df = pd.DataFrame(all_insights)
print("\nFinal Insights Table for The Magnificent Seven:")
print(insight_df)
print(f"\nAverage MAPE Test Set: {np.nanmean(mape_before_list):.2f}%")
print(f"Average MAPE Adjusted (Sentiment): {np.nanmean(mape_after_list):.2f}%")
