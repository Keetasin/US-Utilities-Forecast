# pipeline_multi.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Config
# =========================
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
START = "2016-01-01"
END   = None           # None = today
LOOKBACK = 60
TEST_RATIO = 0.15
VAL_RATIO  = 0.15
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Indicators (pure pandas)
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    # SMA / EMA
    df["SMA"] = c.rolling(window=20, min_periods=20).mean()
    df["EMA"] = c.ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd - signal

    # Drop warmup NaN rows
    df = df.dropna().copy()
    return df

# =========================
# Sentiment hook
# =========================
def get_sentiment_factor(ticker: str) -> float:
    """
    ใส่ logic ของคุณได้เลย (เช่น median ของคะแนนข่าวล่าสุด)
    ค่าที่คืนกลับควรเล็ก ๆ ~ [-0.02, +0.02] เพื่อเป็น multiplicative tweak
    ตอนนี้ใส่ค่า 0.0 เป็นค่าเริ่มต้น (กลาง ๆ) ไว้ก่อน
    """
    return 0.0

# =========================
# Data preparation (no leakage)
# =========================
def make_splits_scaled(df: pd.DataFrame, lookback: int):
    """
    - แยก train/val/test ตามเวลา
    - fit scaler เฉพาะ train แล้ว transform val/test (ป้องกัน leakage)
    - สร้าง X,y sequences โดย y คือ 'Close' ที่ scale แล้ว
    """
    n = len(df)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)
    n_train = n - n_test - n_val
    if n_train <= lookback + 5:
        raise ValueError(f"Rows not enough after indicators. Need > {lookback+5}, got {n}.")

    feats = ["Open","High","Low","Close","Volume","RSI","EMA","SMA","MACD"]
    data = df[feats].values.astype(np.float32)

    # Split
    train = data[:n_train]
    val   = data[n_train:n_train+n_val]
    test  = data[n_train+n_val:]

    # Scale per feature with MinMax on TRAIN only
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    train_s = scaler.transform(train)
    val_s   = scaler.transform(val)
    test_s  = scaler.transform(test)

    # Helper to create sequences
    def to_seq(arr):
        X, y = [], []
        for i in range(lookback, len(arr)):
            X.append(arr[i-lookback:i, :])      # all features in window
            y.append(arr[i, 3])                 # index 3 = 'Close' (scaled)
        return np.array(X), np.array(y)

    X_train, y_train = to_seq(train_s)
    X_val,   y_val   = to_seq(val_s)
    X_test,  y_test  = to_seq(test_s)

    # For inverse transform later
    # Build a helper that inverse only the Close column
    def inverse_close(scaled_close_values: np.ndarray):
        # Build an empty array to pass into scaler.inverse_transform
        template = np.zeros((scaled_close_values.shape[0], train.shape[1]), dtype=np.float32)
        template[:, 3] = scaled_close_values  # put close into its column index
        inv = scaler.inverse_transform(template)
        return inv[:, 3]

    # Keep aligned dates for plotting
    dates = df.index
    dates_test = dates[n_train+n_val+lookback:]  # y_test timeline

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, inverse_close, dates_test)

# =========================
# Model
# =========================
def build_lstm(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

# =========================
# Metrics
# =========================
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0

# =========================
# Plot per ticker (show, not save)
# =========================
def plot_result(ticker, dates_test, actual_test, pred_raw_test, pred_adj_test, next_raw, next_adj, mape_val):
    plt.figure(figsize=(12,6))
    plt.plot(actual_test.index, actual_test.values, label="Actual")
    plt.plot(dates_test, pred_raw_test, label="LSTM (Raw)")
    plt.plot(dates_test, pred_adj_test, label="LSTM + Sentiment (Adjusted)", linestyle="--")
    title = (f"{ticker} | Last={actual_test.values[-1]:.2f} | Next Raw={next_raw:.2f} | "
             f"Next Adj={next_adj:.2f} | MAPE={mape_val:.2f}%")
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    # โชว์ทันทีหลังเทรนเสร็จแต่ละตัว
    plt.show()

# =========================
# Main per ticker
# =========================
def run_one(ticker: str):
    print("\n" + "="*30 + f"\n📌 Processing {ticker} ...\n" + "="*30)
    raw = yf.download(ticker, start=START, end=END, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        print(f"[{ticker}] No OHLCV from yfinance, skip.")
        return None

    raw = raw[["Open","High","Low","Close","Volume"]].dropna()
    df = add_indicators(raw)

    # Prepare data
    (X_tr, y_tr, X_val, y_val, X_te, y_te,
     scaler, inv_close, dates_test) = make_splits_scaled(df, LOOKBACK)

    # Build & fit
    model = build_lstm(X_tr.shape[1:])
    es = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=2,
        callbacks=[es]
    )

    # Predict on TEST (scaled -> inverse to price)
    yhat_te_s = model.predict(X_te, verbose=0).ravel()
    yhat_te = inv_close(yhat_te_s)
    y_true_te = inv_close(y_te)

    # Last window (for next-day forecast)
    last_window = df[["Open","High","Low","Close","Volume","RSI","EMA","SMA","MACD"]].values[-LOOKBACK:]
    last_window_s = scaler.transform(last_window)[None, ...]
    next_raw_s = float(model.predict(last_window_s, verbose=0).ravel()[0])
    next_raw = float(inv_close(np.array([next_raw_s]))[0])

    # Sentiment adjust
    s_factor = get_sentiment_factor(ticker)  # e.g., +0.01 = +1%
    yhat_te_adj = yhat_te * (1.0 + s_factor)
    next_adj = next_raw * (1.0 + s_factor)

    # MAPE on test (raw)
    mape_test = mape(y_true_te, yhat_te)

    # Assemble actual series aligned with dates_test for plotting
    actual_test_series = pd.Series(y_true_te, index=dates_test)
    raw_series = pd.Series(yhat_te, index=dates_test)
    adj_series = pd.Series(yhat_te_adj, index=dates_test)

    print(f"Ticker: {ticker}")
    print(f"Last Price: {actual_test_series.values[-1]:.2f}")
    print(f"Forecast Price (next day) - Raw: {next_raw:.2f}")
    print(f"Forecast Price (next day) - Adjusted: {next_adj:.2f} (sentiment factor={s_factor:+.4f})")
    print(f"MAPE Test Set: {mape_test:.2f}%")

    # plot and show (no saving)
    plot_result(
        ticker=ticker,
        dates_test=dates_test,
        actual_test=actual_test_series,
        pred_raw_test=raw_series,
        pred_adj_test=adj_series,
        next_raw=next_raw,
        next_adj=next_adj,
        mape_val=mape_test
    )

    return {
        "ticker": ticker,
        "last": float(actual_test_series.values[-1]),
        "next_raw": float(next_raw),
        "next_adj": float(next_adj),
        "mape": float(mape_test),
        "sentiment_factor": float(s_factor)
    }

# =========================
# Entry
# =========================
def main():
    results = []
    for t in TICKERS:
        try:
            res = run_one(t)
            if res: results.append(res)
        except Exception as e:
            print(f"[{t}] Error: {e}")

    if results:
        out = pd.DataFrame(results)
        print("\nFinal Summary:")
        print(out.to_string(index=False))
        print(f"\nAverage MAPE Test Set: {out['mape'].mean():.2f}%")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
