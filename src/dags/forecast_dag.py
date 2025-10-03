from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
from sqlalchemy import create_engine, text
import json


# --------------------
# Config
# --------------------

# connection string postgres (ใช้ container name postgres)
PG_CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(PG_CONN)

# TICKERS = ["AEP", "SO"]
TICKERS = ["AEP"]
MODELS = ["ARIMA", "SARIMA", "SARIMAX", "LSTM"]
CALENDAR_TO_BDAYS = {7: 5, 180: 126, 365: 252}
OUTPUT_DIR = "/opt/airflow/data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --------------------
# Helper: map horizon -> training period
# --------------------
def get_train_period(horizon: int) -> str:
    if horizon <= 7:
        return "6mo"
    elif horizon <= 180:
        return "2y"
    elif horizon <= 365:
        return "5y"
    return "2y"

# --------------------
# Run one model (Forecast + Backtest)
# --------------------

def run_model(symbol, model_name, horizon, ti):
    steps = CALENDAR_TO_BDAYS[horizon]
    period = get_train_period(horizon)

    data = yf.download(symbol, period=period, progress=False, auto_adjust=True)["Close"].dropna()
    if len(data) < 50:
        print(f"⚠️ Skip {symbol}-{model_name}-{horizon}, too few rows")
        return None

    results_forecast, results_backtest = [], []
    mae = None

    try:
        # ---------------- classical models ----------------
        if model_name in ["ARIMA", "SARIMA", "SARIMAX"]:
            if model_name == "ARIMA":
                model = ARIMA(data, order=(2,1,2)).fit()
                fc = model.forecast(steps=steps)

            elif model_name == "SARIMA":
                model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,0,20)).fit(disp=False)
                fc = model.forecast(steps=steps)

            elif model_name == "SARIMAX":
                exog = yf.download("CL=F", period=period, progress=False, auto_adjust=True)["Close"].dropna()
                exog = exog.reindex(data.index).ffill()
                model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,0,20), exog=exog).fit(disp=False)
                fc = model.forecast(steps=steps, exog=exog.iloc[-steps:])

            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
            for d, v in zip(future_dates, fc):
                results_forecast.append({"date": d.strftime("%Y-%m-%d"), "forecast": float(v)})

        elif model_name == "LSTM":
            values = data.values.reshape(-1,1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)

            lookback = 10
            X, y = [], []
            for i in range(len(scaled)-lookback):
                X.append(scaled[i:i+lookback])
                y.append(scaled[i+lookback])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                LSTM(10, input_shape=(lookback,1)),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=1, verbose=0)

            last_seq = X[-1]
            preds = []
            for _ in range(steps):
                p = model.predict(last_seq.reshape(1,lookback,1), verbose=0)[0][0]
                preds.append(p)
                new_seq = np.append(last_seq[1:], [[p]], axis=0)
                last_seq = new_seq
            preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
            for d, v in zip(future_dates, preds):
                results_forecast.append({"date": d.strftime("%Y-%m-%d"), "forecast": float(v)})

        # ---------------- Backtest (last 5 days) ----------------
        train, test = data.iloc[:-5], data.iloc[-5:]
        if len(train) >= 20:
            if model_name == "LSTM":
                # (ทำ backtest LSTM แบบด้านบน)
                pass
            else:
                bt_model = model.__class__(train, **model.specification).fit()
                preds = bt_model.forecast(steps=5)

            mae = mean_absolute_error(test, preds)
            for d, v, a in zip(test.index, preds, test):
                results_backtest.append({"date": d.strftime("%Y-%m-%d"),
                                         "actual": float(a),
                                         "predicted": float(v)})

        # ---------------- Save to Postgres ----------------
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO stock_forecasts (symbol, model, steps, forecast_json, backtest_json, backtest_mae, last_price, updated_at)
                VALUES (:symbol, :model, :steps, :forecast_json, :backtest_json, :backtest_mae, :last_price, NOW())
                ON CONFLICT (symbol, model, steps) DO UPDATE
                SET forecast_json = EXCLUDED.forecast_json,
                    backtest_json = EXCLUDED.backtest_json,
                    backtest_mae = EXCLUDED.backtest_mae,
                    last_price = EXCLUDED.last_price,
                    updated_at = NOW();
            """), {
                "symbol": symbol,
                "model": model_name,
                "steps": steps,
                "forecast_json": json.dumps(results_forecast),
                "backtest_json": json.dumps(results_backtest) if results_backtest else None,
                "backtest_mae": mae,
                "last_price": float(data.iloc[-1]) if len(data) else None
            })

        print(f"✅ Saved {symbol}-{model_name}-{horizon} to Postgres")

    except Exception as e:
        print(f"[{model_name} Error] {e}")


# --------------------
# Merge results
# --------------------
def merge_results(ti):
    try:
        with engine.begin() as conn:
            # รวมจำนวน record ทั้งหมด
            total = conn.execute(text("SELECT COUNT(*) FROM stock_forecasts")).scalar()

            # เฉลี่ย MAE ของแต่ละ model
            rows = conn.execute(text("""
                SELECT model, AVG(backtest_mae) AS avg_mae, COUNT(*) AS n
                FROM stock_forecasts
                GROUP BY model
            """)).fetchall()

            # วันที่ล่าสุดที่มีการอัปเดต
            last_update = conn.execute(text("SELECT MAX(updated_at) FROM stock_forecasts")).scalar()

        print("📊 StockForecasts Summary in Postgres")
        print(f"   Total records: {total}")
        if rows:
            for r in rows:
                print(f"   {r.model}: avg MAE={r.avg_mae:.4f} (n={r.n})")
        print(f"   Last updated: {last_update}")

    except Exception as e:
        print(f"[merge_results Error] {e}")


# --------------------
# DAG
# --------------------
with DAG(
    "forecast_pipeline_dag",
    default_args=default_args,
    description="Forecast DAG with ARIMA/SARIMA/SARIMAX/LSTM + Spark + Branch + XCom + Backtest + Horizons",
    schedule_interval="@daily",
    start_date=datetime(2025,1,1),
    catchup=False,
    tags=["assignment","forecast"],
) as dag:

    start = DummyOperator(task_id="start")

    spark_transform = SparkSubmitOperator(
        task_id="spark_transform",
        application="/opt/airflow/spark/applications/spark_jobs/forecast_timeseries.py",
        conn_id="spark_default",
        verbose=True,
        name="local-spark-job",
        conf={"spark.master": "local[*]"}
    )

    branch = BranchPythonOperator(
        task_id="branching",
        python_callable=lambda: [f"forecast_group.{symbol}_{model}_{h}"
                                 for symbol in TICKERS for model in MODELS for h in CALENDAR_TO_BDAYS.keys()]
    )

    with TaskGroup("forecast_group") as forecast_group:
        for symbol in TICKERS:
            for model in MODELS:
                for h in CALENDAR_TO_BDAYS.keys():
                    PythonOperator(
                        task_id=f"{symbol}_{model}_{h}",
                        python_callable=run_model,
                        op_kwargs={"symbol": symbol, "model_name": model, "horizon": h},
                    )

    merge = PythonOperator(
        task_id="merge_results",
        python_callable=merge_results
    )

    end = DummyOperator(task_id="end")

    start >> spark_transform >> branch >> forecast_group >> merge >> end
