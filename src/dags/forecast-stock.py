from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from datetime import datetime, timedelta
import os, json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sqlalchemy import create_engine, text
import pendulum
import pyarrow.parquet as pq

# ==========================================================
# CONFIG
# ==========================================================
PG_CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(PG_CONN)

TICKERS = ["AEP"]
MODELS = ["ARIMA", "SARIMA", "SARIMAX", "LSTM"]
CALENDAR_TO_BDAYS = {7: 5, 180: 126, 365: 252}
tz_th = pendulum.timezone("Asia/Bangkok")
PARQUET_PATH = "/opt/airflow/src/spark/data/spark_out/market.parquet"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ==========================================================
# CORE FUNCTION: RUN MODEL
# ==========================================================
def run_model(symbol, model_name, horizon, ti):
    try:
        steps = CALENDAR_TO_BDAYS[horizon]
        model_name = model_name.lower()
        print(f"🔹 Running {symbol}-{model_name}-{horizon}d | steps={steps}")

        # === Load parquet ===
        pdf = pq.read_table(PARQUET_PATH).to_pandas()
        pdf.columns = [c.lower() for c in pdf.columns]
        df = pdf[pdf["symbol"].str.lower() == symbol.lower()]
        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date").asfreq("B").ffill().bfill()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        data = df["close"].astype(float)
        exog_cols = [c for c in df.columns if any(x in c for x in ["cl", "ng", "xlu"])]
        exog = df[exog_cols].asfreq("B").ffill().bfill() if exog_cols else None

        if len(data) < max(70, steps + 10):
            print(f"[Skip] {symbol}-{model_name}-{horizon}d (not enough data)")
            return {"symbol": symbol, "model": model_name, "horizon": horizon, "status": "skipped"}

        # === Model parameters ===
        MODEL_PARAMS = {
            "AEP": {"arima": (1,1,0), "sarima": (0,1,0,20), "sarimax": (2,0,2,20)},
            "DUK": {"arima": (1,1,1), "sarima": (2,1,2,63), "sarimax": (0,1,2,63)},
            "SO":  {"arima": (2,1,0), "sarima": (1,0,0,20), "sarimax": (0,1,2,20)},
            "ED":  {"arima": (2,1,2), "sarima": (2,0,2,63), "sarimax": (2,1,1,63)},
            "EXC": {"arima": (2,1,2), "sarima": (1,1,1,20), "sarimax": (1,0,0,63)},
        }

        def get_orders(sym, model):
            conf = MODEL_PARAMS.get(sym, {})
            if model == "arima":
                return (conf.get("arima", (2,1,0)), None)
            elif model == "sarima":
                o = conf.get("sarima", (1,1,1,20))
                return ((o[0], o[1], o[2]), (1,1,0,o[3]))
            elif model == "sarimax":
                o = conf.get("sarimax", (1,1,1,20))
                return ((o[0], o[1], o[2]), (1,1,0,o[3]))
            else:
                return ((1,1,1), (1,1,0,20))

        order, seasonal_order = get_orders(symbol, model_name)

        # === TRAIN + FORECAST ===
        if model_name == "arima":
            m = ARIMA(data, order=order).fit()
            fc = m.forecast(steps=steps)
        elif model_name == "sarima":
            m = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = m.forecast(steps=steps)
        elif model_name == "sarimax":
            if exog is None or exog.empty:
                exog = pd.DataFrame(0, index=data.index, columns=["dummy"])
            m = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        exog=exog, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            future_exog = exog.iloc[-steps:].copy()
            fc = m.forecast(steps=steps, exog=future_exog)
        elif model_name == "lstm":
            values = data.values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)
            lookback = 20
            X, y = [], []
            for i in range(len(scaled) - lookback):
                X.append(scaled[i:i+lookback])
                y.append(scaled[i+lookback])
            X, y = np.array(X), np.array(y)
            model = Sequential([LSTM(32, input_shape=(lookback, 1)), Dense(1)])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=3, verbose=0)
            last_seq = X[-1]
            preds = []
            for _ in range(steps):
                p = model.predict(last_seq.reshape(1, lookback, 1), verbose=0)[0][0]
                preds.append(p)
                last_seq = np.append(last_seq[1:], [[p]], axis=0)
            fc = pd.Series(
                scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten(),
                index=pd.bdate_range(start=data.index[-1] + pd.offsets.BDay(1), periods=steps)
            )
        else:
            raise ValueError(f"Unknown model {model_name}")

        # === BACKTEST (แบบเดียวกับ Flask — ใช้ {date, price}) ===
        back_days = min(steps, len(data)//5)
        train, test = data.iloc[:-back_days], data.iloc[-back_days:]
        mae, results_backtest = None, []

        if len(train) > 30:
            bt = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            preds = bt.forecast(steps=back_days)
            mae = mean_absolute_error(test.values, preds)
            for d, v in zip(test.index, preds):
                results_backtest.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "price": float(v)
                })

        # === FORECAST JSON ===
        forecast_index = pd.bdate_range(start=data.index[-1] + pd.offsets.BDay(1), periods=steps)
        results_forecast = [{"date": d.strftime("%Y-%m-%d"), "price": float(v)} for d, v in zip(forecast_index, fc)]

        # === SAVE TO DB ===
        last_price = float(data.iloc[-1])
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_forecasts (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    steps INT NOT NULL,
                    forecast_json JSONB,
                    backtest_json JSONB,
                    backtest_mae DOUBLE PRECISION,
                    last_price DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(symbol, model, steps)
                );
            """))
            conn.execute(text("""
                INSERT INTO stock_forecasts (symbol, model, steps, forecast_json, backtest_json, backtest_mae, last_price, updated_at)
                VALUES (:symbol, :model, :steps, :forecast_json, :backtest_json, :mae, :last_price, NOW())
                ON CONFLICT (symbol, model, steps)
                DO UPDATE SET
                    forecast_json = EXCLUDED.forecast_json,
                    backtest_json = EXCLUDED.backtest_json,
                    backtest_mae = EXCLUDED.backtest_mae,
                    last_price = EXCLUDED.last_price,
                    updated_at = NOW();
            """), {
                "symbol": symbol,
                "model": model_name,
                "steps": horizon,
                "forecast_json": json.dumps(results_forecast),
                "backtest_json": json.dumps(results_backtest),
                "mae": mae,
                "last_price": last_price
            })

        print(f"✅ Updated DB: {symbol}-{model_name}-{horizon}d | MAE={mae}")
        return {"symbol": symbol, "model": model_name, "horizon": horizon, "status": "ok"}

    except Exception as e:
        print(f"[{model_name.upper()} Error] {e}")
        return {"symbol": symbol, "model": model_name, "horizon": horizon, "status": "error"}


# ==========================================================
# DAG SETUP
# ==========================================================
def read_latest(ti):
    pdf = pq.read_table(PARQUET_PATH).to_pandas()
    pdf.columns = [c.lower() for c in pdf.columns]
    df = pdf[pdf["symbol"].str.lower() == TICKERS[0].lower()].sort_values("date")
    last_close = float(df.iloc[-1]["close"])
    ti.xcom_push(key="last_close", value=last_close)
    print(f"[read_latest] last_close={last_close}")


def decide_branch(**ctx):
    return [f"forecast_group.{s}_{m}_{h}" for s in TICKERS for m in MODELS for h in CALENDAR_TO_BDAYS.keys()]


def merge_results(ti):
    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM stock_forecasts")).scalar()
        last = conn.execute(text("SELECT MAX(updated_at) FROM stock_forecasts")).scalar()
        print(f"📊 Total records: {total}, Last updated: {last}")


# ==========================================================
# DAG DEFINITION
# ==========================================================
with DAG(
    "forecast_stock_pipeline",
    default_args=default_args,
    description="Forecast DAG with ARIMA/SARIMA/SARIMAX/LSTM using Spark parquet + Postgres update",
    schedule_interval="30 19 * * 1-5",
    start_date=datetime(2025, 1, 1, tzinfo=tz_th),
    catchup=False,
) as dag:

    start = DummyOperator(task_id="start")

    spark_transform = SparkSubmitOperator(
        task_id="spark_transform",
        application="/opt/airflow/src/spark/applications/spark_jobs/pull_yf.py",
        conn_id="spark_default",
        name="pull-yfinance-to-parquet",
        verbose=True,
        conf={"spark.master": "local[*]"},
        application_args=["--period", "5y", "--tickers", ",".join(TICKERS), "--exog", "CL=F,NG=F,XLU"],
    )

    read_latest_task = PythonOperator(task_id="read_latest", python_callable=read_latest)
    branch = BranchPythonOperator(task_id="branching", python_callable=decide_branch, provide_context=True)

    with TaskGroup("forecast_group") as forecast_group:
        for sym in TICKERS:
            for model in MODELS:
                for h in CALENDAR_TO_BDAYS.keys():
                    PythonOperator(
                        task_id=f"{sym}_{model}_{h}",
                        python_callable=run_model,
                        op_kwargs={"symbol": sym, "model_name": model, "horizon": h},
                    )

    merge = PythonOperator(
        task_id="merge_results",
        python_callable=merge_results,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    end = DummyOperator(task_id="end")

    start >> spark_transform >> read_latest_task >> branch >> forecast_group >> merge >> end