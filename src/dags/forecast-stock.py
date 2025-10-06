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
import pendulum  # ✅ ใช้สำหรับตั้ง timezone



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
# กำหนด timezone เป็น Asia/Bangkok
tz_th = pendulum.timezone("Asia/Bangkok")

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
    """Train + forecast + backtest + save to Postgres (aligned with web forecast.py)"""
    try:
        steps = CALENDAR_TO_BDAYS[horizon]
        period = get_train_period(horizon)
        model_name = model_name.lower()

        # --- Load price data ---
        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)["Close"].dropna()
        data = data.asfreq("B").ffill()
        if len(data) < max(70, steps + 10):
            print(f"[Forecast] Skip {symbol}-{model_name}-{horizon}d (not enough data)")
            return None

        # --- Load exogenous variables ---
        EXOG_TICKERS = {"oil": "CL=F", "gas": "NG=F", "xlu": "XLU"}
        exog = None
        if model_name in ["sarimax", "lstm"]:
            exog_df = pd.DataFrame()
            for name, tkr in EXOG_TICKERS.items():
                try:
                    s = yf.download(tkr, period=period, progress=False, auto_adjust=True)["Close"].dropna()
                    s = s.asfreq("B").ffill()
                    exog_df[name] = s
                except Exception as e:
                    print(f"[Exog] Failed {tkr}: {e}")
            exog_df = exog_df.reindex(data.index).ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(0)
            exog = exog_df

        # --- Get params ---
        MODEL_PARAMS = {
            "AEP": {"arima": (1,1,0), "sarima": (0,1,0,20), "sarimax": (2,0,2,20)},
            "DUK": {"arima": (1,1,1), "sarima": (2,1,2,63), "sarimax": (0,1,2,63)},
            "SO":  {"arima":  (2, 1, 0), "sarima": (1,0,0,20), "sarimax": (0,1,2,20)},
            "ED":  {"arima": (2,1,2), "sarima": (2,0,2,63), "sarimax": (2,1,1,63)},
            "EXC": {"arima": (2,1,2), "sarima": (1,1,1,20), "sarimax": (1,0,0,63)},
        }

        def get_orders(sym, model):
            conf = MODEL_PARAMS.get(sym, {})
            if model == "arima": return (conf.get("arima", (2,1,0)), None)
            elif model == "sarima":
                o = (conf.get("sarima", (1,1,1,20)))
                return ((o[0],o[1],o[2]), (1,1,0,o[3]))
            elif model == "sarimax":
                o = (conf.get("sarimax", (1,1,1,20)))
                return ((o[0],o[1],o[2]), (1,1,0,o[3]))
            else:
                return ((1,1,1), (1,1,0,20))

        order, seasonal_order = get_orders(symbol, model_name)

        results_forecast, results_backtest = [], []
        mae = None

        # ================================
        #  Classical Models
        # ================================
        if model_name == "arima":
            m = ARIMA(data, order=order).fit()
            fc = m.forecast(steps=steps)

        elif model_name == "sarima":
            m = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = m.forecast(steps=steps)

        elif model_name == "sarimax":
            if exog is None:
                print(f"[SARIMAX] Skip {symbol}, no exogenous data")
                return None
            m = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        exog=exog, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            future_exog = exog.iloc[-steps:].copy()
            fc = m.forecast(steps=steps, exog=future_exog)

        # ================================
        #  LSTM Model (light version)
        # ================================
        elif model_name == "lstm":
            from sklearn.preprocessing import MinMaxScaler
            values = data.values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)
            lookback = 20

            X, y = [], []
            for i in range(len(scaled) - lookback):
                X.append(scaled[i:i+lookback])
                y.append(scaled[i+lookback])
            X, y = np.array(X), np.array(y)

            model = Sequential([
                LSTM(32, input_shape=(lookback, 1)),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=1, verbose=0)

            last_seq = X[-1]
            preds = []
            for _ in range(steps):
                p = model.predict(last_seq.reshape(1, lookback, 1), verbose=0)[0][0]
                preds.append(p)
                new_seq = np.append(last_seq[1:], [[p]], axis=0)
                last_seq = new_seq
            fc = pd.Series(scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten(),
                           index=pd.bdate_range(start=data.index[-1] + pd.offsets.BDay(1), periods=steps))

        else:
            print(f"[Forecast] Unknown model: {model_name}")
            return None

        # --- Convert forecast to list of dicts ---
        for d, v in zip(pd.bdate_range(start=data.index[-1] + pd.offsets.BDay(1), periods=steps), fc):
            results_forecast.append({"date": d.strftime("%Y-%m-%d"), "price": float(v)})

        # ================================
        # Backtest (last 5 BDays)
        # ================================
        train, test = data.iloc[:-5], data.iloc[-5:]
        if len(train) > 30:
            if model_name == "lstm":
                preds = np.array([test.mean()] * len(test))
            elif model_name == "sarimax" and exog is not None:
                bt = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                             exog=exog.iloc[:-5], enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                preds = bt.forecast(steps=5, exog=exog.iloc[-5:])
            else:
                bt = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                preds = bt.forecast(steps=5)

            mae = mean_absolute_error(test.values, preds)
            for d, v, a in zip(test.index, preds, test.values):
                results_backtest.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "actual_price": float(a),
                    "predicted_price": float(v)
                })


        # ================================
        # Save to Postgres
        # ================================
        # ================================
        # Save to Postgres (aligned with web app)
        # ================================
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import inspect

        engine = create_engine(PG_CONN, isolation_level="AUTOCOMMIT")
        Session = sessionmaker(bind=engine)
        session = Session()

        # --- create table if not exists ---
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

        # --- prepare record ---
        record = {
            "symbol": symbol,
            "model": model_name,
            "steps": horizon,  # ใช้ key 7, 180, 365
            "forecast_json": json.dumps(results_forecast),
            "backtest_json": json.dumps(results_backtest) if results_backtest else None,
            "backtest_mae": mae,
            "last_price": float(data.iloc[-1]) if len(data) else None,
            "updated_at": datetime.now()
        }

        # --- UPSERT (insert or update) ---
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO stock_forecasts (symbol, model, steps, forecast_json, backtest_json, backtest_mae, last_price, updated_at)
                    VALUES (:symbol, :model, :steps, :forecast_json, :backtest_json, :backtest_mae, :last_price, :updated_at)
                    ON CONFLICT (symbol, model, steps)
                    DO UPDATE SET
                        forecast_json = EXCLUDED.forecast_json,
                        backtest_json = EXCLUDED.backtest_json,
                        backtest_mae = EXCLUDED.backtest_mae,
                        last_price = EXCLUDED.last_price,
                        updated_at = EXCLUDED.updated_at;
                """), record)

            print(f"✅ Saved {symbol}-{model_name}-{horizon}d → Postgres (MAE={mae:.4f})" if mae else
                  f"✅ Saved {symbol}-{model_name}-{horizon}d → Postgres")
        except Exception as e:
            print(f"[Save Error] {e}")
        finally:
            session.close()


    except Exception as e:
        print(f"[{model_name.upper()} Error] {e}")


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
    "forecast_stock_pipeline",
    default_args=default_args,
    description="Forecast DAG with ARIMA/SARIMA/SARIMAX/LSTM + Spark + Branch + XCom + Backtest + Horizons",
    schedule_interval="47 09 * * 1-5",   # ⏰ 
    start_date=datetime(2025, 1, 1, tzinfo=tz_th),
    catchup=False,
    tags=["assignment","forecast"],
) as dag:

    start = DummyOperator(task_id="start")

    spark_transform = SparkSubmitOperator(
        task_id="spark_transform",
        application="/opt/airflow/spark/applications/spark_jobs/forecast-stock_timeseries.py",
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