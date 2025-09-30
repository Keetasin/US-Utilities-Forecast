import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------
# Project paths
# ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # src
sys.path.append('/usr/local/airflow/web')  # absolute path ใน container

from web import create_app
from web.utils.forecast import update_forecast
from web.utils.stock import TICKERS

# ---------------------------
# DAG default args
# ---------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ---------------------------
# DAG definition
# ---------------------------
dag = DAG(
    'forecast_pipeline_test03',
    default_args=default_args,
    description='Forecast stock prices manually',
    schedule_interval=None,   # manual trigger
    start_date=datetime.now(),
    catchup=False,
    tags=['stocks', 'forecast'],
)

# ---------------------------
# Python callable tasks
# ---------------------------
def run_update_forecast_task(symbols=None):
    app = create_app()
    symbols = symbols or TICKERS
    # update_forecast(app, tickers=symbols, models=["arima","sarima","sarimax","lstm"], steps_list=[7,90,365])
    update_forecast(app, tickers=symbols, models=["arima"], steps_list=[7,90,365])

# ---------------------------
# Airflow tasks
# ---------------------------
t_update_forecast = PythonOperator(
    task_id='update_forecast_all',
    python_callable=run_update_forecast_task,
    dag=dag,
)

# ---------------------------
# Task dependencies
# ---------------------------
# ตอนนี้ task เดียว ไม่มี dependency
