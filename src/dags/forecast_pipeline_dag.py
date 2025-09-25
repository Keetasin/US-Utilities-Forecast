from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime
import random

# ============= Branch logic =============
def decide_forecast(**context):
    """สุ่มตัดสินใจว่าจะทำ forecast ต่อหรือไม่"""
    choice = random.choice(["do_forecast", "skip_forecast"])
    print("Decision =", choice)
    return choice

# Push ค่า MAE ไป XCom
def push_mae(**context):
    mae = round(random.uniform(0.5, 2.0), 3)
    context['ti'].xcom_push(key='mae', value=mae)
    print("Pushed MAE =", mae)

# Pull ค่า MAE จาก XCom
def update_db(**context):
    mae = context['ti'].xcom_pull(task_ids='backtest_forecast', key='mae')
    print("Update DB with forecast result, MAE =", mae)

# ============= Define DAG =============
with DAG(
    dag_id="forecast_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["project", "forecast"],
) as dag:

    start = DummyOperator(task_id="start")

    # Spark jobs
    fetch_stock = SparkSubmitOperator(
        task_id="fetch_stock",
        application="/opt/spark-apps/load_data.py",
        conn_id="spark_default"
    )

    update_stock = SparkSubmitOperator(
        task_id="update_stock",
        application="/opt/spark-apps/update_stock.py",
        conn_id="spark_default"
    )

    forecast_pipeline = SparkSubmitOperator(
        task_id="forecast_pipeline",
        application="/opt/spark-apps/forecast_job.py",
        conn_id="spark_default"
    )

    branch = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=decide_forecast,
        provide_context=True,
    )

    backtest = PythonOperator(
        task_id="backtest_forecast",
        python_callable=push_mae,
        provide_context=True,
    )

    update_forecast = PythonOperator(
        task_id="update_forecast_db",
        python_callable=update_db,
        provide_context=True,
    )

    skip = DummyOperator(task_id="skip_forecast")
    end = DummyOperator(task_id="end")

    # Workflow
    start >> fetch_stock >> update_stock >> forecast_pipeline >> branch
    branch >> backtest >> update_forecast >> end
    branch >> skip >> end
