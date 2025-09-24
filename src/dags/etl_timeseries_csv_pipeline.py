import os
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
spark_master = "spark://spark:7077"
postgres_driver_jar = "/usr/local/spark/assets/jars/postgresql-42.2.6.jar"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="etl_timeseries_csv_pipeline",
    default_args=default_args,
    description="ETL Time Series Data from Local CSV using Spark",
    schedule_interval="@daily",
    catchup=False,
) as dag:

    # Transform CSV with Spark
    transform_task = SparkSubmitOperator(
        task_id="transform_with_spark",
        application="/usr/local/spark/applications/spark_jobs/transform_timeseries.py",
        conn_id="spark_default",
        application_args=[
            "/usr/local/spark/applications/spark_jobs/raw_sales.csv",
            "/usr/local/spark/applications/spark_jobs/processed/"
        ],
        executor_memory="2g",
        driver_memory="1g",
    )

    # Load to PostgreSQL
    load_task = SparkSubmitOperator(
        task_id="load_to_postgres",
        application="/usr/local/spark/applications/spark_jobs/load_timeseries.py",
        name="load-postgres",
        conn_id="spark_default",
        verbose=1,
        conf={"spark.master": spark_master},
        application_args=[
            "/usr/local/spark/applications/spark_jobs/processed/",
            "jdbc:postgresql://172.24.0.2:5432/airflow",
            "timeseries_table",
            "airflow",
            "airflow"
        ],jars=postgres_driver_jar,
        driver_class_path=postgres_driver_jar,
        executor_memory="2g",
        driver_memory="1g",        
        dag=dag)
    
    validate_task = SparkSubmitOperator(
        task_id="validate_timeseries_data",
        application="/usr/local/spark/applications/spark_jobs/validate_timeseries.py",
        conn_id="spark_default",
        application_args=[
            "/usr/local/spark/applications/spark_jobs/raw_sales.csv"
        ],
        executor_memory="1g",
        driver_memory="1g",
    )

    from airflow.operators.python import PythonOperator
    import sys
    sys.path.append("/usr/local/spark/applications/spark_jobs")
    from forecast_timeseries import run_forecast

    # Forecast task พร้อม parameter จำนวนวัน
    forecast_task = PythonOperator(
        task_id="forecast_timeseries",
        python_callable=run_forecast,
        op_kwargs={"days_to_forecast": 14},  # ใช้ชื่อเดียวกับฟังก์ชัน
        dag=dag,
    )

    # DAG Chain
    validate_task >> transform_task >> load_task >> forecast_task


