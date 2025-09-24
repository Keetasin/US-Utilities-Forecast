import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import psycopg2
from psycopg2.extras import execute_values
import os

def run_forecast(days_to_forecast=7):
    input_path = "/usr/local/spark/applications/spark_jobs/processed/"
    output_path = "/usr/local/spark/applications/spark_jobs/forecast/"
    os.makedirs(output_path, exist_ok=True)

    # โหลดข้อมูล Parquet
    df = pd.read_parquet(input_path)
    df = df.rename(columns={"datetime": "ds", "price": "y"})
    df = df[["ds", "y"]].dropna()

    # สร้างโมเดล
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future)

    # วาดกราฟ 3 แบบ
    fig1, ax = plt.subplots(figsize=(10,6))
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.4)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Forecast with Confidence Interval')
    ax.legend()
    fig1.savefig(os.path.join(output_path, "forecast_plot1.png"))

    fig2 = model.plot(forecast, uncertainty=True)
    plt.scatter(df['ds'], df['y'], color='red', s=10, label='Actual')
    plt.legend()
    fig2.savefig(os.path.join(output_path, "forecast_plot2.png"))

    fig3 = model.plot_components(forecast)
    fig3.savefig(os.path.join(output_path, "forecast_plot3.png"))

    # บันทึกผลลง PostgreSQL
    conn = psycopg2.connect(
        host="172.24.0.2",
        dbname="airflow",
        user="airflow",
        password="airflow"
    )
    cur = conn.cursor()
    table_name = "forecast_timeseries"
    # ลบข้อมูลเก่า
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(f"""
        CREATE TABLE {table_name} (
            ds TIMESTAMP,
            yhat DOUBLE PRECISION,
            yhat_lower DOUBLE PRECISION,
            yhat_upper DOUBLE PRECISION
        )
    """)
    values = list(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].itertuples(index=False, name=None))
    execute_values(cur, f"INSERT INTO {table_name} (ds, yhat, yhat_lower, yhat_upper) VALUES %s", values)
    conn.commit()
    cur.close()
    conn.close()

    print("Forecast saved to PostgreSQL and plots generated.")
