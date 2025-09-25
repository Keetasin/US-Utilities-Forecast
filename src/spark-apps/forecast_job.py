from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ForecastJob").getOrCreate()
    forecast = [("AEP", "ARIMA", 7, 95.1),
                ("DUK", "SARIMA", 90, 103.4)]
    df = spark.createDataFrame(forecast, ["symbol", "model", "steps", "forecast_price"])
    df.show()
    print("✅ ForecastJob finished successfully")
    spark.stop()
