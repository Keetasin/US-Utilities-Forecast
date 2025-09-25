from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("LoadDataJob").getOrCreate()
    data = [("AEP", 92.5), ("DUK", 101.2), ("SO", 68.7)]
    df = spark.createDataFrame(data, ["symbol", "price"])
    df.show()
    print("✅ LoadDataJob finished successfully")
    spark.stop()
