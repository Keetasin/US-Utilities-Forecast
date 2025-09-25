from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("UpdateStockJob").getOrCreate()
    # สมมติว่าอัพเดต DB (mock)
    print("✅ UpdateStockJob finished successfully (mock DB update)")
    spark.stop()
