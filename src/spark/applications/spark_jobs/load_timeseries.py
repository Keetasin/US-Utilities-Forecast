from pyspark.sql import SparkSession
import sys

input_path = sys.argv[1]
db_url = sys.argv[2]
table_name = sys.argv[3]
db_user = sys.argv[4]
db_password = sys.argv[5]

spark = SparkSession.builder.appName("LoadTimeSeries").getOrCreate()

df = spark.read.parquet(input_path)

df.write \
    .format("jdbc") \
    .option("url", db_url) \
    .option("dbtable", table_name) \
    .option("user", db_user) \
    .option("password", db_password) \
    .mode("overwrite") \
    .save()
