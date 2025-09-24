from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg
from pyspark.sql.window import Window
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

spark = SparkSession.builder.appName("TransformTimeSeries").getOrCreate()

# Load CSV
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Clean & transform
df = df.withColumn("datetime", to_timestamp(col("datesold"), "yyyy-MM-dd HH:mm:ss"))
df = df.na.drop()   # ลบ missing values

# Rolling average feature
windowSpec = Window.orderBy("datetime").rowsBetween(-3, 0)
df = df.withColumn("rolling_avg", avg("price").over(windowSpec))

# Save processed data
df.write.mode("overwrite").parquet(output_path)
