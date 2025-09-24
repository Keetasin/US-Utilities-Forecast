from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import sys

# รับ input path จาก argument
input_path = sys.argv[1]

# สร้าง SparkSession
spark = SparkSession.builder.appName("ValidateTimeSeries").getOrCreate()

# โหลด CSV
df = spark.read.csv(input_path, header=True, inferSchema=True)

# -------------------------------
# เช็คคอลัมน์ที่ต้องมี
# -------------------------------
required_cols = ["datesold", "price"]
for col_name in required_cols:
    if col_name not in df.columns:
        raise Exception(f"Missing column: {col_name}")

# -------------------------------
# เช็ค missing values
# -------------------------------
for col_name in required_cols:
    if df.filter(col(col_name).isNull()).count() > 0:
        raise Exception(f"Null values in column: {col_name}")

# -------------------------------
# เช็ค column price > 0
# -------------------------------
invalid_price_count = df.filter(col("price") <= 0).count()
if invalid_price_count > 0:
    raise Exception(f"Found {invalid_price_count} rows where price <= 0")

# -------------------------------
# Passed all validations
# -------------------------------
print("Validation passed. All required columns exist, no nulls, and price > 0.")
