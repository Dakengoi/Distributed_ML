from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, IntegerType
from pyspark.sql import Row
from tensorflow.keras.datasets import cifar10
import numpy as np
import time
import os

os.environ["SPARK_DRIVER_MEMORY"] = "2g"
os.environ["SPARK_EXECUTOR_MEMORY"] = "2g"

spark = SparkSession.builder \
    .appName("DistributedML") \
    .master("spark://26.130.203.43:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print("Spark session initialized")

start_time = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")


x_train, x_test = x_train / 255.0, x_test / 255.0



schema = StructType([
    StructField("features", ArrayType(ArrayType(ArrayType(FloatType()))), True),
    StructField("label", IntegerType(), True)
])

start_time = time.time()
train_data = [Row(features=x_train[i].tolist(), label=int(y_train[i][0])) for i in range(len(y_train))]
print(f"Converted dataset to list in {time.time() - start_time:.2f} seconds")

# Create Spark DataFrame
start_time = time.time()
train_df = spark.createDataFrame(train_data, schema=schema)
print(f"Created Spark DataFrame in {time.time() - start_time:.2f} seconds")

# Increase partition count to reduce memory usage per worker
num_partitions = 5
train_df = train_df.repartition(num_partitions)
print(f"Dataset repartitioned into {num_partitions} partitions")

# Save partitioned data in Parquet format (efficient & optimized storage)
output_path = "D:/Distributed_ML/data/partitioned_data"
start_time = time.time()
train_df.write.mode("overwrite").parquet(output_path)
print(f"Data saved in {time.time() - start_time:.2f} seconds at {output_path}")

# Stop Spark session
spark.stop()
print("Spark session stopped successfully")
