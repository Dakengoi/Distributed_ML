from tensorflow.keras.datasets import cifar10
import numpy as np
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.master("spark://26.130.203.43:7077").appName("CIFAR10_Distributed").getOrCreate()

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize Data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert to List for Distribution
train_data = list(zip(x_train.tolist(), y_train.tolist()))

# Parallelize Data using Spark RDDs
train_rdd = spark.sparkContext.parallelize(train_data, numSlices=4)

# Save RDD for Further Processing
train_rdd.saveAsTextFile("hdfs://your-hadoop-cluster/cifar10_train")
