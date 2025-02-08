from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.master("spark://26.130.203.43:7077").appName("CIFAR10_Training").getOrCreate()

# Load Training Data from HDFS
train_rdd = spark.sparkContext.textFile("file:///D:/Distributed_ML/data/cifar10_train.txt")


# Load the Pretrained Model on Each Node
def train_model(batch):
    model = load_model("cifar10_model.h5")

    # Convert batch to numpy arrays
    images, labels = zip(*batch)
    x_train = np.array(images)
    y_train = np.array(labels)

    # Train Model
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Save Model Weights
    model.save("cifar10_trained_model.h5")
    return "Trained a batch!"


# Apply Function to RDD
train_rdd.mapPartitions(train_model).collect()
