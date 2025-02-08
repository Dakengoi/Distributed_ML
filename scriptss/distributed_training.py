from pyspark.sql import SparkSession
import tensorflow as tf


spark = SparkSession.builder \
   .appName("DistributedTensorFlow") \
   .master("spark://26.130.203.43:7077") \
   .config("spark.executor.memory", "6g") \
   .config("spark.driver.memory", "6g") \
   .getOrCreate()


print("Spark session initialized.")


train_df = spark.read.parquet("D:/Distributed_ML/data/partitioned_data")


def create_model():
   model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
       tf.keras.layers.MaxPooling2D((2, 2)),
       tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D((2, 2)),
       tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   return model


def train_partition(partition):
   import tensorflow as tf
   import numpy as np


   data = list(partition)
   if len(data) == 0:
       return []


   x_train = np.array([np.array(row["features"]) for row in data])
   y_train = np.array([row["label"] for row in data])


   strategy = tf.distribute.MirroredStrategy()


   with strategy.scope():
       model = create_model()
       model.fit(x_train, y_train, epochs=5, batch_size=64)


   model.save(f"D:/Distributed_ML/data/worker_model_{np.random.randint(1000)}.h5")


   return [("Success", len(data))]


result = train_df.rdd.mapPartitions(train_partition).collect()


spark.stop()
print("Spark training completed successfully!")
