from kafka import KafkaConsumer
import json
import numpy as np
import tensorflow as tf

# Create Kafka consumer
consumer = KafkaConsumer(
    'cifar10_training',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

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

# Process messages from Kafka
x_train, y_train = [], []
max_samples = 5000  # Limit to 5,000 samples

for message in consumer:
    data = message.value
    x_train.append(np.array(data["features"]))
    y_train.append(data["label"])

    # Stop after collecting 5000 samples
    if len(x_train) >= max_samples:
        break

x_train, y_train = np.array(x_train), np.array(y_train)
print(f"Loaded {len(x_train)} samples from Kafka.")

# Train model
model = create_model()
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Save trained model
model.save("D:/Distributed_ML/data/final_model_from_kafka.h5")
print("Model trained and saved from Kafka data.")
