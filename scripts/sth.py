import numpy as np
import json
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values (scale from 0 to 1)
x_train = x_train.astype("float32") / 255.0
y_train = y_train.flatten()  # Convert labels to 1D array

# Convert dataset to a list of JSON strings (each line represents an image-label pair)
data = [json.dumps({"image": x.tolist(), "label": int(y)}) for x, y in zip(x_train, y_train)]

# Save to a text file
output_path = "D:/Distributed_ML/data/cifar10_train.txt"
with open(output_path, "w") as f:
    f.write("\n".join(data))

print(f"✅ CIFAR-10 training dataset saved at: {output_path}")
