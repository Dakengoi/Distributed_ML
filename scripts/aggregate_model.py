import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load Models from Worker Nodes
models = [load_model(f"cifar10_model.h5") for i in range(4)]

# Average Weights
new_weights = np.mean([model.get_weights() for model in models], axis=0)

# Load the Original Model
global_model = load_model("cifar10_model.h5")

# Set New Weights
global_model.set_weights(new_weights)

# Save Final Aggregated Model
global_model.save("cifar10_final_model.h5")
