import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
import os

H5_MODEL_PATH = "D:/Distributed_ML/data/partitioned_data/final_model.h5"
KERAS_MODEL_PATH = "D:/Distributed_ML/data/partitioned_data/final_model.keras"
SAMPLE_IMAGE_PATH = "sample_image.png"

if not os.path.exists(H5_MODEL_PATH):
    print(f"Model file not found: {H5_MODEL_PATH}")
    exit(1)

print("Loading final model...")
final_model = tf.keras.models.load_model(H5_MODEL_PATH)
final_model.summary()

for i, layer in enumerate(final_model.weights):
    print(f"Layer {i} shape: {layer.shape}")


print("\nLoading CIFAR-10 test dataset...")
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

print("Evaluating model on test data...")
test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=2)
print(f"Model Test Accuracy: {test_acc * 100:.2f}%")

print("\nConverting model to .keras format...")
final_model.save(KERAS_MODEL_PATH)
print(f"Model converted and saved as: {KERAS_MODEL_PATH}")

if os.path.exists(SAMPLE_IMAGE_PATH):
    print("\nRunning inference on a sample image...")


    img = cv2.imread(SAMPLE_IMAGE_PATH)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    print(f"Predicted Class: {predicted_class}")
else:
    print("No sample image found. Place an image at 'sample_image.png' to test inference.")

print("\nAll steps completed successfully!")
