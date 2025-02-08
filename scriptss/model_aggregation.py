import tensorflow as tf
import numpy as np
import glob

model_path = "D:/Distributed_ML/data/worker_model_*.h5"

model_files = glob.glob(model_path)

if not model_files:
    print("No worker models found! Check your directory path:", model_path)
    exit(1)

worker_models = [tf.keras.models.load_model(f) for f in model_files]
print(f"Loaded {len(worker_models)} models for aggregation.")


def aggregate_models(worker_models):
    """Aggregates model weights from different nodes using Federated Averaging."""
    avg_weights = []

    for layer_weights in zip(*[model.get_weights() for model in worker_models]):
        avg_weights.append(np.mean(layer_weights, axis=0))

    return avg_weights


final_weights = aggregate_models(worker_models)

global_model = worker_models[0]
global_model.set_weights(final_weights)

final_model_path = "D:/Distributed_ML/data/partitioned_data/final_model.h5"
global_model.save(final_model_path)

print(f"Final model saved successfully at: {final_model_path}")
