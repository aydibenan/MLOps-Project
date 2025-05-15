import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
import numpy as np
import time


def simulate_production_data():
    data = load_wine()
    for i in range(100):
        idx = np.random.randint(0, len(data.data))
        yield data.data[idx], data.target[idx]


def main():
    mlflow.set_tracking_uri("http://0.0.0.0:5002")
    client = MlflowClient()

    # Load model
    model_name = "WineQualityClassifier"
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    # Simulate monitoring
    with mlflow.start_run(run_name="Production Monitoring"):
        accuracies = []
        for i, (features, true_label) in enumerate(simulate_production_data()):
            # Predict
            pred = model.predict([features])[0]

            # Calculate accuracy
            accuracies.append(1 if pred == true_label else 0)

            # Log every 10 predictions
            if i % 10 == 0 and i > 0:
                current_acc = sum(accuracies[-10:]) / 10
                mlflow.log_metric("rolling_accuracy", current_acc, step=i)
                print(f"Rolling accuracy at step {i}: {current_acc:.2f}")

            time.sleep(0.1)  # Simulate real-time delay


if __name__ == "__main__":
    main()