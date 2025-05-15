import mlflow
import pandas as pd
from sklearn.datasets import load_wine


def main():
    mlflow.set_tracking_uri("http://0.0.0.0:5002")

    # Load sample data
    data = load_wine()
    sample = pd.DataFrame([data.data[0]], columns=data.feature_names)

    # Load model from registry
    model_name = "WineQualityClassifier"
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    # Make prediction
    prediction = model.predict(sample)
    print(f"Predicted class: {prediction[0]}")
    print(f"Actual class: {data.target[0]}")


if __name__ == "__main__":
    main()