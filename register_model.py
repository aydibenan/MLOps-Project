import mlflow
from mlflow.tracking import MlflowClient


def main():
    mlflow.set_tracking_uri("http://0.0.0.0:5002")

    # Get the best run
    experiment = mlflow.get_experiment_by_name("Wine Quality Classification")
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["metrics.accuracy DESC"])
    best_run = runs.iloc[0]

    # Register model
    model_uri = f"runs:/{best_run.run_id}/model"
    model_name = "WineQualityClassifier"

    mlflow.register_model(model_uri, model_name)

    # Transition to Production
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name)

    for version in latest_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Production"
        )

    print(f"Model {model_name} registered and transitioned to Production.")


if __name__ == "__main__":
    main()