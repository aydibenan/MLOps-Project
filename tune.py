import mlflow
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

def main():
    mlflow.set_tracking_uri("http://0.0.0.0:5002")
    mlflow.set_experiment("Wine Quality Tuning")

    X, y = load_data()

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'random_state': 42
    }

    with mlflow.start_run(run_name="Random Forest Tuning"):
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials()
        )

        print("Best parameters:", best)

        # Log best parameters
        mlflow.log_params(best)


def load_data():
    data = load_wine()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def objective(params):
    X, y = load_data()
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params)
        score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", score)

        # Since Hyperopt minimizes, we return negative accuracy
        return -score





if __name__ == "__main__":
    main()