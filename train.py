import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} trained and logged.")


def main():
    # Set tracking URI to your localhost
    mlflow.set_tracking_uri("http://0.0.0.0:5002")
    mlflow.set_experiment("Wine Quality Classification")

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # Train and log models
    train_models(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    main()