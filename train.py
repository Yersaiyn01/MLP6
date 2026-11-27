import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent

COLUMNS = [
    "checking", "duration", "credit_history", "purpose", "amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_years", "property", "age", "other_plans",
    "housing", "existing_credits", "job", "dependents", "telephone",
    "foreign_worker", "credit_risk",
]


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "german-credit")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "german-credit-logistic-regression")
    model_path = Path(os.getenv("MODEL_PATH", str(PROJECT_DIR / "model.joblib"))).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path
    model_path = model_path.resolve()
    metadata_path = model_path.with_name("model-info.json")

    # Fail promptly when the tracking server is unavailable; explicit overrides win.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    data = pd.read_csv(PROJECT_DIR / "german.data", sep=r"\s+", names=COLUMNS)
    features = data.drop(columns=["credit_risk", "age", "personal_status_sex"])
    target = data["credit_risk"].eq(1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), X_train.select_dtypes(include="number").columns.tolist()),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            X_train.select_dtypes(include="object").columns.tolist(),
        ),
    ])
    classifier = LogisticRegression(max_iter=1000)
    model = make_pipeline(preprocessing, classifier)
    parameters = {
        "dataset": "German Credit",
        "algorithm": "LogisticRegression",
        "split_test_size": 0.2,
        "split_random_state": 42,
        "split_stratified": True,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": features.shape[1],
        "excluded_features": "age,personal_status_sex",
        "numeric_scaler": "StandardScaler",
        "categorical_encoding": "OneHotEncoder",
        "categorical_handle_unknown": "ignore",
        "metric_average": "macro",
        **{f"classifier_{key}": value for key, value in classifier.get_params(deep=False).items()},
    }

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with TemporaryDirectory(prefix="german-credit-training-") as temporary_dir:
            artifact_path = Path(temporary_dir) / "model.joblib"
            with mlflow.start_run(run_name="logistic-regression") as run:
                mlflow.log_params(parameters)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                metrics = {
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                    "precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
                    "recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
                }
                mlflow.log_metrics(metrics)
                mlflow.log_dict(
                    {
                        "labels": [0, 1],
                        "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
                    },
                    "confusion-matrix.json",
                )

                joblib.dump(model, artifact_path)
                mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")
                input_example = X_train.head(5)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=model_name,
                    await_registration_for=60,
                    signature=infer_signature(input_example, model.predict(input_example)),
                    input_example=input_example,
                    pip_requirements=[
                        f"{package}=={version(package)}"
                        for package in (
                            "scikit-learn", "pandas", "numpy", "scipy", "joblib", "cloudpickle", "skops",
                        )
                    ],
                )
                if model_info.registered_model_version is None:
                    raise MlflowException("The model was logged but no registered version was returned.")
                metadata = {
                    "model_name": model_name,
                    "model_version": str(model_info.registered_model_version),
                    "run_id": run.info.run_id,
                    "experiment_name": experiment_name,
                    "metric_average": "macro",
                    "metrics": metrics,
                }
                mlflow.log_dict(metadata, "model-info.json")

            # Publish local files only after tracking and registration succeed.
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_path, model_path)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except MlflowException as exc:
        raise SystemExit(
            "MLflow tracking or model registration failed. Start the MLflow server "
            "with a database-backed registry, check MLFLOW_TRACKING_URI and the server logs, "
            "then run train.py again.\n"
            f"Details: {exc}"
        ) from exc

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Test metrics (F1, precision and recall use macro averaging):")
    print(" | ".join(f"{name}: {value:.3f}" for name, value in metrics.items()))
    print("Confusion matrix (class order: 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Experiment: {experiment_name} | Run: {metadata['run_id']}")
    print(f"Registered model: {model_name} | Version: {metadata['model_version']}")
    print(f"Saved {model_path} and {metadata_path}")


if __name__ == "__main__":
    main()
