from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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
    data = pd.read_csv(PROJECT_DIR / "german.data", sep=r"\s+", names=COLUMNS)
    features = data.drop(columns=["credit_risk", "age", "personal_status_sex"])
    target = data["credit_risk"].eq(1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), make_column_selector(dtype_include="number")),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            make_column_selector(dtype_include="object"),
        ),
    ])
    model = make_pipeline(preprocessing, LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (class order: 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    model_path = PROJECT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved {model_path.name}")


if __name__ == "__main__":
    main()
