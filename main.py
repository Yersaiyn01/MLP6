import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from schemas import CreditFeatures

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_DIR / "model.joblib"))).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_DIR / MODEL_PATH

model = joblib.load(MODEL_PATH)
metadata_path = MODEL_PATH.with_name("model-info.json")
model_info = (
    json.loads(metadata_path.read_text())
    if metadata_path.is_file()
    else {
        "model_name": "german-credit-logistic-regression",
        "model_version": None,
        "run_id": None,
        "experiment_name": None,
        "metrics": {},
    }
)

app = FastAPI(
    title="German Credit ML API",
    description="Credit risk predictions with a model tracked and registered in MLflow.",
)


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def get_model_info():
    """Describe the model loaded by this API process."""
    return model_info


@app.post("/predict")
def predict(features: CreditFeatures):
    row = pd.DataFrame([features.model_dump()], columns=model.feature_names_in_)
    prediction = int(model.predict(row)[0])
    label = "good_credit_risk" if prediction == 1 else "bad_credit_risk"
    return {"prediction": prediction, "label": label}
