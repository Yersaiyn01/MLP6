from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

model = joblib.load(Path(__file__).with_name("model.joblib"))
app = FastAPI(title="German Credit ML API")


class CreditFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checking: str
    duration: int
    credit_history: str
    purpose: str
    amount: int
    savings: str
    employment: str
    installment_rate: int
    other_debtors: str
    residence_years: int
    property: str
    other_plans: str
    housing: str
    existing_credits: int
    job: str
    dependents: int
    telephone: str
    foreign_worker: str


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: CreditFeatures):
    row = pd.DataFrame([features.model_dump()], columns=model.feature_names_in_)
    prediction = int(model.predict(row)[0])
    label = "good_credit_risk" if prediction == 1 else "bad_credit_risk"
    return {"prediction": prediction, "label": label}
