"""Validated API inputs for the original UCI German Credit dataset."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreditFeatures(BaseModel):
    """The 18 features used by the trained preprocessing/model pipeline."""

    model_config = ConfigDict(extra="forbid")

    checking: Literal["A11", "A12", "A13", "A14"]
    duration: int = Field(ge=1, le=1200, description="Loan duration in months")
    credit_history: Literal["A30", "A31", "A32", "A33", "A34"]
    purpose: Literal[
        "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"
    ]
    amount: int = Field(ge=1, le=1_000_000_000, description="Credit amount in DM")
    savings: Literal["A61", "A62", "A63", "A64", "A65"]
    employment: Literal["A71", "A72", "A73", "A74", "A75"]
    installment_rate: int = Field(ge=1, le=4)
    other_debtors: Literal["A101", "A102", "A103"]
    residence_years: int = Field(ge=1, le=4)
    property: Literal["A121", "A122", "A123", "A124"]
    other_plans: Literal["A141", "A142", "A143"]
    housing: Literal["A151", "A152", "A153"]
    existing_credits: int = Field(ge=1, le=4)
    job: Literal["A171", "A172", "A173", "A174"]
    dependents: int = Field(ge=1, le=2)
    telephone: Literal["A191", "A192"]
    foreign_worker: Literal["A201", "A202"]
