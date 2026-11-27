"""Streamlit interface for the German Credit FastAPI prediction service."""

import os

import requests
import streamlit as st
from pydantic import ValidationError

from schemas import CreditFeatures


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Labels follow the original UCI German Credit documentation. The original
# dataset uses Deutsche Mark (DM), so amounts are not converted to a new unit.
# https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata
CATEGORY_LABELS = {
    "checking": {
        "A11": "Balance below 0 DM",
        "A12": "Balance from 0 to under 200 DM",
        "A13": "At least 200 DM / salary assigned for at least 1 year",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No previous credit / all credit repaid on time",
        "A31": "All credit at this bank repaid on time",
        "A32": "Existing credit repaid on time so far",
        "A33": "Previous repayment delay",
        "A34": "Critical account / existing credit at another bank",
    },
    "purpose": {
        "A40": "New car",
        "A41": "Used car",
        "A42": "Furniture or equipment",
        "A43": "Radio or television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Other purpose",
    },
    "savings": {
        "A61": "Under 100 DM",
        "A62": "100 to under 500 DM",
        "A63": "500 to under 1,000 DM",
        "A64": "At least 1,000 DM",
        "A65": "Unknown / no savings account",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "Under 1 year",
        "A73": "1 to under 4 years",
        "A74": "4 to under 7 years",
        "A75": "At least 7 years",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Savings agreement or life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / no property",
    },
    "other_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own home", "A153": "Provided free"},
    "job": {
        "A171": "Unemployed or unskilled, non-resident",
        "A172": "Unskilled, resident",
        "A173": "Skilled employee or official",
        "A174": "Management, self-employed or highly qualified",
    },
    "telephone": {"A191": "None", "A192": "Registered in applicant's name"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


def category_input(label: str, field: str, default: str) -> str:
    labels = CATEGORY_LABELS[field]
    options = list(labels)
    return st.selectbox(
        label,
        options,
        index=options.index(default),
        format_func=labels.__getitem__,
        key=field,
    )


def get_model_info(api_url: str) -> dict:
    """Metadata is optional so the form also works with the original API."""
    try:
        response = requests.get(f"{api_url}/model-info", timeout=3)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except (requests.RequestException, ValueError):
        return {}


def request_prediction(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or type(result.get("prediction")) is not int:
        raise ValueError("The service returned an invalid prediction.")
    prediction = result["prediction"]
    if prediction not in (0, 1):
        raise ValueError("The service returned an unknown prediction class.")
    expected_label = "good_credit_risk" if prediction == 1 else "bad_credit_risk"
    if result.get("label") != expected_label:
        raise ValueError("The service returned an inconsistent prediction label.")
    return result


def show_service_error(error: requests.HTTPError) -> None:
    if error.response is not None and error.response.status_code == 422:
        st.error("Some input values were rejected. Check the form and try again.")
    else:
        st.error("The prediction service could not process the request. Please try again.")


def main() -> None:
    st.set_page_config(page_title="German Credit | ML Lab", layout="wide")
    st.caption("GERMAN CREDIT / SIS-3")
    st.title("Credit risk explorer")
    st.write("Enter the applicant's details to get a prediction from the trained model.")

    model_info = get_model_info(API_URL)
    metrics = model_info.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    if model_info:
        version = model_info.get("model_version")
        version_column, accuracy_column, f1_column = st.columns(3)
        version_column.metric("Model version", str(version) if version else "Bundled model")
        accuracy = metrics.get("accuracy")
        accuracy_column.metric(
            "Test accuracy", f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else "Not recorded"
        )
        f1 = metrics.get("f1")
        f1_column.metric("Test F1 (macro)", f"{f1:.3f}" if isinstance(f1, (int, float)) else "Not recorded")
    else:
        st.caption("Model details will appear when the prediction service is available.")

    st.caption("The form starts with the bundled example. Amounts use the original dataset's DM units.")
    with st.form("credit_application", clear_on_submit=False):
        st.subheader("Loan details")
        left, right = st.columns(2)
        with left:
            duration = st.number_input(
                "Duration (months)", min_value=1, max_value=1200, value=6, step=1, key="duration"
            )
            purpose = category_input("Loan purpose", "purpose", "A43")
        with right:
            amount = st.number_input(
                "Credit amount (DM)", min_value=1, max_value=1_000_000_000,
                value=1169, step=1, key="amount",
            )
            installment_rate = st.number_input(
                "Installment rate (% of disposable income)", min_value=1, max_value=4,
                value=4, step=1, key="installment_rate",
            )

        st.subheader("Accounts and credit history")
        left, middle, right = st.columns(3)
        with left:
            checking = category_input("Checking account", "checking", "A11")
            savings = category_input("Savings account", "savings", "A65")
        with middle:
            credit_history = category_input("Credit history", "credit_history", "A34")
            existing_credits = st.number_input(
                "Existing credits at this bank", min_value=1, max_value=4,
                value=2, step=1, key="existing_credits",
            )
        with right:
            other_plans = category_input("Other installment plans", "other_plans", "A143")
            other_debtors = category_input("Other debtors or guarantors", "other_debtors", "A101")

        st.subheader("Household and employment")
        left, middle, right = st.columns(3)
        with left:
            employment = category_input("Time in current employment", "employment", "A75")
            job = category_input("Occupation", "job", "A173")
            foreign_worker = category_input("Foreign worker", "foreign_worker", "A201")
        with middle:
            housing = category_input("Housing", "housing", "A152")
            property_value = category_input("Property", "property", "A121")
            residence_years = st.number_input(
                "Residence duration (years)", min_value=1, max_value=4,
                value=4, step=1, key="residence_years",
                help="The bundled dataset records values from 1 to 4.",
            )
        with right:
            dependents = st.number_input(
                "People supported financially", min_value=1, max_value=2,
                value=1, step=1, key="dependents",
                help="The bundled dataset records values of 1 or 2.",
            )
            telephone = category_input("Telephone", "telephone", "A192")

        submitted = st.form_submit_button(
            "Predict credit risk", type="primary", width="stretch", key="predict"
        )

    if submitted:
        st.session_state.pop("prediction_result", None)
        payload = {
            "checking": checking,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "amount": amount,
            "savings": savings,
            "employment": employment,
            "installment_rate": installment_rate,
            "other_debtors": other_debtors,
            "residence_years": residence_years,
            "property": property_value,
            "other_plans": other_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "dependents": dependents,
            "telephone": telephone,
            "foreign_worker": foreign_worker,
        }
        try:
            payload = CreditFeatures.model_validate(payload).model_dump()
            with st.spinner("Getting prediction..."):
                st.session_state["prediction_result"] = request_prediction(payload)
        except ValidationError:
            st.error("Please check the input values and try again.")
        except requests.Timeout:
            st.error("The prediction service took too long to respond. Please try again.")
        except requests.ConnectionError:
            st.error("The prediction service is unavailable. Start the project services and try again.")
        except requests.HTTPError as error:
            show_service_error(error)
        except (requests.RequestException, ValueError):
            st.error("The prediction service returned an unexpected response. Please try again.")

    result = st.session_state.get("prediction_result")
    if result:
        st.subheader("Last submitted prediction")
        if result["prediction"] == 1:
            st.success("Predicted credit risk: good")
        else:
            st.warning("Predicted credit risk: bad")

    st.caption("Data: UCI Statlog (German Credit Data). Age and personal status/sex are excluded from the model.")


if __name__ == "__main__":
    main()
