"""Check the running API, frontend, MLflow run, artifacts, and model version.

Uses only the Python standard library. Start the stack before running this file.
"""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]


def request(base, path, payload=None, expected_status=200):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = Request(base.rstrip("/") + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=15) as response:
            status, content = response.status, response.read()
    except HTTPError as error:
        status, content = error.code, error.read()
    if status != expected_status:
        raise AssertionError(f"{path}: expected HTTP {expected_status}, got {status}: {content.decode()[:500]}")
    return content


def get_json(base, path, payload=None):
    return json.loads(request(base, path, payload))


def verify(api_url, tracking_uri, frontend_url):
    assert get_json(api_url, "/") == {"message": "ML API is running"}
    assert b"swagger-ui" in request(api_url, "/docs")
    example = json.loads((PROJECT_DIR / "example.json").read_text())
    assert get_json(api_url, "/predict", example) == {
        "prediction": 1,
        "label": "good_credit_risk",
    }
    for invalid in (
        {**example, "amount": -1},
        {**example, "checking": "INVALID"},
        {**example, "age": 30},
        {key: value for key, value in example.items() if key != "amount"},
    ):
        request(api_url, "/predict", invalid, expected_status=422)
    print("PASS: API predictions, Swagger, and input validation")

    metadata = get_json(api_url, "/model-info")
    assert metadata.get("model_version"), "Train a registered model and restart the API first."
    assert metadata.get("run_id"), "The served model is missing its MLflow run ID."
    run = get_json(
        tracking_uri, "/api/2.0/mlflow/runs/get?" + urlencode({"run_id": metadata["run_id"]})
    )["run"]
    assert run["info"]["status"] == "FINISHED", run["info"]["status"]
    params = {item["key"]: item["value"] for item in run["data"]["params"]}
    assert {"classifier_C", "classifier_max_iter", "split_random_state", "metric_average"} <= params.keys()
    assert params["metric_average"] == "macro"
    metrics = {item["key"]: item["value"] for item in run["data"]["metrics"]}
    for name in ("accuracy", "f1", "precision", "recall"):
        assert math.isclose(metrics[name], metadata["metrics"][name], abs_tol=1e-12), name
    print("PASS: finished MLflow experiment with hyperparameters and all four metrics")

    artifact_listing = get_json(
        tracking_uri,
        "/api/2.0/mlflow/artifacts/list?"
        + urlencode({"run_id": metadata["run_id"], "path": "artifacts"}),
    )
    assert any(
        item["path"] == "artifacts/model.joblib" and int(item.get("file_size", 0)) > 0
        for item in artifact_listing.get("files", [])
    ), "Saved model artifact is missing."
    registered = get_json(
        tracking_uri,
        "/api/2.0/mlflow/model-versions/get?"
        + urlencode({"name": metadata["model_name"], "version": metadata["model_version"]}),
    )["model_version"]
    assert registered["status"] == "READY", registered["status"]
    assert registered["run_id"] == metadata["run_id"]
    print(f"PASS: model artifact and registered {metadata['model_name']} version {metadata['model_version']}")

    assert request(frontend_url, "/_stcore/health").strip() == b"ok"
    assert b"<html" in request(frontend_url, "/").lower()
    print("PASS: Streamlit frontend is serving")
    print("All SIS-3 integration checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    try:
        verify(args.api_url, args.tracking_uri, args.frontend_url)
    except (AssertionError, KeyError, ValueError, URLError) as error:
        raise SystemExit(f"Verification failed: {error}") from error


if __name__ == "__main__":
    main()
