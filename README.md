# Practical Task 6 — Deploy an ML Model with FastAPI and Docker

This project serves a logistic regression model trained on the German Credit dataset from Practical Task 5. The pipeline standardizes numeric features and one-hot encodes categorical features. Age and the combined personal status and sex field are excluded, as in Practical Task 5.

## Train the model

Use Python 3.13. Run these commands from the project folder.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

The project includes a pre-trained `model.joblib`. Running `train.py` regenerates it using the included dataset.

| Result | Value |
| --- | ---: |
| Training rows | 800 |
| Test rows | 200 |
| Test accuracy | 70.0% |

The split is stratified and uses random seed 42.

## Run locally

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In a second terminal, run the commands below from the project folder. Opening [Swagger UI](http://127.0.0.1:8000/docs) also lets you test the requests. For `POST /predict`, select **Try it out**, paste the contents of `example.json`, and select **Execute**.

```bash
curl -fsS http://127.0.0.1:8000/
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example.json
curl -fsS -o /dev/null -w 'Swagger HTTP %{http_code}\n' http://127.0.0.1:8000/docs
```

| Endpoint | Expected response |
| --- | --- |
| `GET /` | `{"message":"ML API is running"}` |
| `POST /predict` with `example.json` | `{"prediction":1,"label":"good_credit_risk"}` |
| `GET /docs` | Interactive Swagger UI, HTTP 200 |

The prediction is `1` for good credit risk and `0` for bad credit risk.

## Build and run with Docker

Start Docker Desktop or another Docker Engine. The image uses the included model artifact.

```bash
docker build -t german-credit-api .
docker run --rm -p 127.0.0.1:8001:8000 german-credit-api
```

Repeat the local checks using port `8001`. The container Swagger page is at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

## Dataset

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. The bundled `german.data` contains the original dataset and is provided under CC BY 4.0. [Dataset and license](https://doi.org/10.24432/C5NC77).
