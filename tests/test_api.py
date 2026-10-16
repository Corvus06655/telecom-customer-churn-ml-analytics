from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 844.2,
}


def test_health_and_metrics():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        metrics = client.get("/metrics").json()
    assert metrics["model"] == "logistic_regression"
    assert 0 <= metrics["accuracy"] <= 1


def test_prediction_contract():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in {"Yes", "No"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["retention_priority"] in {"high", "standard"}


def test_unknown_fields_are_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={**SAMPLE, "unexpected": 1})
    assert response.status_code == 422
