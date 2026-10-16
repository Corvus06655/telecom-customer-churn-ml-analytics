from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Customer.csv"

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in {"tenure", "MonthlyCharges", "TotalCharges"}]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[844.2])


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    retention_priority: str
    model: str
    disclaimer: str


class ModelService:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.metrics: dict[str, Any] = {}

    def train(self) -> None:
        df = pd.read_csv(DATA_PATH)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", "0"), errors="raise")
        X = df[FEATURE_COLUMNS].copy()
        y = df["Churn"].map({"No": 0, "Yes": 1})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), NUMERIC_COLUMNS),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ]
        )
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

    def predict(self, features: CustomerFeatures) -> tuple[int, float]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained")
        values = features.model_dump()
        frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        prediction = int(self.pipeline.predict(frame)[0])
        probability = float(self.pipeline.predict_proba(frame)[0][1])
        return prediction, probability


service = ModelService()
app = FastAPI(
    title="Telecom Customer Churn API",
    description="Predicts observed churn risk using the project's reproducible Logistic Regression benchmark.",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    service.train()


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"service": "Telecom Customer Churn API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok" if service.pipeline is not None else "starting"}


@app.get("/metrics", tags=["system"])
def metrics() -> dict[str, Any]:
    if not service.metrics:
        raise HTTPException(status_code=503, detail="Model is still starting")
    return {"model": "logistic_regression", **service.metrics}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: CustomerFeatures) -> PredictionResponse:
    try:
        prediction, probability = service.predict(features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(
        churn_prediction="Yes" if prediction else "No",
        churn_probability=round(probability, 4),
        retention_priority="high" if probability >= 0.5 else "standard",
        model="logistic_regression",
        disclaimer="This is a benchmark prediction on an anonymized extract, not a causal or production decision.",
    )
