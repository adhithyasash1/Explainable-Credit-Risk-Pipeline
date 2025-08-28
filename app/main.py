import os
import json
import logging
import uuid
import pandas as pd
import numpy as np
import joblib
import mlflow
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Dict
from cachetools import TTLCache
from prometheus_fastapi_instrumentator import Instrumentator

# --- 1. Application and Model Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-xgb")
DEFAULT_MODEL_STAGE = os.getenv("DEFAULT_MODEL_STAGE", "Production")
LOSS_GIVEN_DEFAULT = 0.9

app = FastAPI(
    title="Production-Ready Credit Risk API",
    description="An API for predicting credit risk with model versioning, explainability, and business metrics.",
    version="2.0.0"
)

# --- 2. Monitoring & Caching Setup ---
logger = logging.getLogger("uvicorn.access")
Instrumentator().instrument(app).expose(app)
explanation_cache = TTLCache(maxsize=128, ttl=600)

# --- 3. Model Loading (Dynamic from MLflow Registry) ---
class ModelStore:
    model: any = None
    explainer: any = None
    feature_names: List[str] = None
    version: str = None
model_store = ModelStore()

@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        latest_version = client.get_latest_versions(MODEL_NAME, stages=[DEFAULT_MODEL_STAGE])[0]
        model_store.version = latest_version.version
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        model_store.model = mlflow.xgboost.load_model(model_uri)
        explainer_path = client.download_artifacts(latest_version.run_id, "explainer.joblib")
        model_store.explainer = joblib.load(explainer_path)
        features_path = client.download_artifacts(latest_version.run_id, "feature_names.txt")
        with open(features_path, "r") as f:
            model_store.feature_names = [line.strip() for line in f]
        logger.info(f"✅ Successfully loaded model '{MODEL_NAME}' version '{model_store.version}'.")
    except Exception as e:
        logger.error(f"❌ Failed to load model from MLflow Registry: {e}")
        model_store.model = None

# --- 4. Data Schemas ---
class CreditApp(BaseModel):
    Seniority: int
    Home: str
    Time: int
    Age: int
    Marital: str
    Records: str
    Job: str
    Expenses: int
    Income: float
    Assets: float
    Debt: float
    Amount: int
    Price: int

class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: int
    probability_good_credit: float
    risk_score: float
    risk_tier: str
    expected_loss: float

class FeatureContribution(BaseModel):
    feature: str
    contribution: float

class ExplanationResponse(PredictionResponse):
    top_features: List[FeatureContribution]

# --- 5. Feature Engineering and Helper Functions ---
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['Income'] = df['Income'].replace(0, 1)
    df['Price'] = df['Price'].replace(0, 1)
    df['Debt-to-Income'] = df['Debt'] / df['Income']
    df['Loan-to-Price'] = df['Amount'] / df['Price']
    df['Loan-to-Income'] = df['Amount'] / df['Income']
    df = pd.get_dummies(df, columns=['Home', 'Marital', 'Records', 'Job'], drop_first=True)
    return df

def get_risk_tier(p: float) -> str:
    if p > 0.8: return "Low"
    if p > 0.6: return "Medium"
    if p > 0.4: return "High"
    return "Very High"

# --- 6. API Endpoints ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    return await call_next(request)

@app.get("/health")
def health_check():
    if model_store.model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": model_store.version}

@app.post("/predict/{model_version}", response_model=PredictionResponse)
async def predict(request: Request, data: CreditApp, model_version: str = DEFAULT_MODEL_STAGE):
    if model_store.model is None: raise HTTPException(status_code=503, detail="Model not available")
    try:
        input_df = pd.DataFrame([data.dict()])
        processed_df = feature_engineer(input_df)
        final_df = processed_df.reindex(columns=model_store.feature_names, fill_value=0)
        
        proba_good_credit = model_store.model.predict_proba(final_df)[0][1]
        prediction = 1 if proba_good_credit >= 0.5 else 0
        
        proba_default = 1 - proba_good_credit
        risk_score = proba_default * 1000
        risk_tier = get_risk_tier(proba_good_credit)
        expected_loss = proba_default * LOSS_GIVEN_DEFAULT * data.Amount
        
        return {
            "request_id": request.state.request_id, "model_version": model_store.version,
            "prediction": prediction, "probability_good_credit": proba_good_credit,
            "risk_score": risk_score, "risk_tier": risk_tier, "expected_loss": expected_loss
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/{model_version}", response_model=ExplanationResponse)
async def explain(request: Request, data: CreditApp, model_version: str = DEFAULT_MODEL_STAGE):
    prediction_data = await predict(request, data, model_version)
    input_df = pd.DataFrame([data.dict()])
    processed_df = feature_engineer(input_df)
    final_df = processed_df.reindex(columns=model_store.feature_names, fill_value=0)
    shap_values = model_store.explainer.shap_values(final_df)
    contributions = [{"feature": f, "contribution": round(v, 4)} for f, v in zip(final_df.columns, shap_values[0])]
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return {**prediction_data.dict(), "top_features": contributions[:5]}
