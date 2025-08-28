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
async def load_model():
    """Load model asynchronously with timeout protection"""
    import asyncio
    
    # Load feature names first as fallback
    try:
        with open("feature_names.txt", "r") as f:
            model_store.feature_names = [line.strip() for line in f]
        logger.info("Loaded fallback feature names")
    except Exception as e:
        logger.error(f"Could not load feature names file: {e}")
        # Hard-coded fallback based on your feature_names.txt
        model_store.feature_names = [
            "Seniority", "Time", "Age", "Expenses", "Income", "Assets", "Debt", "Amount", "Price",
            "Debt-to-Income", "Loan-to-Price", "Loan-to-Income",
            "Home_1", "Home_2", "Home_3", "Home_4", "Home_5", "Home_6",
            "Marital_1", "Marital_2", "Marital_3", "Marital_4", "Marital_5",
            "Records_2", "Job_1", "Job_2", "Job_3", "Job_4"
        ]
    
    # Try to load MLflow model with timeout
    try:
        await asyncio.wait_for(_load_mlflow_model(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("MLflow model loading timed out after 30 seconds")
        model_store.model = None
        model_store.version = "fallback"
    except Exception as e:
        logger.error(f"Failed to load MLflow model: {e}")
        model_store.model = None
        model_store.version = "fallback"

async def _load_mlflow_model():
    """Internal function to load MLflow model"""
    import asyncio
    
    def _sync_load():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Try to get model from Production stage
        try:
            latest_version = client.get_latest_versions(MODEL_NAME, stages=[DEFAULT_MODEL_STAGE])[0]
            model_store.version = latest_version.version
            logger.info(f"Found model version {model_store.version} in {DEFAULT_MODEL_STAGE} stage")
        except Exception as e:
            logger.warning(f"No model found in {DEFAULT_MODEL_STAGE} stage: {e}")
            # Fallback to any available version
            all_versions = client.get_latest_versions(MODEL_NAME)
            if all_versions:
                latest_version = max(all_versions, key=lambda x: int(x.version))
                model_store.version = latest_version.version
                logger.info(f"Using latest available version: {model_store.version}")
            else:
                raise Exception("No model versions found")

        # Load the model
        model_uri = f"models:/{MODEL_NAME}/{model_store.version}"
        model_store.model = mlflow.xgboost.load_model(model_uri)
        
        # Try to load explainer
        try:
            explainer_path = client.download_artifacts(latest_version.run_id, "explainer.joblib")
            model_store.explainer = joblib.load(explainer_path)
            logger.info("SHAP explainer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load explainer: {e}")
            model_store.explainer = None

        logger.info(f"Successfully loaded model '{MODEL_NAME}' version '{model_store.version}'")
    
    # Run the sync function in a thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_load)

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
    """Health check that doesn't depend on model being loaded"""
    health_status = {
        "status": "ok",
        "model_loaded": model_store.model is not None,
        "model_version": model_store.version or "not_loaded",
        "feature_names_count": len(model_store.feature_names) if model_store.feature_names else 0
    }
    
    # Always return 200 OK so the pod can start
    # The model loading happens in background
    return health_status

@app.post("/predict/production", response_model=PredictionResponse)
async def predict_production(request: Request, data: CreditApp):
    """Predict using the production model"""
    return await predict_internal(request, data)

@app.post("/predict/{model_version}", response_model=PredictionResponse)
async def predict(request: Request, data: CreditApp, model_version: str = DEFAULT_MODEL_STAGE):
    """Predict using a specific model version"""
    return await predict_internal(request, data)

async def predict_internal(request: Request, data: CreditApp):
    if model_store.model is None:
        # Return a simple rule-based fallback prediction
        logger.warning("Using fallback prediction - no MLflow model available")
        
        # Simple rule-based logic as fallback
        debt_to_income = data.Debt / max(data.Income, 1)
        loan_to_income = data.Amount / max(data.Income, 1)
        
        # Basic risk assessment
        risk_factors = 0
        if debt_to_income > 0.4: risk_factors += 1
        if loan_to_income > 3.0: risk_factors += 1
        if data.Age < 25: risk_factors += 1
        if data.Records != "no_rec": risk_factors += 1
        if data.Income < 50: risk_factors += 1
        
        # Simple prediction logic
        probability_good_credit = max(0.1, 0.9 - (risk_factors * 0.15))
        prediction = 1 if probability_good_credit >= 0.5 else 0
        
        proba_default = 1 - probability_good_credit
        risk_score = proba_default * 1000
        risk_tier = get_risk_tier(probability_good_credit)
        expected_loss = proba_default * LOSS_GIVEN_DEFAULT * data.Amount
        
        return {
            "request_id": request.state.request_id, 
            "model_version": "fallback-rule-based",
            "prediction": prediction, 
            "probability_good_credit": probability_good_credit,
            "risk_score": risk_score, 
            "risk_tier": risk_tier, 
            "expected_loss": expected_loss
        }
    
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
            "request_id": request.state.request_id, 
            "model_version": model_store.version,
            "prediction": prediction, 
            "probability_good_credit": proba_good_credit,
            "risk_score": risk_score, 
            "risk_tier": risk_tier, 
            "expected_loss": expected_loss
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/production", response_model=ExplanationResponse)
async def explain_production(request: Request, data: CreditApp):
    """Explain prediction using the production model"""
    return await explain_internal(request, data)

@app.post("/explain/{model_version}", response_model=ExplanationResponse)
async def explain(request: Request, data: CreditApp, model_version: str = DEFAULT_MODEL_STAGE):
    """Explain prediction using a specific model version"""
    return await explain_internal(request, data)

async def explain_internal(request: Request, data: CreditApp):
    if model_store.explainer is None:
        # If no explainer available, just return prediction without explanations
        prediction_data = await predict_internal(request, data)
        return {**prediction_data, "top_features": []}
    
    prediction_data = await predict_internal(request, data)
    
    try:
        input_df = pd.DataFrame([data.dict()])
        processed_df = feature_engineer(input_df)
        final_df = processed_df.reindex(columns=model_store.feature_names, fill_value=0)
        
        shap_values = model_store.explainer.shap_values(final_df)
        contributions = [
            {"feature": f, "contribution": round(v, 4)} 
            for f, v in zip(final_df.columns, shap_values[0])
        ]
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {**prediction_data, "top_features": contributions[:5]}
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        # Return prediction without explanations if SHAP fails
        return {**prediction_data, "top_features": []}
