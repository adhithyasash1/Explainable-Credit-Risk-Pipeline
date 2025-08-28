import os
import uuid
import json
import joblib
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger

from app.utils import feature_engineering_api, get_risk_tier, format_shap_explanation
from prometheus_fastapi_instrumentator import Instrumentator

# --- 1. Logging Setup ---
# Setup structured JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logging.basicConfig(handlers=[logHandler], level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Application State ---
# A simple dictionary to hold our model artifacts.
# This avoids using global variables and is managed by the lifespan context manager.
model_store = {}

# --- 3. Lifespan Context Manager (replaces on_event("startup")) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model artifacts
    logger.info("Application startup: Loading model artifacts...")
    try:
        model_store["model"] = joblib.load("app/model.joblib")
        model_store["explainer"] = joblib.load("app/explainer.joblib")
        model_store["feature_names"] = joblib.load("app/feature_names.joblib")
        with open("app/global_feature_importance.json", 'r') as f:
            model_store["global_importance"] = json.load(f)
        model_store["model_version"] = os.getenv("MODEL_VERSION", "local")
        logger.info(f"Successfully loaded model version: {model_store['model_version']}")
    except FileNotFoundError:
        logger.error("FATAL: Model artifacts not found. The API cannot serve predictions.")
        # In a real scenario, you might have a fallback or a more robust health check failure
        model_store["model"] = None 
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}")
        model_store["model"] = None
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Application shutdown.")
    model_store.clear()

# --- 4. FastAPI App Initialization ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="A production-ready API for credit scoring with SHAP explainability and Prometheus monitoring.",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument the app with Prometheus metrics
Instrumentator().instrument(app).expose(app)

# --- 5. Middleware ---
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    # Assign a unique ID to each request for traceability
    request.state.request_id = str(uuid.uuid4())
    logger.info(f"Request started", extra={"request_id": request.state.request_id, "path": request.url.path})
    
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info(f"Request finished", extra={"request_id": request.state.request_id})
    return response

# --- 6. Custom Exception Handler ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error(f"An unhandled exception occurred: {exc}", exc_info=True, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"request_id": request_id, "error": "Internal Server Error", "detail": "An unexpected error occurred."},
    )

# --- 7. Pydantic Schemas for API I/O ---
class CreditApplication(BaseModel):
    Seniority: int = Field(..., example=10)
    Home: str = Field(..., example="owner")
    Time: int = Field(..., example=36)
    Age: int = Field(..., example=45)
    Marital: str = Field(..., example="married")
    Records: str = Field(..., example="no_rec")
    Job: str = Field(..., example="fixed")
    Expenses: int = Field(..., example=75)
    Income: float = Field(..., example=250.0)
    Assets: float = Field(..., example=10000.0)
    Debt: float = Field(..., example=2000.0)
    Amount: int = Field(..., example=1500)
    Price: int = Field(..., example=2000)

class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    prediction: str # 'Approve' or 'Deny'
    probability_good_credit: float
    risk_tier: str
    expected_loss: float

class Explanation(BaseModel):
    feature: str
    explanation: str

class ExplanationResponse(PredictionResponse):
    explanation: list[Explanation]

# --- 8. API Endpoints ---
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Performs a health check of the API."""
    if model_store.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service is unavailable.")
    return {"status": "ok", "model_version": model_store.get("model_version")}

@app.get("/global-importance", tags=["Explainability"])
def get_global_importance():
    """Returns the top features influencing the model globally."""
    if not model_store.get("global_importance"):
        raise HTTPException(status_code=503, detail="Global importance data not available.")
    return model_store["global_importance"]

def process_and_predict(data: CreditApplication, request: Request):
    """Core logic for preprocessing and prediction."""
    if not model_store.get("model"):
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot process request.")

    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Apply feature engineering
    engineered_df = feature_engineering_api(input_df)
    
    # Align columns with training data
    final_df = engineered_df.reindex(columns=model_store["feature_names"], fill_value=0)
    
    # Make prediction
    probability_good = model_store["model"].predict_proba(final_df)[0][1]
    
    # Calculate business metrics
    loss_given_default = 0.9  # Assumed LGD
    probability_default = 1 - probability_good
    expected_loss = probability_default * loss_given_default * data.Amount
    
    response_data = {
        "request_id": request.state.request_id,
        "model_version": model_store["model_version"],
        "prediction": "Approve" if probability_good >= 0.5 else "Deny",
        "probability_good_credit": probability_good,
        "risk_tier": get_risk_tier(probability_good),
        "expected_loss": expected_loss,
    }
    return response_data, final_df

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: CreditApplication, request: Request):
    """
    Predicts credit risk for a single application.
    Returns a prediction, probability, risk tier, and expected loss.
    """
    response, _ = process_and_predict(data, request)
    return response

@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain(data: CreditApplication, request: Request):
    """

    Provides a detailed explanation for a credit risk prediction.
    Includes the prediction and the top 5 features driving the outcome.
    """
    response, final_df = process_and_predict(data, request)
    
    # Generate SHAP explanation
    shap_values = model_store["explainer"].shap_values(final_df)
    formatted_explanation = format_shap_explanation(shap_values, model_store["feature_names"], final_df)
    
    response["explanation"] = formatted_explanation
    return response
