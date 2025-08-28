import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Explainable Credit Risk API")

# Load artifacts from the flat directory structure
model = joblib.load("model.joblib")
explainer = joblib.load("explainer.joblib")
feature_names = joblib.load("feature_names.joblib")

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

class FeatureContribution(BaseModel):
    feature: str
    contribution: float

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 for Good Credit, 0 for Bad Credit")
    probability_good_credit: float

class ExplanationResponse(PredictionResponse):
    top_features: List[FeatureContribution]

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['Income'] = df['Income'].replace(0, 1)
    df['Price'] = df['Price'].replace(0, 1)
    df['Debt-to-Income'] = df['Debt'] / df['Income']
    df['Loan-to-Price'] = df['Amount'] / df['Price']
    df['Loan-to-Income'] = df['Amount'] / df['Income']
    df = pd.get_dummies(df, columns=['Home', 'Marital', 'Records', 'Job'], drop_first=True)
    return df

@app.get("/health")
def health_check(): return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CreditApp):
    try:
        input_df = pd.DataFrame([data.dict()])
        processed_df = feature_engineer(input_df)
        final_df = processed_df.reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        return {"prediction": int(prediction), "probability_good_credit": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain", response_model=ExplanationResponse)
def explain(data: CreditApp):
    try:
        input_df = pd.DataFrame([data.dict()])
        processed_df = feature_engineer(input_df)
        final_df = processed_df.reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(final_df)[0]
        probability = model.predict_proba(final_df)[0][1]
        shap_values = explainer.shap_values(final_df)
        contributions = [{"feature": f, "contribution": round(v, 4)} for f, v in zip(final_df.columns, shap_values[0])]
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return {"prediction": int(prediction), "probability_good_credit": float(probability), "top_features": contributions[:5]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
