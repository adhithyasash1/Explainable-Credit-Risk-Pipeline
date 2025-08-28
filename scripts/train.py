import os
import json
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
DATA_PATH = "data/CreditScoring.csv"
ARTIFACT_PATH = "app" # Local path for temporary artifacts
MODEL_NAME = "credit-risk-xgb"

# --- Helper Functions (omitted for brevity, they are unchanged) ---
def calculate_ks_statistic(y_true, y_proba):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df_good = df[df['y_true'] == 1]
    df_bad = df[df['y_true'] == 0]
    all_thresholds = np.unique(y_proba)
    ks_values = []
    for threshold in all_thresholds:
        good_pct = np.sum(df_good['y_proba'] >= threshold) / len(df_good)
        bad_pct = np.sum(df_bad['y_proba'] >= threshold) / len(df_bad)
        ks_values.append(abs(good_pct - bad_pct))
    return max(ks_values) if ks_values else 0

def feature_engineering(df):
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    epsilon = 1e-6
    df['Debt-to-Income'] = df['Debt'] / (df['Income'] + epsilon)
    df['Loan-to-Price'] = df['Amount'] / (df['Price'] + epsilon)
    df['Loan-to-Income'] = df['Amount'] / (df['Income'] + epsilon)
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=['Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
    df['Income_Bracket'] = pd.cut(df['Income'], bins=[0, 200, 500, 1000, 10000], labels=['Low', 'Medium', 'High', 'Very_High'], right=False)
    categorical_cols = ['Home', 'Marital', 'Records', 'Job', 'Age_Group', 'Income_Bracket']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    return df

# --- Main Training Logic ---
def train():
    print("🚀 Starting training process...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit Risk Prediction")

    with mlflow.start_run() as run:
        # ... (Data loading, feature engineering, train/test split are unchanged) ...
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.capitalize()
        df['Status'] = df['Status'].apply(lambda x: 1 if x == 1 else 0)
        X = df.drop('Status', axis=1)
        y = df['Status']
        X_engineered = feature_engineering(X.copy())
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.25, random_state=42, stratify=y
        )
        train_cols = X_train.columns
        test_cols = X_test.columns
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        X_test = X_test[train_cols]
        feature_names = X_train.columns.tolist()

        # ... (Model Training is unchanged) ...
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 200,
            'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': 42, 'scale_pos_weight': scale_pos_weight, 'early_stopping_rounds': 20
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ... (Evaluation is unchanged) ...
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        ks = calculate_ks_statistic(y_test, y_proba)
        metrics = {"AUC": auc, "KS_Statistic": ks}
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        # --- ARTIFACTS AND LOGGING (CORRECTED) ---
        print("Generating and logging artifacts...")
        os.makedirs(ARTIFACT_PATH, exist_ok=True)
        explainer = shap.TreeExplainer(model)
        explainer_path = os.path.join(ARTIFACT_PATH, "explainer.joblib")
        joblib.dump(explainer, explainer_path)
        
        feature_names_path = os.path.join(ARTIFACT_PATH, "feature_names.joblib")
        joblib.dump(feature_names, feature_names_path)
        
        # Log the XGBoost model with an input example to create a signature
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model", # Use 'name' instead of the deprecated 'artifact_path'
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:5] # Add input example
        )
        
        # Now, log the other artifacts to the SAME directory.
        mlflow.log_artifact(explainer_path, artifact_path="model")
        mlflow.log_artifact(feature_names_path, artifact_path="model")
        
        print("\n✅ Training complete! Model and artifacts logged to MLflow.")

if __name__ == "__main__":
    train()
