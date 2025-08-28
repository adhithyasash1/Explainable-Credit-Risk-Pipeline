import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- MLflow Configuration ---
# Set this to your MLflow tracking server URI. Can be a local folder or a remote server.
# For this example, we'll use a local directory named 'mlruns'.
MLFLOW_TRACKING_URI = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("credit_risk_training")

print(f"✅ MLflow configured to track to: {MLFLOW_TRACKING_URI}")

def train_model():
    with mlflow.start_run() as run:
        print(f"🚀 Starting MLflow Run: {run.info.run_id}")

        # --- 1. Load and Process Data ---
        df = pd.read_csv("data/CreditScoring.csv")
        
        # Log data version (e.g., its hash)
        mlflow.log_param("data_hash", joblib.hash(df))

        df['Status'] = df['Status'].replace({1: 1, 2: 0})
        df = df.dropna(subset=['Status'])
        df['Status'] = df['Status'].astype(int)

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]
        
        # --- 2. Feature Engineering ---
        df['Income'] = df['Income'].replace(0, 1)
        df['Price'] = df['Price'].replace(0, 1)
        df['Debt-to-Income'] = df['Debt'] / df['Income']
        df['Loan-to-Price'] = df['Amount'] / df['Price']
        df['Loan-to-Income'] = df['Amount'] / df['Income']
        df = pd.get_dummies(df, columns=['Home', 'Marital', 'Records', 'Job'], drop_first=True)
        print("✅ Data preprocessing and feature engineering complete.")

        # --- 3. Model Training ---
        X = df.drop('Status', axis=1)
        y = df['Status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        
        params = {
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "use_label_encoder": False
        }
        mlflow.log_params(params)
        
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(X_train, y_train)
        print("✅ XGBoost model training complete.")

        # --- 4. Evaluation ---
        y_proba = xgb_clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", auc_score)
        print(f"📈 Test Set AUC: {auc_score:.4f}")

        # --- 5. Create and Log SHAP Explainer ---
        explainer = shap.TreeExplainer(xgb_clf)
        
        # Log feature names as a text artifact
        feature_names_path = "feature_names.txt"
        with open(feature_names_path, "w") as f:
            for item in X_train.columns.tolist():
                f.write("%s\n" % item)
        mlflow.log_artifact(feature_names_path)

        # --- 6. Log Model to MLflow Registry ---
        print("📦 Logging model to MLflow Registry...")
        mlflow.xgboost.log_model(
            xgb_model=xgb_clf,
            artifact_path="model",
            registered_model_name="credit-risk-xgb" # This name controls the versioning
        )
        
        # Log the SHAP explainer as a separate artifact
        explainer_path = "explainer.joblib"
        joblib.dump(explainer, explainer_path)
        mlflow.log_artifact(explainer_path)

        print("🎉 Training run complete and artifacts logged to MLflow.")

if __name__ == "__main__":
    train_model()
