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
# Force local MLflow tracking
MLFLOW_TRACKING_URI = "file:./mlruns"  # Always use local storage
DATA_PATH = "data/CreditScoring.csv"
ARTIFACT_PATH = "app"  # Local path for temporary artifacts
MODEL_NAME = "credit-risk-xgb"

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

def train():
    print("🚀 Starting training process with LOCAL MLflow...")
    print(f"📁 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set local tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    try:
        experiment = mlflow.set_experiment("Credit Risk Prediction")
        print(f"✅ Using experiment: {experiment.name} (ID: {experiment.experiment_id})")
    except Exception as e:
        print(f"⚠️ Warning creating experiment: {e}")
        print("Creating new experiment...")
        mlflow.create_experiment("Credit Risk Prediction")
        experiment = mlflow.set_experiment("Credit Risk Prediction")

    with mlflow.start_run() as run:
        print(f"📊 MLflow Run ID: {run.info.run_id}")
        
        # Load and prepare data
        print("Loading data...")
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.capitalize()
        df['Status'] = df['Status'].apply(lambda x: 1 if x == 1 else 0)
        
        X = df.drop('Status', axis=1)
        y = df['Status']
        
        # Feature engineering
        print("Applying feature engineering...")
        X_engineered = feature_engineering(X.copy())
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Align columns
        train_cols = X_train.columns
        test_cols = X_test.columns
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        X_test = X_test[train_cols]
        feature_names = X_train.columns.tolist()
        
        # Model training
        print("Training XGBoost model...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'early_stopping_rounds': 20
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Evaluation
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        ks = calculate_ks_statistic(y_test, y_proba)
        
        # Convert numpy float to Python float for proper JSON serialization
        metrics = {
            "AUC": float(auc),
            "KS_Statistic": float(ks)
        }
        
        print(f"📈 Model Performance:")
        print(f"   - AUC: {metrics['AUC']:.4f}")
        print(f"   - KS Statistic: {metrics['KS_Statistic']:.4f}")
        
        mlflow.log_metrics(metrics)
        
        # Save artifacts locally first
        print("Generating artifacts...")
        os.makedirs(ARTIFACT_PATH, exist_ok=True)
        
        # Save model
        model_path = os.path.join(ARTIFACT_PATH, "model.joblib")
        joblib.dump(model, model_path)
        
        # Create and save explainer
        explainer = shap.TreeExplainer(model)
        explainer_path = os.path.join(ARTIFACT_PATH, "explainer.joblib")
        joblib.dump(explainer, explainer_path)
        
        # Save feature names
        feature_names_path = os.path.join(ARTIFACT_PATH, "feature_names.joblib")
        joblib.dump(feature_names, feature_names_path)
        
        # Calculate and save global feature importance
        print("Calculating global feature importance...")
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(
            zip(feature_names, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True
        )
        
        global_importance = [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in feature_importance[:10]
        ]
        
        global_importance_path = os.path.join(ARTIFACT_PATH, "global_feature_importance.json")
        with open(global_importance_path, 'w') as f:
            json.dump(global_importance, f, indent=2)
        
        # Log to MLflow
        print("Logging artifacts to MLflow...")
        
        # Log the model with MLflow
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.iloc[:5]
        )
        
        # Log additional artifacts
        mlflow.log_artifact(explainer_path, artifact_path="model")
        mlflow.log_artifact(feature_names_path, artifact_path="model")
        mlflow.log_artifact(global_importance_path, artifact_path="model")
        
        print(f"\n✅ Training complete!")
        print(f"📦 Model artifacts saved to: {ARTIFACT_PATH}/")
        print(f"🔍 MLflow Run ID: {run.info.run_id}")
        print(f"📊 Model registered as: {MODEL_NAME}")

if __name__ == "__main__":
    train()
