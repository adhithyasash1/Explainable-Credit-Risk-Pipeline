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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
DATA_PATH = "data/CreditScoring.csv"
ARTIFACT_PATH = "app" # Save artifacts directly to the app folder for Docker build
MODEL_NAME = "credit-risk-xgb"

# --- Helper Functions ---
def calculate_ks_statistic(y_true, y_proba):
    """Calculate the Kolmogorov-Smirnov (KS) statistic."""
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df_good = df[df['y_true'] == 1]
    df_bad = df[df['y_true'] == 0]

    # Calculate cumulative percentages
    df_good_sorted = df_good.sort_values('y_proba')
    df_bad_sorted = df_bad.sort_values('y_proba')

    good_cum = np.arange(1, len(df_good_sorted) + 1) / len(df_good_sorted)
    bad_cum = np.arange(1, len(df_bad_sorted) + 1) / len(df_bad_sorted)

    # To compare at the same probability points, we need a common set of thresholds
    all_thresholds = np.unique(y_proba)
    ks_values = []
    for threshold in all_thresholds:
        good_pct = np.sum(df_good['y_proba'] >= threshold) / len(df_good)
        bad_pct = np.sum(df_bad['y_proba'] >= threshold) / len(df_bad)
        ks_values.append(abs(good_pct - bad_pct))

    return max(ks_values) if ks_values else 0

def feature_engineering(df):
    """Applies all feature engineering steps."""
    # Fill NaNs with 0 (assuming missing means absence of the attribute)
    df.fillna(0, inplace=True)

    # Replace infinite values that might arise from division by zero
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # 1. Financial Ratios
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    df['Debt-to-Income'] = df['Debt'] / (df['Income'] + epsilon)
    df['Loan-to-Price'] = df['Amount'] / (df['Price'] + epsilon)
    df['Loan-to-Income'] = df['Amount'] / (df['Income'] + epsilon)

    # 2. Binning/Grouping numerical features
    df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=['Young_Adult', 'Adult', 'Middle_Aged', 'Senior'])
    df['Income_Bracket'] = pd.cut(df['Income'], bins=[0, 200, 500, 1000, 10000], labels=['Low', 'Medium', 'High', 'Very_High'], right=False)

    # 3. One-Hot Encoding for categorical features
    categorical_cols = ['Home', 'Marital', 'Records', 'Job', 'Age_Group', 'Income_Bracket']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    return df

# --- Main Training Logic ---
def train():
    """Main function to train and log the credit risk model."""
    print("🚀 Starting training process...")

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit Risk Prediction")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.set_tag("mlflow.runName", "XGBoost Training Run")

        # 1. Load Data
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.capitalize()

        # 2. Preprocessing & Feature Engineering
        print("Performing data cleaning and feature engineering...")
        # Target variable: 1 for 'good', 0 for 'bad'
        df['Status'] = df['Status'].apply(lambda x: 1 if x == 1 else 0)
        X = df.drop('Status', axis=1)
        y = df['Status']

        X_engineered = feature_engineering(X.copy())

        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.25, random_state=42, stratify=y
        )

        # Align columns after splitting to handle missing columns from one-hot encoding
        train_cols = X_train.columns
        test_cols = X_test.columns
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        X_test = X_test[train_cols] # Ensure order is the same

        feature_names = X_train.columns.tolist()
        mlflow.log_param("feature_count", len(feature_names))

        # 4. Model Training
        print("Training XGBoost model...")
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # UPDATED: Moved early_stopping_rounds into the main parameter dictionary
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
            'early_stopping_rounds': 20 # MOVED HERE
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        
        # UPDATED: Removed early_stopping_rounds from the .fit() call
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # 5. Evaluation
        print("Evaluating model and calculating business metrics...")
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, (y_proba > 0.5).astype(int))
        recall = recall_score(y_test, (y_proba > 0.5).astype(int))
        ks = calculate_ks_statistic(y_test, y_proba)
        gini = 2 * auc - 1

        metrics = {"AUC": auc, "Precision": precision, "Recall": recall, "KS_Statistic": ks, "Gini": gini}
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        # 6. SHAP Explainability
        print("Generating SHAP explainer...")
        explainer = shap.TreeExplainer(model)

        # Global feature importance
        shap_values = explainer.shap_values(X_test)
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        global_importance = importance_df.head(15).to_dict(orient='records')

        # 7. Save Artifacts
        print(f"Saving artifacts to '{ARTIFACT_PATH}' directory...")
        os.makedirs(ARTIFACT_PATH, exist_ok=True)
        joblib.dump(model, os.path.join(ARTIFACT_PATH, "model.joblib"))
        joblib.dump(explainer, os.path.join(ARTIFACT_PATH, "explainer.joblib"))
        joblib.dump(feature_names, os.path.join(ARTIFACT_PATH, "feature_names.joblib"))
        with open(os.path.join(ARTIFACT_PATH, "global_feature_importance.json"), 'w') as f:
            json.dump(global_importance, f, indent=4)

        # 8. Log artifacts to MLflow
        print("Logging artifacts to MLflow...")
        mlflow.log_artifact(os.path.join(ARTIFACT_PATH, "model.joblib"))
        mlflow.log_artifact(os.path.join(ARTIFACT_PATH, "explainer.joblib"))
        mlflow.log_artifact(os.path.join(ARTIFACT_PATH, "feature_names.joblib"))
        mlflow.log_artifact(os.path.join(ARTIFACT_PATH, "global_feature_importance.json"))

        # Log and register model in MLflow Model Registry
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgb-model",
            registered_model_name=MODEL_NAME
        )

        print("\n✅ Training complete! Model and artifacts are ready.")

if __name__ == "__main__":
    train()
