import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

print("🚀 Starting model training and artifact creation process...")

# --- 1. Load and Process Data ---
df = pd.read_csv("data/CreditScoring.csv")

# --- FIX IS HERE: The target column is 'Status' ---
# First, correctly preprocess the 'Status' column
df['Status'] = df['Status'].replace({1: 1, 2: 0})
df = df.dropna(subset=['Status'])
df['Status'] = df['Status'].astype(int)

# Now, preprocess the feature columns
features = df.drop('Status', axis=1)
categorical_cols = features.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# --- 2. Feature Engineering ---
df['Income'] = df['Income'].replace(0, 1)
df['Price'] = df['Price'].replace(0, 1)
df['Debt-to-Income'] = df['Debt'] / df['Income']
df['Loan-to-Price'] = df['Amount'] / df['Price']
df['Loan-to-Income'] = df['Amount'] / df['Income']

# The dummify step was causing issues with factorized columns, this is a safer order
cat_cols_for_dummies = ['Home', 'Marital', 'Records', 'Job']
# Ensure these columns exist before trying to dummify them
existing_cat_cols = [col for col in cat_cols_for_dummies if col in df.columns]
df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

print("✅ Data preprocessing and feature engineering complete.")

# --- 3. Model Training ---
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
xgb_clf = xgb.XGBClassifier(
    random_state=42, objective="binary:logistic", eval_metric="auc",
    scale_pos_weight=scale_pos_weight, use_label_encoder=False
)
xgb_clf.fit(X_train, y_train)
print("✅ XGBoost model training complete.")

# --- 4. Create SHAP Explainer ---
explainer = shap.TreeExplainer(xgb_clf)
print("✅ SHAP TreeExplainer created.")

# --- 5. Save All Artifacts to 'app/' Directory ---
joblib.dump(xgb_clf, "app/model.joblib")
joblib.dump(explainer, "app/explainer.joblib")
joblib.dump(X_train.columns.tolist(), "app/feature_names.joblib")

print("📦 All artifacts (model, explainer, features) saved successfully to 'app/'.")
