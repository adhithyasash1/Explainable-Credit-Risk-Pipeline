#!/usr/bin/env python3
"""
Creates dummy model artifacts for CI/CD pipeline when real models are not available.
"""
import os
import joblib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

def create_dummy_artifacts():
    os.makedirs('app', exist_ok=True)
    
    # Create a proper dummy model with correct number of features
    X = np.random.rand(100, 27)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save all required artifacts
    joblib.dump(model, 'app/model.joblib')
    joblib.dump({}, 'app/explainer.joblib')
    
    # Generate feature names matching the expected count
    feature_names = [
        'Seniority', 'Time', 'Age', 'Expenses', 'Income', 'Assets', 'Debt', 
        'Amount', 'Price', 'Debt-to-Income', 'Loan-to-Price', 'Loan-to-Income',
        'Home_1', 'Home_2', 'Home_3', 'Home_4', 'Home_5', 'Home_6',
        'Marital_1', 'Marital_2', 'Marital_3', 'Marital_4', 'Marital_5',
        'Records_2', 'Job_1', 'Job_2', 'Job_3', 'Job_4'
    ]
    joblib.dump(feature_names, 'app/feature_names.joblib')
    
    # Create global importance file
    importance = [{'feature': f, 'importance': float(np.random.rand())} 
                  for f in feature_names[:10]]
    with open('app/global_feature_importance.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    print('✅ Created dummy model artifacts for CI/CD')
    return True

if __name__ == "__main__":
    create_dummy_artifacts()
