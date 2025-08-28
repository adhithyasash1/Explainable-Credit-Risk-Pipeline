import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def feature_engineering_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering steps used in training for API requests.
    This function is critical for maintaining consistency between training and inference.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

def get_risk_tier(probability_good: float) -> str:
    """Classifies a probability score into a business-friendly risk tier."""
    if probability_good > 0.85:
        return "Tier 1: Very Low Risk"
    elif probability_good > 0.70:
        return "Tier 2: Low Risk"
    elif probability_good > 0.50:
        return "Tier 3: Medium Risk"
    elif probability_good > 0.35:
        return "Tier 4: High Risk"
    else:
        return "Tier 5: Very High Risk"

def generate_business_explanation(feature: str, value: float, shap_value: float) -> str:
    """Creates a human-readable explanation from a SHAP value."""
    direction = "increased" if shap_value > 0 else "decreased"
    impact_level = "significantly" if abs(shap_value) > 0.1 else "moderately"
    
    # Custom narratives for key features
    if 'Debt-to-Income' in feature:
        return f"The applicant's Debt-to-Income ratio of {value:.2f} {impact_level} {direction} the predicted risk of default."
    if 'Income' in feature:
        return f"An income of ${value:,.0f} {impact_level} {direction} the predicted risk of default."
    if 'Age' in feature:
        return f"The applicant's age of {int(value)} {impact_level} {direction} the predicted risk."
    if 'Records_no_rec_False' in feature: # This corresponds to Records=yes
        return f"Having previous credit records {impact_level} {direction} the predicted risk."
        
    return f"The feature '{feature}' with value '{value}' {impact_level} {direction} the risk of default."

def format_shap_explanation(shap_values, feature_names: list, input_data: pd.DataFrame) -> list:
    """Formats SHAP values into a structured, business-friendly list."""
    # SHAP values for the positive class (good credit)
    shap_values_good_credit = shap_values
    
    contributions = []
    for feature, shap_val in zip(feature_names, shap_values_good_credit[0]):
        # A positive SHAP value for the 'good credit' class means it pushes the prediction towards 1 (good).
        # To explain risk (default), we invert the logic.
        risk_contribution = -shap_val 
        contributions.append({
            'feature': feature,
            'value': input_data[feature].iloc[0] if feature in input_data.columns else 'N/A',
            'risk_contribution': risk_contribution
        })

    # Sort by the absolute magnitude of risk contribution
    contributions.sort(key=lambda x: abs(x['risk_contribution']), reverse=True)

    explained_contributions = []
    for item in contributions[:5]: # Top 5 contributors
        explanation = generate_business_explanation(item['feature'], item['value'], item['risk_contribution'])
        explained_contributions.append({
            "feature": item['feature'],
            "explanation": explanation
        })
        
    return explained_contributions
