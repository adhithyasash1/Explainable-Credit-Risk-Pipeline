import pandas as pd
import pytest
from app.utils import feature_engineering_api

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame([{
        'Seniority': 5, 'Home': 'rent', 'Time': 60, 'Age': 30, 'Marital': 'single',
        'Records': 'no_rec', 'Job': 'freelance', 'Expenses': 45, 'Income': 100.0,
        'Assets': 500.0, 'Debt': 100.0, 'Amount': 800, 'Price': 1000
    }])

def test_dti_ltp_lti_calculation(sample_data):
    """Tests the calculation of key financial ratios."""
    engineered_df = feature_engineering_api(sample_data)
    
    assert 'Debt-to-Income' in engineered_df.columns
    assert 'Loan-to-Price' in engineered_df.columns
    assert 'Loan-to-Income' in engineered_df.columns
    
    # Expected values: DTI=100/100=1, LTP=800/1000=0.8, LTI=800/100=8
    assert engineered_df['Debt-to-Income'].iloc[0] == pytest.approx(1.0)
    assert engineered_df['Loan-to-Price'].iloc[0] == pytest.approx(0.8)
    assert engineered_df['Loan-to-Income'].iloc[0] == pytest.approx(8.0)

def test_one_hot_encoding(sample_data):
    """Tests that one-hot encoding is applied correctly."""
    engineered_df = feature_engineering_api(sample_data)
    
    # Check that original columns are gone and new ones are present
    assert 'Home' not in engineered_df.columns
    assert 'Home_owner' in engineered_df.columns # Example of a generated column
    assert 'Marital_married' in engineered_df.columns
    
    # Check the values are correct (drop_first=True)
    assert engineered_df['Marital_single'].iloc[0] == 1
    assert engineered_df['Marital_married'].iloc[0] == 0
