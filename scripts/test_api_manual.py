import requests
import json

# API base URL (adjust if needed)
BASE_URL = "http://localhost:8000"

# Sample application data
sample_application = {
    "Seniority": 10,
    "Home": "owner",
    "Time": 36,
    "Age": 45,
    "Marital": "married",
    "Records": "no_rec",
    "Job": "fixed",
    "Expenses": 75,
    "Income": 250.0,
    "Assets": 10000.0,
    "Debt": 2000.0,
    "Amount": 1500,
    "Price": 2000
}

# Test health check
print("Testing Health Check...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test prediction
print("Testing Prediction...")
response = requests.post(f"{BASE_URL}/predict", json=sample_application)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test explanation
print("Testing Explanation...")
response = requests.post(f"{BASE_URL}/explain", json=sample_application)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of Good Credit: {result['probability_good_credit']:.2%}")
    print(f"Risk Tier: {result['risk_tier']}")
    print(f"Expected Loss: ${result['expected_loss']:.2f}")
    print("\nTop Feature Explanations:")
    for exp in result.get('explanation', [])[:3]:
        print(f"  - {exp['feature']}: {exp['explanation']}")

# Test global feature importance
print("\nTesting Global Feature Importance...")
response = requests.get(f"{BASE_URL}/global-importance")
print(f"Status: {response.status_code}")
if response.status_code == 200:
    importance = response.json()
    print("Top 5 Most Important Features:")
    # Assuming the response is a list of feature importance objects
    for i, feature in enumerate(importance[:5], 1):
        print(f"  {i}. {feature}")
