import pytest
from fastapi.testclient import TestClient
from app.main import app

# This creates a client that can be used to make requests to the FastAPI app
client = TestClient(app)

@pytest.fixture
def valid_payload():
    """Provides a valid payload for prediction and explanation requests."""
    return {
      "Seniority": 10, "Home": "owner", "Time": 36, "Age": 45, "Marital": "married",
      "Records": "no_rec", "Job": "fixed", "Expenses": 75, "Income": 250.0,
      "Assets": 10000.0, "Debt": 2000.0, "Amount": 1500, "Price": 2000
    }

def test_health_check():
    """Tests the /health endpoint for a 200 OK response."""
    # This test might fail if run before the model is loaded during startup.
    # In a real CI/CD, you would ensure the app is fully started.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint(valid_payload):
    """Tests the /predict endpoint with a valid payload."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "prediction" in ["Approve", "Deny"]
    assert "probability_good_credit" in data
    assert 0 <= data["probability_good_credit"] <= 1

def test_explain_endpoint(valid_payload):
    """Tests the /explain endpoint for a valid payload and explanation structure."""
    response = client.post("/explain", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    assert "feature" in data["explanation"][0]
    assert "explanation" in data["explanation"][0]

def test_predict_invalid_payload():
    """Tests the /predict endpoint with an invalid payload (missing field)."""
    invalid_payload = {
      "Seniority": 10, "Home": "owner" # Missing many required fields
    }
    response = client.post("/predict", json=invalid_payload)
    # FastAPI's Pydantic validation should return a 422 Unprocessable Entity
    assert response.status_code == 422
