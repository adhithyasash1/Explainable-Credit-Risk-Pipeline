import requests
import json
from datetime import datetime

# --- Configuration ---
# Ensure your FastAPI application is running and accessible at this address.
BASE_URL = "http://localhost:8000"
LOG_FILE = "endpoint_outputs.log"

# A sample payload to send to the prediction endpoints.
SAMPLE_APPLICATION = {
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

def log_transaction(file_handle, endpoint, method, response):
    """A helper function to format and write the log entry."""
    timestamp = datetime.now().isoformat()
    
    log_header = f"--- [ {timestamp} ] ---"
    log_request_line = f"REQUEST: {method} {endpoint}"
    log_status_line = f"STATUS: {response.status_code}"
    
    file_handle.write(f"{log_header}\n{log_request_line}\n{log_status_line}\n")
    
    # Try to format the response as pretty JSON, otherwise write as plain text.
    try:
        pretty_json = json.dumps(response.json(), indent=2)
        file_handle.write("RESPONSE BODY:\n")
        file_handle.write(pretty_json)
    except json.JSONDecodeError:
        file_handle.write("RESPONSE BODY (non-JSON):\n")
        file_handle.write(response.text)
        
    file_handle.write("\n\n")
    print(f"✅ Logged response from {endpoint}")

def main():
    """Sends requests to all API endpoints and logs their outputs to a file."""
    print(f"🚀 Starting API requests to {BASE_URL}...")
    print(f"📝 Logging outputs to '{LOG_FILE}'")
    
    # Using 'w' mode to create a fresh log file each time the script is run.
    # Change to 'a' if you want to append to the file instead.
    with open(LOG_FILE, 'w') as f:
        try:
            # 1. Hit the /health endpoint
            print("Pinging /health endpoint...")
            health_response = requests.get(f"{BASE_URL}/health", timeout=10)
            log_transaction(f, "/health", "GET", health_response)

            # 2. Hit the /predict endpoint
            print("Pinging /predict endpoint...")
            predict_response = requests.post(f"{BASE_URL}/predict", json=SAMPLE_APPLICATION, timeout=10)
            log_transaction(f, "/predict", "POST", predict_response)

            # 3. Hit the /explain endpoint
            print("Pinging /explain endpoint...")
            explain_response = requests.post(f"{BASE_URL}/explain", json=SAMPLE_APPLICATION, timeout=10)
            log_transaction(f, "/explain", "POST", explain_response)

        except requests.exceptions.ConnectionError:
            error_msg = f"❌ FAILED: Could not connect to the API at {BASE_URL}.\nIs the server running?"
            print(error_msg)
            f.write(error_msg)
        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {e}"
            print(error_msg)
            f.write(error_msg)

    print(f"🎉 Done. All outputs have been logged to '{LOG_FILE}'.")

if __name__ == "__main__":
    main()
