import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "credit-risk-xgb"
MODEL_STAGE = "Production"
DESTINATION_PATH = "app"

def download_production_model():
    """
    Downloads the latest model version from the 'Production' stage
    using the model URI.
    """
    if not MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set.")

    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Define the model URI to download from
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Downloading artifacts from model URI: '{model_uri}' to '{DESTINATION_PATH}'...")
        
        os.makedirs(DESTINATION_PATH, exist_ok=True)
        
        # This function downloads all artifacts associated with the registered model version
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=DESTINATION_PATH
        )

        print("✅ Successfully downloaded model artifacts:")
        # List all files in the destination, including subdirectories
        for root, _, files in os.walk(DESTINATION_PATH):
            for name in files:
                print(f"  - {os.path.join(root, name)}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        exit(1)

if __name__ == "__main__":
    download_production_model()
