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
    in the MLflow Model Registry.
    """
    if not MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set.")

    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Get the latest version in the 'Production' stage
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not latest_versions:
            raise ValueError(f"No model named '{MODEL_NAME}' found in stage '{MODEL_STAGE}'.")
        
        model_version = latest_versions[0]
        run_id = model_version.run_id
        print(f"Found model version {model_version.version} from run_id: {run_id}")

        # Download all artifacts from the model's run
        print(f"Downloading artifacts to '{DESTINATION_PATH}'...")
        os.makedirs(DESTINATION_PATH, exist_ok=True)
        client.download_artifacts(run_id, ".", dst_path=DESTINATION_PATH)

        print("✅ Successfully downloaded model artifacts:")
        for f in os.listdir(DESTINATION_PATH):
            if os.path.isfile(os.path.join(DESTINATION_PATH, f)):
                print(f"  - {f}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        exit(1)

if __name__ == "__main__":
    download_production_model()
