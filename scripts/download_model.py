import os
import mlflow
from mlflow.tracking import MlflowClient
import shutil

# --- Configuration ---
MLFLOW_TRACKING_URI = "file:./mlruns"  # Use local MLflow
MODEL_NAME = "credit-risk-xgb"
MODEL_STAGE = "Production"
DESTINATION_PATH = "app"

def download_latest_model():
    """
    Downloads the latest model version from local MLflow.
    If Production stage doesn't exist, uses the latest version.
    """
    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # First, let's see what models and versions we have
        print(f"\nChecking available models...")
        
        try:
            # Get all versions of the model
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            
            if not versions:
                print(f"❌ No versions found for model '{MODEL_NAME}'")
                print("Please train a model first using: python scripts/train.py")
                return
            
            print(f"Found {len(versions)} version(s) of '{MODEL_NAME}':")
            for v in versions:
                print(f"  - Version {v.version}: Stage={v.current_stage}, Run ID={v.run_id}")
            
            # Try to find a Production version
            production_version = None
            for v in versions:
                if v.current_stage == "Production":
                    production_version = v
                    break
            
            # If no Production version, use the latest version
            if production_version:
                version_to_use = production_version
                print(f"\n✓ Using Production version: {version_to_use.version}")
            else:
                # Get the latest version (highest version number)
                latest_version = max(versions, key=lambda x: int(x.version))
                version_to_use = latest_version
                print(f"\n⚠️ No Production version found. Using latest version: {version_to_use.version}")
            
            # Get the run to find artifacts
            run = client.get_run(version_to_use.run_id)
            artifact_uri = run.info.artifact_uri
            
            print(f"Artifact URI: {artifact_uri}")
            
            # Download artifacts directly from the run
            model_artifacts_path = os.path.join(artifact_uri.replace("file://", ""), "model")
            
            if not os.path.exists(model_artifacts_path):
                print(f"❌ Model artifacts not found at: {model_artifacts_path}")
                print("The model might not have been saved properly. Please retrain.")
                return
            
            # Create destination directory
            os.makedirs(DESTINATION_PATH, exist_ok=True)
            
            # Copy all files from model artifacts to app directory
            print(f"\nCopying artifacts from {model_artifacts_path} to {DESTINATION_PATH}...")
            
            for item in os.listdir(model_artifacts_path):
                src = os.path.join(model_artifacts_path, item)
                dst = os.path.join(DESTINATION_PATH, item)
                
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"  ✓ Copied: {item}")
                elif os.path.isdir(src) and item != "model":  # Skip nested model directories
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"  ✓ Copied directory: {item}")
            
            # Special handling for the model file if it's in a subdirectory
            model_subdir = os.path.join(model_artifacts_path, "model")
            if os.path.exists(model_subdir):
                # Copy model.pkl or model.xgb if it exists
                for model_file in ["model.pkl", "model.xgb", "model.joblib"]:
                    model_file_path = os.path.join(model_subdir, model_file)
                    if os.path.exists(model_file_path):
                        dst = os.path.join(DESTINATION_PATH, "model.joblib")
                        shutil.copy2(model_file_path, dst)
                        print(f"  ✓ Copied model file as: model.joblib")
                        break
            
            print(f"\n✅ Successfully downloaded model artifacts to '{DESTINATION_PATH}'")
            print("\nFiles in app directory:")
            for file in os.listdir(DESTINATION_PATH):
                file_path = os.path.join(DESTINATION_PATH, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else "[DIR]"
                print(f"  - {file}: {size} bytes" if isinstance(size, int) else f"  - {file}: {size}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\nTrying alternative approach...")
            
            # Alternative: Look for the most recent run with the model
            experiment = client.get_experiment_by_name("Credit Risk Prediction")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10
                )
                
                for run in runs:
                    artifact_path = os.path.join(run.info.artifact_uri.replace("file://", ""), "model")
                    if os.path.exists(artifact_path):
                        print(f"Found model artifacts in run: {run.info.run_id}")
                        
                        # Copy artifacts
                        os.makedirs(DESTINATION_PATH, exist_ok=True)
                        for item in os.listdir(artifact_path):
                            src = os.path.join(artifact_path, item)
                            dst = os.path.join(DESTINATION_PATH, item)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                                print(f"  ✓ Copied: {item}")
                        
                        print(f"\n✅ Successfully downloaded model artifacts")
                        return
                
                print("❌ No model artifacts found in recent runs")
            else:
                print("❌ Experiment 'Credit Risk Prediction' not found")
                
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_latest_model()
