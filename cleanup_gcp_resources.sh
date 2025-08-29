#!/bin/bash
# This script deletes all the GCP resources created by the deploy_on_gcp.sh script.
# It is designed to prevent ongoing costs after you are done with the project.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🔥 Starting GCP Resource Cleanup Process..."

# --- Phase 1: Configure Environment Variables ---
# Ensure these variables match the ones used during deployment.

# Set your Google Cloud Project ID
export PROJECT_ID=$(gcloud config get-value project)

# Configure GCP settings
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="credit-risk-cluster"

# Configure application-specific names
export REPO="credit-risk-repo"

echo "✅ Variables configured for Project ID: $PROJECT_ID"
echo "Target Cluster: $CLUSTER_NAME"
echo "Target Repo: $REPO"
echo ""

# --- Phase 2: Delete GCP Resources ---

# Delete the GKE cluster
# This will also delete the associated Load Balancer and other GKE-managed resources.
echo "☸️ Deleting GKE cluster '$CLUSTER_NAME' in zone '$ZONE'..."
echo "This may take several minutes."
gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet

echo "✅ GKE cluster deleted."

# Delete the Artifact Registry repository
echo "🖼️ Deleting Artifact Registry repository '$REPO' in region '$REGION'..."
gcloud artifacts repositories delete $REPO --location=$REGION --quiet

echo "✅ Artifact Registry repository deleted."

# --- Phase 3: Clean Up Local Processes ---

# Stop the background MLflow server screen session
echo "🛑 Stopping the local MLflow server process..."
screen -X -S mlflow_server quit || echo "No 'mlflow_server' screen session found to stop."
pkill -f "mlflow server" || echo "No MLflow server process found to kill."

echo "✅ Local MLflow server process cleaned up."
echo ""
echo "🎉 Cleanup complete! All specified GCP resources and local processes have been removed."
