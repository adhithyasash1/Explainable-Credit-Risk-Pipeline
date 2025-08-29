#!/bin/bash
# This script automates the entire deployment process for the Credit Risk API on Google Cloud.
# It sets up infrastructure, trains the model, packages the application, and deploys it to GKE.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting Full Deployment Process for Credit Risk API..."

# --- Phase 1: GCP Environment and Infrastructure Setup ---
echo "--- PHASE 1: GCP Setup ---"

# Set your Google Cloud Project ID
export PROJECT_ID=$(gcloud config get-value project)

# Configure GCP settings
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="credit-risk-cluster"

# Configure application-specific names
export REPO="credit-risk-repo"
export IMAGE_NAME="credit-risk-api"

echo "✅ Variables configured for Project ID: $PROJECT_ID"

# Enable necessary GCP services
echo "🔐 Enabling required Google Cloud APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com

# Create Artifact Registry repository
echo "🖼️ Creating Artifact Registry repository '$REPO'..."
gcloud artifacts repositories create $REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for credit risk API" \
    || echo "Repository '$REPO' already exists."

# Create GKE cluster (this may take 5-10 minutes)
echo "☸️ Creating GKE cluster '$CLUSTER_NAME'... This might take a few minutes."
gcloud container clusters create $CLUSTER_NAME \
    --num-nodes=2 \
    --machine-type=e2-standard-2 \
    --zone=$ZONE \
    --release-channel=regular \
    || echo "Cluster '$CLUSTER_NAME' already exists."

# Get credentials to connect kubectl to your new cluster
echo "🔗 Configuring kubectl to connect to the cluster..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

echo "✅ GCP Infrastructure is ready."

# --- Phase 2: Train and Register Model with MLflow ---
echo "--- PHASE 2: Model Training ---"

# Install Python dependencies
echo "🐍 Installing Python requirements..."
pip install -r requirements.txt

# Kill any old MLflow servers
echo "🛑 Stopping any existing MLflow servers..."
pkill -f "mlflow server" || echo "No existing MLflow server found."
sleep 2

# Start a new MLflow server in a detached screen session
echo "🚀 Starting new MLflow server in a background 'screen' session..."
screen -S mlflow_server -d -m mlflow server --host 0.0.0.0

echo "✅ MLflow server is running. Access the UI at http://$(curl -s ifconfig.me):5000"

# Run the training script
echo "🏋️‍♀️ Training the model... This will create a new run in MLflow."
python scripts/train.py

# MANUAL STEP: Promote the model to Production
echo "------------------------------------------------------------------"
echo "👉 ACTION REQUIRED: Promote the model in the MLflow UI."
echo "1. Open your browser to http://$(curl -s ifconfig.me):5000"
echo "2. Find the latest run under the 'Credit Risk Prediction' experiment."
echo "3. Register the model with the name 'credit-risk-xgb'."
echo "4. Go to the 'Models' page and transition the latest version to 'Production'."
echo "------------------------------------------------------------------"
read -p "Once you have promoted the model to 'Production', press ENTER to continue..."

echo "✅ Model promotion confirmed."

# --- Phase 3: Package Application into a Docker Image ---
echo "--- PHASE 3: Build and Push Docker Image ---"

# Download the production model artifacts from MLflow into the app/ directory
echo "📦 Downloading production model artifacts..."
python scripts/download_model.py

# Authenticate Docker with Artifact Registry
echo "🔐 Authenticating Docker with Google Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build the Docker image
IMAGE_URI="${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest"
echo "🔨 Building the Docker image: $IMAGE_URI"
docker build -t $IMAGE_URI .

# Push the Docker image to Artifact Registry
echo "⬆️ Pushing the image to Artifact Registry..."
docker push $IMAGE_URI

echo "✅ Docker image has been successfully built and pushed."

# --- Phase 4: Deploy the Application to GKE ---
echo "--- PHASE 4: Deploying to GKE ---"

# Substitute variables in the deployment manifest
echo "✍️ Updating Kubernetes deployment manifest..."
sed -i "s/\$PROJECT_ID/$PROJECT_ID/g" k8s/deployment.yaml
sed -i "s/\$REPO/$REPO/g" k8s/deployment.yaml
sed -i "s/\$IMAGE_NAME/$IMAGE_NAME/g" k8s/deployment.yaml

# Apply all Kubernetes manifests
echo "🚀 Applying Kubernetes manifests..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

echo "✅ Application manifests applied to the GKE cluster."

# --- Phase 5: Verify and Test the Deployment ---
echo "--- PHASE 5: Verification ---"

# Check the rollout status
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/credit-risk-deployment --timeout=5m

# Get the external IP address of the service
echo "🌐 Fetching the external IP address for the service..."
EXTERNAL_IP=""
while [ -z $EXTERNAL_IP ]; do
  echo "Waiting for Load Balancer to provision an IP..."
  EXTERNAL_IP=$(kubectl get svc credit-risk-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  [ -z "$EXTERNAL_IP" ] && sleep 10
done

echo "🎉 DEPLOYMENT COMPLETE! 🎉"
echo "Your Credit Risk API is now live at: http://$EXTERNAL_IP"
echo ""

# Test the /health endpoint
echo "🩺 Testing /health endpoint..."
curl http://$EXTERNAL_IP/health
echo ""

# Test the /explain endpoint
echo "\n💡 Testing /explain endpoint with sample data..."
curl -s -X POST http://$EXTERNAL_IP/explain \
-H "Content-Type: application/json" \
-d @test_request.json | python -m json.tool

echo "\n✅ Script finished."
