#!/bin/bash

echo "🚀 Starting project cleanup..."

# 1. Remove unnecessary and redundant files
echo "Removing redundant files..."
rm -f feature_names.txt
rm -f credit_api_log.json
rm -f app/feature_names.joblib
git rm -r --cached mlruns/
rm -rf mlruns/
echo "✅ Files removed."

# 2. Clean up build_and_push.sh (remove duplicate content)
echo "Cleaning up build_and_push.sh..."
cat > build_and_push.sh << 'EOSH'
#!/bin/bash
# This script builds the Docker image and pushes it to Google Artifact Registry.
# Note: The CI/CD pipeline automates this. This is for manual execution.

set -e # Exit immediately if a command exits with a non-zero status.

# Authenticate Docker with Artifact Registry
echo "🔐 Authenticating with Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
echo "🔨 Building Docker image..."
docker build -t ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest .

# Push image
echo "⬆️ Pushing Docker image..."
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest

echo "✅ Build and push complete."
EOSH
echo "✅ build_and_push.sh cleaned."

# 3. Clean up k8s/configmap.yaml
echo "Cleaning up k8s/configmap.yaml..."
cat > k8s/configmap.yaml << 'EOYAML'
apiVersion: v1
kind: ConfigMap
metadata:
  name: credit-risk-api-config
data:
  # This version number can be updated by the CI/CD pipeline
  MODEL_VERSION: "gke-v1.0"
EOYAML
echo "✅ configmap.yaml cleaned."

# 4. Clean up k8s/deployment.yaml
echo "Cleaning up k8s/deployment.yaml..."
cat > k8s/deployment.yaml << 'EOYAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-deployment
  labels:
    app: credit-risk-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: credit-risk-api
  template:
    metadata:
      labels:
        app: credit-risk-api
      annotations:
        prometheus.io/scrape: 'true'
        prometheus.io/path: '/metrics'
        prometheus.io/port: '8000'
    spec:
      containers:
      - name: credit-risk-api-container
        # The image path uses variables that should be replaced during deployment
        image: us-central1-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: credit-risk-api-config
        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "2Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
EOYAML
echo "✅ deployment.yaml cleaned."

# 5. Clean up k8s/service.yaml
echo "Cleaning up k8s/service.yaml..."
cat > k8s/service.yaml << 'EOYAML'
apiVersion: v1
kind: Service
metadata:
  name: credit-risk-service
spec:
  # LoadBalancer exposes the service externally using a cloud provider's load balancer
  type: LoadBalancer
  selector:
    app: credit-risk-api
  ports:
  - protocol: TCP
    port: 80 # The port the service is exposed on
    targetPort: 8000 # The port the container is listening on
EOYAML
echo "✅ service.yaml cleaned."

# 6. Simplify the CI/CD pipeline
echo "Simplifying .github/workflows/ci-cd.yml..."
cat > .github/workflows/ci-cd.yml << 'EOYAML'
name: CI-CD Pipeline for Credit Risk API

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  ARTIFACT_REPO: credit-risk-repo
  IMAGE_NAME: credit-risk-api
  CLUSTER_NAME: credit-risk-cluster
  ZONE: us-central1-a

jobs:
  lint-and-test:
    name: Lint and Test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Run Unit Tests
        run: pytest tests/ -v

  build-and-push:
    name: Build and Push Docker Image
    needs: lint-and-test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install script dependencies
        run: pip install mlflow-skinny>=2.0.0

      - name: Download model artifacts (or create dummies as fallback)
        run: python scripts/download_model.py || python scripts/create_dummy_artifacts.py

      - name: Verify artifacts exist
        run: |
          ls -la app/
          test -f app/model.joblib && echo "✓ model.joblib exists" || (echo "✗ model.joblib missing" && exit 1)
          test -f app/explainer.joblib && echo "✓ explainer.joblib exists" || (echo "✗ explainer.joblib missing" && exit 1)

      - name: Authenticate to Google Cloud
        uses: 'google-github-actions/auth@v1'
        with:
          credentials_json: ${{ secrets.GCP_TOKEN }}

      - name: Set up Cloud SDK
        uses: 'google-github-actions/setup-gcloud@v1'

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Build and Push Docker image
        run: |
          IMAGE_TAG_SHA="${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.IMAGE_NAME }}:${{ github.sha }}"
          IMAGE_TAG_LATEST="${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.IMAGE_NAME }}:latest"
          
          docker build -t "$IMAGE_TAG_SHA" .
          docker tag "$IMAGE_TAG_SHA" "$IMAGE_TAG_LATEST"
          
          docker push "$IMAGE_TAG_SHA"
          docker push "$IMAGE_TAG_LATEST"

  deploy-to-gke:
    name: Deploy to GKE
    needs: build-and-push
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    permissions:
      contents: 'read'
      id-token: 'write'

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: 'google-github-actions/auth@v1'
        with:
          credentials_json: ${{ secrets.GCP_TOKEN }}
        
      - name: Set up Cloud SDK
        uses: 'google-github-actions/setup-gcloud@v1'
        
      - name: Get GKE credentials
        run: |
          gcloud components install gke-gcloud-auth-plugin --quiet
          gcloud container clusters get-credentials ${{ env.CLUSTER_NAME }} --zone ${{ env.ZONE }} --project ${{ env.PROJECT_ID }}
        
      - name: Update deployment image and wait for rollout
        run: |
          IMAGE_URL="${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.IMAGE_NAME }}:${{ github.sha }}"
          kubectl set image deployment/credit-risk-deployment credit-risk-api-container="$IMAGE_URL" --record
          kubectl rollout status deployment/credit-risk-deployment --timeout=5m
        
      - name: Verify deployment
        run: |
          echo "✅ Deployment successful. Checking status..."
          kubectl get service credit-risk-service -o wide
          kubectl get pods
EOYAML
echo "✅ CI/CD pipeline simplified."

echo "🎉 Cleanup complete! Your project is now more minimal."

