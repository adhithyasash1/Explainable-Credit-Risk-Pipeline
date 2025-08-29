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
