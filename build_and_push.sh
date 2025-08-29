#!/bin/bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
docker build -t ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest .

# Push image
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest
#!/bin/bash
# Authenticate Docker with Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
docker build -t ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest .

# Push image
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest
