# explainable credit risk pipeline

production-ready credit scoring api with ml explainability, containerized deployment, and automated infrastructure provisioning on google cloud platform.

## overview

end-to-end machine learning pipeline for credit risk assessment featuring xgboost modeling with shap explainability, fastapi serving, kubernetes orchestration, and automated deployment. implements transparent, auditable credit decisions with business-friendly explanations.

## architecture

**core ml pipeline:**
- xgboost classifier with class balancing and early stopping
- shap tree explainer for local and global feature importance
- mlflow experiment tracking with model registry
- automated feature engineering with financial ratio calculations

**production api:**
- fastapi with gunicorn workers and uvicorn async handling
- structured json logging with request correlation
- prometheus metrics instrumentation
- kubernetes health checks and readiness probes

**deployment automation:**
- one-click gcp infrastructure provisioning
- docker multi-stage builds with security hardening
- kubernetes deployment with horizontal pod autoscaling
- ci/cd pipeline with automated testing and deployment

## project structure

```
├── app/
│   ├── main.py              # fastapi application with prediction endpoints
│   └── utils.py             # feature engineering and explanation utilities
├── k8s/
│   ├── deployment.yaml      # kubernetes deployment configuration  
│   ├── service.yaml         # load balancer service
│   ├── hpa.yaml            # horizontal pod autoscaler (2-5 pods, 75% cpu)
│   └── configmap.yaml      # environment configuration
├── scripts/
│   ├── train.py            # xgboost training with mlflow integration
│   ├── download_model.py   # model artifact retrieval from mlflow registry
│   ├── test_api_manual.py  # api endpoint testing
│   └── create_dummy_artifacts.py  # ci/cd fallback artifacts
├── tests/
│   ├── test_api.py         # fastapi endpoint unit tests
│   └── test_feature_engineering.py  # feature pipeline tests
├── deploy_on_gcp.sh        # automated gcp deployment script
├── Dockerfile              # multi-stage container build
└── .github/workflows/ci-cd.yml  # ci/cd pipeline with gcp integration
```

### key components

**app/main.py** - production fastapi server implementing:
- lifespan context management for model loading
- request middleware with uuid correlation
- custom exception handling with structured logging
- prometheus instrumentator integration
- health monitoring and global importance endpoints

**app/utils.py** - feature processing pipeline:
- financial ratio calculations (debt-to-income, loan-to-price, loan-to-income)
- categorical encoding with age groups and income brackets
- shap value formatting with business-friendly explanations
- risk tier classification (tier 1-5 based on probability thresholds)

**scripts/train.py** - ml training workflow:
- data preprocessing with categorical encoding and ratio features
- xgboost hyperparameter configuration with scale_pos_weight
- model evaluation using auc and ks statistic
- mlflow logging with artifact registration
- shap explainer generation and global importance calculation

**deploy_on_gcp.sh** - infrastructure automation:
- gcp service enablement (artifact registry, container, iam)
- gke cluster provisioning with e2-standard-2 nodes
- docker authentication and image building
- kubernetes manifest deployment with variable substitution

## deployment workflow

### local development
```bash
git clone https://github.com/adhithyasash1/explainable-credit-risk-pipeline.git
cd explainable-credit-risk-pipeline

pip install -r requirements.txt
python scripts/train.py
uvicorn app.main:app --reload
python scripts/test_api_manual.py
```

### google cloud deployment
```bash
gcloud config set project YOUR_PROJECT_ID
chmod +x deploy_on_gcp.sh
./deploy_on_gcp.sh
```

**deployment process:**
1. **infrastructure setup**: creates artifact registry repository and gke cluster
2. **model training**: runs mlflow experiment and registers model artifacts
3. **containerization**: builds multi-stage docker image with security hardening
4. **kubernetes deployment**: applies manifests with configmap, deployment, service, hpa
5. **verification**: health checks and load balancer ip provisioning

## api endpoints

### health monitoring
```http
GET /health
```
response:
```json
{
  "status": "ok",
  "model_version": "gke-v1.0"
}
```

### basic prediction
```http
POST /predict
```

### explainable prediction
```http
POST /explain
```

**request payload:**
```json
{
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
```

**response structure:**
```json
{
  "request_id": "3a483d00-8862-40b2-9825-3a5e051d8768",
  "model_version": "gke-v1.0", 
  "prediction": "Approve",
  "probability_good_credit": 0.7445618510246277,
  "risk_tier": "Tier 2: Low Risk",
  "expected_loss": 344.84149169921875,
  "explanation": [
    {
      "feature": "Assets",
      "explanation": "The feature 'Assets' with value '10000.0' significantly decreased the risk of default."
    },
    {
      "feature": "Seniority",
      "explanation": "The feature 'Seniority' with value '10' significantly decreased the risk of default."
    },
    {
      "feature": "Debt", 
      "explanation": "The feature 'Debt' with value '2000.0' significantly increased the risk of default."
    },
    {
      "feature": "Records_2",
      "explanation": "The feature 'Records_2' with value '0' significantly decreased the risk of default."
    },
    {
      "feature": "Loan-to-Income",
      "explanation": "An income of $6 significantly decreased the predicted risk of default."
    }
  ]
}
```

### global feature importance
```http
GET /global-importance
```

returns top 10 features with mean absolute shap values across training dataset.

## business intelligence features

**risk tier classification:**
- tier 1: very low risk (probability > 0.85)
- tier 2: low risk (probability > 0.70)  
- tier 3: medium risk (probability > 0.50)
- tier 4: high risk (probability > 0.35)
- tier 5: very high risk (probability ≤ 0.35)

**expected loss calculation:**
```
expected_loss = probability_default × loss_given_default × loan_amount
```
assumes 90% loss given default for conservative risk estimation.

**feature engineering:**
- debt-to-income ratio for leverage assessment
- loan-to-price ratio for collateral coverage
- loan-to-income ratio for affordability analysis
- age grouping (young adult, adult, middle aged, senior)
- income bracketing (low, medium, high, very high)

## production deployment

**container specifications:**
- multi-stage build with python:3.9-slim base
- non-root user execution for security compliance
- health check with curl endpoint verification
- gunicorn with 4 uvicorn workers and 120s timeout

**kubernetes configuration:**
- 2 replica deployment with rolling update strategy
- resource limits: 1 cpu, 2gi memory
- resource requests: 250m cpu, 512mi memory
- horizontal pod autoscaler: 2-5 pods at 75% cpu utilization
- loadbalancer service exposing port 80 to container 8000

**monitoring and observability:**
- structured json logging with pythonjsonlogger
- prometheus metrics on /metrics endpoint
- request correlation with uuid tracking
- readiness probe: /health endpoint with 30s initial delay
- liveness probe: /health endpoint with 60s initial delay

## ci/cd pipeline

**github actions workflow:**
1. **lint and test**: flake8 linting and pytest unit tests
2. **build and push**: docker image build and artifact registry push
3. **deploy to gke**: kubernetes deployment with image update and rollout verification

**testing coverage:**
- api endpoint response validation
- feature engineering pipeline testing  
- health check and error handling verification
- payload validation and schema compliance

## technical implementation

**model training metrics:**
- auc score evaluation for binary classification performance
- ks statistic calculation for score distribution assessment
- early stopping with 20 rounds on validation auc
- class balancing with scale_pos_weight parameter

**explainability implementation:**
- shap tree explainer for xgboost model interpretation
- local explanations for individual predictions (top 5 features)
- global feature importance ranking with mean absolute shap values
- business-friendly explanation generation with impact narratives

**deployment verification:**
after successful deployment, the script provides external load balancer ip and verifies:
- /health endpoint accessibility
- /explain endpoint with sample prediction
- kubernetes pod status and service configuration
- prometheus metrics endpoint exposure

the system automatically scales from 2 to 5 pods based on cpu utilization, ensuring consistent performance under varying load conditions while maintaining cost efficiency.
