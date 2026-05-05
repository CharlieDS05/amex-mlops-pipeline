# AmEx Default Risk — End-to-End MLOps Pipeline

[![CI](https://github.com/CharlieDS05/amex-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/CharlieDS05/amex-mlops-pipeline/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade MLOps pipeline for predicting credit card default for American Express customers. End-to-end implementation covering data engineering, model training, hyperparameter optimization, experiment tracking, automated testing, and containerized deployment.

## Project Overview

This project implements a complete MLOps pipeline for the [AmEx Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) Kaggle competition. The goal is to predict the probability that a customer will default on their credit card balance within 120 days, using 18 months of transaction history per customer.

### Key Results

| Model | M Score (OOF) | ROC-AUC | PR-AUC |
|---|---|---|---|
| **XGBoost** | **0.7929** | 0.9616 | 0.8987 |
| LightGBM | 0.7919 | 0.9613 | 0.8980 |
| CatBoost | 0.7913 | 0.9611 | 0.8977 |

Champion model selected based on out-of-fold cross-validation with the official AmEx M Score metric.

## Tech Stack

**Machine Learning**
- LightGBM, XGBoost, CatBoost (gradient boosting)
- scikit-learn (baseline & utilities)
- Optuna (hyperparameter optimization)
- SHAP (model explainability)

**MLOps Infrastructure**
- MLflow (experiment tracking & model registry)
- Docker + Docker Compose (containerization)
- PostgreSQL (MLflow backend store)
- MinIO (S3-compatible artifact storage)

**API & Serving**
- FastAPI (REST API)
- Pydantic (data validation)
- Uvicorn (ASGI server)

**Testing & CI/CD**
- pytest (unit & integration tests)
- pytest-cov (coverage reporting)
- GitHub Actions (continuous integration)

## Architecture
┌──────────────────────────────────────────────────────────┐
│                  Training Pipeline                       │
│  Data → Feature Engineering → Optuna HPO → MLflow Logs   │
└──────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────┐
│              MLflow Stack (Dockerized)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐        │
│  │ MLflow   │  │ Postgres │  │ MinIO (S3)       │        │
│  │  :5000   │  │ Backend  │  │ Artifact Store   │        │
│  └──────────┘  └──────────┘  └──────────────────┘        │
└──────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────┐
│                  FastAPI Service                         │
│  /predict — POST customer features → default probability │
│  /docs — Auto-generated OpenAPI documentation            │
│  :8000                                                   │
└──────────────────────────────────────────────────────────┘

## Quick Start

### Prerequisites
- Docker Desktop installed
- Python 3.11+ (for local development)
- 8 GB RAM minimum

### Run the full stack
```bash
git clone https://github.com/CharlieDS05/amex-mlops-pipeline.git
cd amex-mlops-pipeline
docker compose up -d
```

This starts:
- MLflow tracking server: http://localhost:5000
- MinIO console: http://localhost:9001
- FastAPI service: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs

### Local development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src/api --cov=src/metrics
```

## Methodology

### Data Pipeline
The raw dataset contains ~5.5M monthly statements across ~460K unique customers, with 190 anonymized features. The pipeline aggregates these to customer-level features using mean, max, and last-statement aggregations, plus temporal features (statement count, months active), resulting in ~950 features per customer.

### Hyperparameter Optimization
Each model family (LightGBM, XGBoost, CatBoost) underwent 30 trials of Bayesian optimization with Optuna's TPE sampler. Each trial used 3-fold StratifiedKFold cross-validation with the official AmEx M Score as the optimization target. The implementation is fully **resumable** — interrupted training can be continued from any point.

### Evaluation
The corrected M Score metric combines:
- **Normalized Gini coefficient** (50% weight)
- **Default rate captured at top 4%** (50% weight)

Results were validated through proper out-of-fold predictions to ensure unbiased generalization estimates.

## Testing

The project includes 14 automated tests covering:
- Custom AmEx M Score metric (5 tests)
- FastAPI endpoints with mocked models (9 tests)

Tests run automatically on every push via GitHub Actions.

## Project Structure

amex-mlops-pipeline/
├── .github/workflows/      # CI/CD configuration
├── docker/                 # Dockerfiles
│   ├── api/
│   └── mlflow/
├── src/
│   ├── api/                # FastAPI service
│   ├── metrics.py          # Custom AmEx metric
│   ├── data_pipeline.py    # Feature engineering
│   ├── train_lgbm.py       # LightGBM training
│   ├── train_xgboost.py    # XGBoost training
│   ├── train_catboost.py   # CatBoost training
│   └── reevaluate_models.py
├── tests/
│   ├── test_metrics.py
│   └── test_api.py
├── docker-compose.yml
├── requirements.txt
└── README.md

## Key Learnings

This project addressed several real-world MLOps challenges:

1. **Crash-resistant training** — implemented persistent Optuna storage so multi-hour training runs survive runtime restarts
2. **Metric debugging** — discovered and fixed a bug in the custom evaluation metric through systematic verification testing  
3. **Cross-platform development** — single codebase works on local machine, Google Colab, and Docker
4. **Honest evaluation** — used proper out-of-fold cross-validation to avoid overfitting-inflated scores

## Author

**Juan Ruiz**  
Master's in Data Science & Analytics - MIOTI  
[LinkedIn](https://www.linkedin.com/in/juan-carlos-ruiz-583850a8/) | [GitHub](https://github.com/CharlieDS05)

## License

MIT