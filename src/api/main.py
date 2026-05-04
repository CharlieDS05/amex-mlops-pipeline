from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from datetime import datetime
import mlflow.pyfunc
import pandas as pd
import os
import glob

from .schemas import CustomerFeatures, PredictionResponse

# Path inside the container (mounted from ./mlruns on host)
MLRUNS_PATH = "/app/mlruns"

models = {}


def find_champion_model():
    """
    Find the XGBoost champion model artifacts in the mounted mlruns folder.
    
    The artifacts are saved under various paths depending on MLflow version:
      - mlruns/<exp_id>/<run_id>/artifacts/model/    (older format)
      - mlruns/<exp_id>/models/m-<id>/artifacts/     (newer format)
    """
    # Try newer format first
    patterns = [
        f"{MLRUNS_PATH}/*/models/m-*/artifacts/MLmodel",
        f"{MLRUNS_PATH}/*/*/artifacts/model/MLmodel",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        # Filter for XGBoost models (look for xgboost in the model files)
        for match in matches:
            model_dir = os.path.dirname(match)
            mlmodel_path = match
            try:
                content = open(mlmodel_path).read()
                if 'xgboost' in content.lower():
                    return model_dir
            except Exception:
                continue
    
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load champion model at API startup."""
    print(f"Searching for champion model in {MLRUNS_PATH}...")
    
    model_dir = find_champion_model()
    
    if model_dir is None:
        print(f"❌ Could not find XGBoost model in {MLRUNS_PATH}")
        # List what's there for debugging
        if os.path.exists(MLRUNS_PATH):
            print(f"Contents of {MLRUNS_PATH}:")
            for item in os.listdir(MLRUNS_PATH)[:10]:
                print(f"  {item}")
    else:
        print(f"📦 Loading model from: {model_dir}")
        try:
            models["champion"] = mlflow.pyfunc.load_model(model_dir)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    yield
    models.clear()


app = FastAPI(
    title="AmEx Default Risk API",
    description="Predicts probability of credit card default within 120 days",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "AmEx Default Risk API",
        "status": "running",
        "model_loaded": "champion" in models,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok" if "champion" in models else "degraded",
        "model_loaded": "champion" in models,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_default(payload: CustomerFeatures):
    if "champion" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check API logs.",
        )

    try:
        features_df = pd.DataFrame([payload.features])
        probability = float(models["champion"].predict(features_df)[0])

        if probability >= 0.7:
            tier = "high"
        elif probability >= 0.3:
            tier = "medium"
        else:
            tier = "low"

        return PredictionResponse(
            customer_id=payload.customer_id,
            default_probability=probability,
            risk_tier=tier,
            model_version="1.0.0",
            prediction_timestamp=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}",
        )