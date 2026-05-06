from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from datetime import datetime
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
import glob

from .schemas import CustomerFeatures, PredictionResponse

MLRUNS_PATH = "/app/mlruns"

models = {}
schema_info = {}  


def find_champion_model():
    """
    Find the XGBoost champion model artifacts.
    """
    patterns = [
        f"{MLRUNS_PATH}/*/models/m-*/artifacts/MLmodel",
        f"{MLRUNS_PATH}/*/*/artifacts/model/MLmodel",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            try:
                content = open(match).read()
                if 'xgboost' in content.lower():
                    return os.path.dirname(match)
            except Exception:
                continue
    return None


def extract_schema_dtypes(model_dir: str) -> dict:
    """
    Read the MLmodel YAML to extract the exact dtype expected per column.
    Returns a dict: {column_name: numpy_dtype}.
    """
    import yaml
    
    mlmodel_path = os.path.join(model_dir, "MLmodel")
    with open(mlmodel_path) as f:
        meta = yaml.safe_load(f)
    
    # Navigate MLflow's schema spec
    signature = meta.get("signature", {})
    inputs = signature.get("inputs", "[]")
    
    # inputs is a JSON-encoded string
    import json
    schema = json.loads(inputs)
    
    type_map = {
        "float": "float32",
        "double": "float64",
        "integer": "int32",
        "long": "int64",
        "boolean": "bool",
    }
    
    return {
        col["name"]: type_map.get(col.get("type", "float"), "float32")
        for col in schema
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load champion model and extract its schema at startup."""
    print(f"Searching for champion model in {MLRUNS_PATH}...")
    model_dir = find_champion_model()
    
    if model_dir is None:
        print(f"❌ Could not find XGBoost model in {MLRUNS_PATH}")
    else:
        print(f"📦 Loading model from: {model_dir}")
        try:
            models["champion"] = mlflow.pyfunc.load_model(model_dir)
            schema_info["dtypes"] = extract_schema_dtypes(model_dir)
            print(f"✅ Model loaded with schema for "
                  f"{len(schema_info['dtypes'])} features")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    
    yield
    models.clear()
    schema_info.clear()


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
        # Build DataFrame from payload
        features_df = pd.DataFrame([payload.features])
        
        # Cast each column to the model's expected dtype
        dtypes = schema_info.get("dtypes", {})
        for col, dtype in dtypes.items():
            if col in features_df.columns:
                # Handle integer types: NaN → 0, then cast
                if dtype in ("int32", "int64"):
                    features_df[col] = (
                        features_df[col].fillna(0).astype(dtype)
                    )
                else:
                    features_df[col] = features_df[col].astype(dtype)
        
        # Reorder columns to match schema (model expects exact order)
        if dtypes:
            ordered_cols = [c for c in dtypes.keys() if c in features_df.columns]
            features_df = features_df[ordered_cols]
        
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}",
        )