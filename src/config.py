"""
Centralized configuration for the AmEx MLOps project.
Auto-detects whether running locally or in Google Colab.
"""
import os
from pathlib import Path

# Environment detection
IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

# Paths — adapt automatically to environment
if IS_COLAB:
    ROOT = Path("/content/amex-mlops")
    DRIVE_ROOT = Path("/content/drive/MyDrive/amex-mlops")
    DATA_RAW = DRIVE_ROOT / "data" / "raw"
    DATA_PROCESSED = DRIVE_ROOT / "data" / "processed"
    MLRUNS_PATH = DRIVE_ROOT / "mlruns"
    MLFLOW_DB_PATH = DRIVE_ROOT / "mlflow.db"
    OPTUNA_DB_PATH = DRIVE_ROOT / "optuna_studies.db"
else:
    ROOT = Path(__file__).parent.parent
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    MLRUNS_PATH = ROOT / "mlruns"
    MLFLOW_DB_PATH = ROOT / "mlflow.db"
    OPTUNA_DB_PATH = ROOT / "optuna_studies.db"


# MLflow configuration

MLFLOW_TRACKING_URI = f"file://{MLRUNS_PATH.absolute()}"

EXPERIMENT_NAME = "amex-default-prediction"


# Optuna configuration

OPTUNA_STORAGE_URI = f"sqlite:///{OPTUNA_DB_PATH.absolute()}"


# Training configuration

RANDOM_STATE = 42
N_SPLITS = 3
N_TRIALS = 30


# Dataset columns

TARGET_COL = "target"
CUSTOMER_ID_COL = "customer_ID"

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "0")) or None