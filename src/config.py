import os
from pathlib import Path

# Detect environment
IS_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

# Paths adapt automatically
if IS_COLAB:
    ROOT = Path("/content/amex-mlops")
    DRIVE_ROOT = Path("/content/drive/MyDrive/amex-mlops")
    DATA_RAW = DRIVE_ROOT / "data" / "raw"
    DATA_PROCESSED = DRIVE_ROOT / "data" / "processed"
    MLRUNS_PATH = DRIVE_ROOT / "mlruns"
else:
    ROOT = Path(__file__).parent.parent
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    MLRUNS_PATH = ROOT / "mlruns"

# MLflow tracking — local file store works in both environments
MLFLOW_TRACKING_URI = f"file://{MLRUNS_PATH.absolute()}"

EXPERIMENT_NAME = "amex-default-prediction"
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50
TARGET_COL = "target"
CUSTOMER_ID_COL = "customer_ID"

# For quick testing — set via environment variable
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "0")) or None