
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "amex-default-prediction"

RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50
TARGET_COL = "target"
CUSTOMER_ID_COL = "customer_ID"
