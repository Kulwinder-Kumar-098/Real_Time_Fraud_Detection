import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "fraud_detection" / "static"
TEMPLATES_DIR = BASE_DIR / "fraud_detection" / "templates"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(DATA_DIR / "fraud_detection.db"))

# Model Artifacts
MODEL_ARTIFACTS_PATH = os.environ.get(
    "MODEL_ARTIFACTS_PATH", str(MODELS_DIR / "model_artifacts.joblib")
)
MODEL_METADATA_PATH = os.environ.get(
    "MODEL_METADATA_PATH", str(MODELS_DIR / "model_metadata.json")
)

# Cost Matrix for Threshold Optimization
# Asymmetric cost in fraud detection:
# Cost of False Negative (Missed fraud): Customer loss, chargeback fees, fraud remediation (~$350 avg)
# Cost of False Positive (Legitimate tx declined/blocked): Customer friction, call-center inquiry (~$20 avg)
COST_FALSE_NEGATIVE = float(os.environ.get("COST_FALSE_NEGATIVE", 350.0))
COST_FALSE_POSITIVE = float(os.environ.get("COST_FALSE_POSITIVE", 20.0))

# Decision Threshold Defaults
DEFAULT_DECISION_THRESHOLD = float(os.environ.get("DEFAULT_DECISION_THRESHOLD", 0.42))
HIGH_RISK_THRESHOLD = float(os.environ.get("HIGH_RISK_THRESHOLD", 0.70))
REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", 0.35))

# Simulation Settings
SIMULATION_INTERVAL_SEC = float(os.environ.get("SIMULATION_INTERVAL_SEC", 1.2))
SIMULATION_FRAUD_RATE = float(os.environ.get("SIMULATION_FRAUD_RATE", 0.08))

# Flask & Server
SECRET_KEY = os.environ.get("SECRET_KEY", "fraud-detection-internal-secret-key-prod-2026")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")

# Seed for reproducibility
RANDOM_SEED = 42
