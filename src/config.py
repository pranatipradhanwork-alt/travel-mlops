"""Central configuration. Every path can be overridden with an environment variable
so the same code runs locally, in Docker, in Airflow and in Kubernetes."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))

USERS_CSV = DATA_DIR / "users.csv"
FLIGHTS_CSV = DATA_DIR / "flights.csv"
HOTELS_CSV = DATA_DIR / "hotels.csv"
WORLD_NAMES_CSV = DATA_DIR / "world_names.csv"  # labelled first names from many countries (gender model)

FLIGHT_MODEL_PATH = MODEL_DIR / "flight_price_model.joblib"
GENDER_MODEL_PATH = MODEL_DIR / "gender_model.joblib"
RECOMMENDER_PATH = MODEL_DIR / "hotel_recommender.joblib"
METRICS_DIR = MODEL_DIR / "metrics"  # one JSON per model -> safe for parallel training
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT_PREFIX = os.getenv("MLFLOW_EXPERIMENT_PREFIX", "travel")

FLIGHT_MODEL_NAME = "flight-price-regressor"
GENDER_MODEL_NAME = "gender-classifier"

# Quality gates used by the training scripts, Airflow and Jenkins.
MIN_FLIGHT_R2 = float(os.getenv("MIN_FLIGHT_R2", "0.90"))
MIN_GENDER_F1 = float(os.getenv("MIN_GENDER_F1", "0.80"))
# Drift: Population Stability Index above this on any feature raises an alert.
PSI_ALERT = float(os.getenv("PSI_ALERT", "0.2"))

RANDOM_STATE = 42

for d in (DATA_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
