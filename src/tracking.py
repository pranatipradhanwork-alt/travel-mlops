"""Thin MLflow helpers so every trainer logs the same way."""
import json
import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", message=".*Integer columns.*")
import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.config import METRICS_DIR, MLFLOW_EXPERIMENT_PREFIX, MLFLOW_TRACKING_URI  # noqa: E402


def setup(experiment):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}-{experiment}")


def log_sklearn_model(model, name, registered_name, input_example=None):
    """Log the model and register a new version. Returns the version number or None."""
    # cloudpickle: MLflow 3's default (skops) rejects HistGradientBoosting tree internals.
    info = mlflow.sklearn.log_model(model, name=name, registered_model_name=registered_name,
                                    input_example=input_example, serialization_format="cloudpickle")
    return getattr(info, "registered_model_version", None)


def promote(registered_name, version, alias="champion"):
    """Point the `champion` alias at a version (MLflow's replacement for stages)."""
    if version is None:
        return
    MlflowClient().set_registered_model_alias(registered_name, alias, str(version))


def write_metrics(section, metrics):
    (METRICS_DIR / f"{section}.json").write_text(json.dumps(metrics, indent=2, default=str))


def read_metrics():
    return {p.stem: json.loads(p.read_text()) for p in sorted(METRICS_DIR.glob("*.json"))}
