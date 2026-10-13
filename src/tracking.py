"""Thin MLflow helpers so every trainer logs the same way.

MLflow is imported lazily, so serving code (API, Streamlit) can use
`read_metrics` without installing MLflow. This keeps the Docker images small.
"""
import json
import os
import warnings

from src.config import METRICS_DIR, MLFLOW_EXPERIMENT_PREFIX, MLFLOW_TRACKING_URI

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", message=".*Integer columns.*")


def setup(experiment):
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}-{experiment}")


def log_sklearn_model(model, name, registered_name, input_example=None):
    """Log the model and register a new version. Returns the version number or None."""
    import mlflow
    # cloudpickle: MLflow 3's default (skops) rejects HistGradientBoosting tree internals.
    info = mlflow.sklearn.log_model(model, name=name, registered_model_name=registered_name,
                                    input_example=input_example, serialization_format="cloudpickle")
    return getattr(info, "registered_model_version", None)


def promote(registered_name, version, alias="champion"):
    """Point the `champion` alias at a version (MLflow's replacement for stages)."""
    from mlflow.tracking import MlflowClient
    if version is None:
        return
    MlflowClient().set_registered_model_alias(registered_name, alias, str(version))


def write_metrics(section, metrics):
    (METRICS_DIR / f"{section}.json").write_text(json.dumps(metrics, indent=2, default=str))


def read_metrics():
    return {p.stem: json.loads(p.read_text()) for p in sorted(METRICS_DIR.glob("*.json"))}
