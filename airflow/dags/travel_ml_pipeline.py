"""Objective 5: Airflow DAG orchestrating the travel ML workflow.

    ingest_raw_data -> validate_data -> check_drift -> [train_flight_price, train_gender, train_recommender]
        -> quality_gate -> (deploy | notify_failure)

Training runs as subprocesses in the project's own virtualenv so Airflow's
environment does not need the ML dependencies. Configure with env vars:
  TRAVEL_PROJECT_ROOT   path to this repo (default: two levels above this file)
  TRAVEL_PYTHON         interpreter to use   (default: <root>/.venv/bin/python)
  TRAVEL_LANDING_DIR    where new raw CSVs land (default: <root>/data/landing)
  JENKINS_DEPLOY_URL    optional Jenkins "build with parameters" URL to trigger a deploy
Works with Airflow 2.7+ and 3.x.
"""
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

ROOT = Path(os.getenv("TRAVEL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PYTHON = os.getenv("TRAVEL_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
LANDING = Path(os.getenv("TRAVEL_LANDING_DIR", ROOT / "data" / "landing"))
DATA = ROOT / "data"
METRICS_DIR = ROOT / "models" / "metrics"
FILES = ("users.csv", "flights.csv", "hotels.csv")


def ingest_raw_data(**_):
    """Move any newly landed CSVs into data/, keeping a timestamped backup of the old ones."""
    LANDING.mkdir(parents=True, exist_ok=True)
    new = [f for f in FILES if (LANDING / f).exists()]
    if not new:
        print("no new files in landing zone; retraining on current data")
        return []
    backup = DATA / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup.mkdir(parents=True)
    for f in new:
        if (DATA / f).exists():
            shutil.copy2(DATA / f, backup / f)
        shutil.move(str(LANDING / f), DATA / f)
    print(f"ingested {new}; previous versions in {backup}")
    return new


def quality_gate(**_):
    m = {p.stem: json.loads(p.read_text()) for p in METRICS_DIR.glob("*.json")}
    ok = all(m.get(k, {}).get("passed_gate") for k in ("flight_price", "gender")) and "recommender" in m
    print(json.dumps(m, indent=2))
    return "deploy" if ok else "notify_failure"


def notify_failure(**_):
    # Swap for a Slack/Email operator in production.
    print("Model quality gate FAILED - production models left unchanged. See MLflow for details.")


def run(module):
    return f'cd "{ROOT}" && MLFLOW_DISABLE_AGENT_HINT=1 "{PYTHON}" -m {module}'


DEPLOY_CMD = (
    'if [ -n "$JENKINS_DEPLOY_URL" ]; then '
    'curl -fsS -X POST "$JENKINS_DEPLOY_URL" --user "$JENKINS_USER:$JENKINS_TOKEN" && echo "Jenkins deploy triggered"; '
    'else echo "JENKINS_DEPLOY_URL not set - models saved to models/, skipping deploy"; fi'
)

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id="travel_ml_pipeline",
    description="Ingest travel data, retrain price/gender/recommender models, gate and deploy",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["travel", "mlops"],
) as dag:
    ingest = PythonOperator(task_id="ingest_raw_data", python_callable=ingest_raw_data)
    validate = BashOperator(task_id="validate_data", bash_command=run("src.validate_cli"))
    # Writes reports/drift_report.json (PSI per feature). Informational: retraining
    # runs either way, but the report shows *why* a new model version may differ.
    drift = BashOperator(task_id="check_drift", bash_command=run("src.monitor"))
    train_price = BashOperator(task_id="train_flight_price", bash_command=run("src.train_flight_price"))
    train_gender = BashOperator(task_id="train_gender", bash_command=run("src.train_gender"))
    train_rec = BashOperator(task_id="train_recommender", bash_command=run("src.train_recommender"))
    gate = BranchPythonOperator(task_id="quality_gate", python_callable=quality_gate,
                                trigger_rule="all_done")
    deploy = BashOperator(task_id="deploy", bash_command=DEPLOY_CMD)
    fail = PythonOperator(task_id="notify_failure", python_callable=notify_failure)

    ingest >> validate >> drift >> [train_price, train_gender, train_rec] >> gate >> [deploy, fail]
