# Airflow Orchestration

DAG: `travel_ml_pipeline` ([dags/travel_ml_pipeline.py](dags/travel_ml_pipeline.py)). It runs weekly and can also be triggered manually.

```
ingest_raw_data → validate_data → check_drift → ┬ train_flight_price ┬ → quality_gate ─┬→ deploy
                                                ├ train_gender       ┤                 └→ notify_failure
                                                └ train_recommender  ┘
```

| Task | What it does | Fails when |
|---|---|---|
| `ingest_raw_data` | Moves new CSVs from `data/landing/` into `data/` and archives the old ones in `data/archive/<timestamp>/` | — |
| `validate_data` | Schema, null, price, `from != to`, `total = price × days` and referential-integrity checks | Any check fails |
| `check_drift` | PSI on key features. Writes `reports/drift_report.json` | — (informational) |
| `train_*` | Train, evaluate, log to MLflow, register the model, promote it to `@champion` | The model's quality gate fails |
| `quality_gate` | Reads `models/metrics/*.json` and branches | — (runs with `trigger_rule=all_done`) |
| `deploy` | Triggers the Jenkins job (`JENKINS_DEPLOY_URL`) to build and roll out new images | Jenkins call fails |
| `notify_failure` | Logs a failure message. Swap in a Slack/email operator for production | — |

Design choices:
* **Training runs in the project's `.venv` through `BashOperator`.** Airflow's own environment does not need
  scikit-learn or MLflow, which avoids dependency conflicts between Airflow and the ML stack.
* **The three trainers run in parallel.** Each writes its own metrics file, so there are no write races.
* **Retries:** 1 retry after 5 minutes. Each task has a 1-hour timeout. `max_active_runs=1` prevents overlapping retrains.
* **Compatible with Airflow 2.7+ and 3.x.** It was tested end to end with `airflow dags test` on Airflow 3.3.2.

## Run locally

Install Airflow in a **separate** virtualenv:

```bash
python3 -m venv ~/airflow-venv && ~/airflow-venv/bin/pip install apache-airflow
export PATH=~/airflow-venv/bin:$PATH
make airflow                     # airflow standalone → http://localhost:8080 (password printed in the logs)
```
Or run the whole DAG once without the scheduler:
```bash
AIRFLOW__CORE__DAGS_FOLDER=$PWD/airflow/dags TRAVEL_PROJECT_ROOT=$PWD airflow dags test travel_ml_pipeline
```

| Env var | Default |
|---|---|
| `TRAVEL_PROJECT_ROOT` | repository root |
| `TRAVEL_PYTHON` | `<root>/.venv/bin/python` |
| `TRAVEL_LANDING_DIR` | `<root>/data/landing` |
| `JENKINS_DEPLOY_URL`, `JENKINS_USER`, `JENKINS_TOKEN` | unset → the deploy step is skipped |

## Retraining on new data
Drop fresh `flights.csv` / `hotels.csv` / `users.csv` into `data/landing/` and trigger the DAG.
The data is validated and checked for drift, then all models are retrained. If they pass their quality gates, Jenkins is triggered
to ship new images.
