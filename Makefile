PY ?= .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data train monitor test api ui mlflow-ui docker-build docker-up k8s-deploy minikube-deploy airflow lint clean

setup:            ## create virtualenv and install everything
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-dev.txt

data:             ## ONLY for demos without the real dataset: overwrites data/*.csv with synthetic data
	$(PY) scripts/generate_data.py

train:            ## validate data + train all models + log to MLflow
	$(PY) -m src.pipeline

monitor:          ## PSI drift report -> reports/drift_report.json
	$(PY) -m src.monitor

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/flake8 src api tests streamlit_app airflow/dags --max-line-length 120

api:              ## Flask dev server on :5000
	$(PY) -m api.app

ui:               ## Streamlit on :8501
	.venv/bin/streamlit run streamlit_app/app.py

mlflow-ui:        ## MLflow UI on :5001
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-build:
	docker build -t travel-price-api:latest . && docker build -f Dockerfile.streamlit -t travel-streamlit:latest .

docker-up:
	docker compose up --build

minikube-deploy: docker-build   ## build into minikube's docker and deploy
	minikube image load travel-price-api:latest && minikube image load travel-streamlit:latest
	kubectl apply -k k8s/
	kubectl -n travel-ml rollout status deployment/travel-price-api

airflow:          ## run Airflow standalone with this repo's DAGs (needs `pip install apache-airflow` in a separate env)
	AIRFLOW_HOME=$(CURDIR)/airflow AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/airflow/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False TRAVEL_PROJECT_ROOT=$(CURDIR) airflow standalone

clean:
	rm -rf mlruns mlflow.db models/*.joblib models/metrics/*.json .pytest_cache
