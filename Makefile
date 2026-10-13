PY ?= .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data train monitor test api ui mlflow-ui docker-build docker-up colima-up k8s-deploy k8s-forward minikube-deploy jenkins-up jenkins-build airflow lint clean

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

api:              ## Flask dev server on :8000
	$(PY) -m api.app

ui:               ## Streamlit on :8501
	.venv/bin/streamlit run streamlit_app/app.py

mlflow-ui:        ## MLflow UI on :5001
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-build:
	docker build -t travel-price-api:latest . && docker build -f Dockerfile.streamlit -t travel-streamlit:latest .

docker-up:
	docker compose up --build

colima-up:        ## Docker + Kubernetes (k3s) in one light VM - recommended on 8 GB Macs
	colima start --cpu 2 --memory 4 --disk 30 --kubernetes

k8s-deploy: docker-build   ## deploy to the current cluster (Colima k3s shares Docker's images - no push needed)
	kubectl apply -k k8s/
	kubectl -n travel-ml rollout status deployment/travel-price-api --timeout=180s
	kubectl -n travel-ml rollout status deployment/travel-streamlit --timeout=180s

k8s-forward:      ## API -> localhost:8000, Streamlit -> localhost:8501
	@kubectl cluster-info >/dev/null 2>&1 || { echo "Kubernetes is not reachable - start it first: colima start (wait ~1 min)"; exit 1; }
	kubectl -n travel-ml port-forward svc/travel-price-api 8000:80 & \
	kubectl -n travel-ml port-forward svc/travel-streamlit 8501:80

jenkins-up:       ## local Jenkins (http://localhost:8080) running this repo's Jenkinsfile - see jenkins/README.md
	docker build -t travel-jenkins:local jenkins/
	mkdir -p $(HOME)/.travel-jenkins && chmod 700 $(HOME)/.travel-jenkins
	colima ssh -- sudo cat /etc/rancher/k3s/k3s.yaml > $(HOME)/.travel-jenkins/kubeconfig
	chmod 600 $(HOME)/.travel-jenkins/kubeconfig
	docker rm -f jenkins >/dev/null 2>&1 || true
	docker run -d --name jenkins --restart unless-stopped --network host \
	  --group-add $$(colima ssh -- stat -c %g /var/run/docker.sock) \
	  -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock \
	  -v $(CURDIR):/repo:ro -v $(HOME)/.travel-jenkins/kubeconfig:/run/secrets/kubeconfig:ro \
	  travel-jenkins:local
	@echo "Jenkins starting on http://localhost:8080/job/travel-mlops/"

jenkins-build:    ## trigger the pipeline (commit first - Jenkins builds the committed main branch)
	@CJ=$$(mktemp); \
	CRUMB=$$(curl -s -c $$CJ localhost:8080/crumbIssuer/api/json | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['crumbRequestField']+':'+d['crumb'])"); \
	curl -s -o /dev/null -w "triggered (HTTP %{http_code})\n" -b $$CJ -H "$$CRUMB" -X POST \
	  "localhost:8080/job/travel-mlops/buildWithParameters?REGISTRY=local&DEPLOY=true&RETRAIN=true"; rm -f $$CJ

minikube-deploy: docker-build   ## build into minikube's docker and deploy
	minikube image load travel-price-api:latest && minikube image load travel-streamlit:latest
	kubectl apply -k k8s/
	kubectl -n travel-ml rollout status deployment/travel-price-api

airflow:          ## run Airflow standalone with this repo's DAGs (needs `pip install apache-airflow` in a separate env)
	AIRFLOW_HOME=$(CURDIR)/airflow AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/airflow/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False TRAVEL_PROJECT_ROOT=$(CURDIR) airflow standalone

clean:
	rm -rf mlruns mlflow.db models/*.joblib models/metrics/*.json .pytest_cache
