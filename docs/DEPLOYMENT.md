# Deployment Guide — Docker & Jenkins

## Docker

Two images:

| Image | Dockerfile | Contents | Port |
|---|---|---|---|
| `travel-price-api` | `Dockerfile` | `src/`, `api/`, `models/`. Runs gunicorn. | 5000 |
| `travel-streamlit` | `Dockerfile.streamlit` | `src/`, `streamlit_app/`, `models/`, `data/`, `reports/` | 8501 |

API image design:
* **Multi-stage build.** Dependencies are installed in a builder stage and only the installed packages are copied over, which gives a smaller final image.
* **Non-root user** (uid 1000), matching `runAsNonRoot` in Kubernetes.
* **Built-in `HEALTHCHECK`** on `/health`.
* **Models are baked into the image.** The image tag `<build>-<sha>` is therefore an immutable model + code
  version, and rollback means redeploying an older tag.

```bash
make train                                   # models must exist before building
docker build -t travel-price-api .
docker run -p 8000:8000 travel-price-api
docker compose up --build                    # API :8000 + Streamlit :8501 (UI calls the API)
```

## Jenkins CI/CD

[`Jenkinsfile`](../Jenkinsfile) is a declarative pipeline triggered on every push (use a webhook or SCM polling).

```
Checkout → Setup venv → Lint (flake8) → Data validation + drift → Train & track (MLflow, quality gates)
→ Tests (JUnit report) → Build images (parallel) → Container smoke test → Push → Deploy to K8s (auto-rollback)
```

| Stage | Guards against |
|---|---|
| Lint | Style errors and unused imports |
| Data validation | Bad or changed input data (schema, nulls, integrity) |
| Train & track | Model regressions: each trainer exits non-zero if below `MIN_FLIGHT_R2` / `MIN_GENDER_F1` |
| Tests | API contract breaks (24 tests) |
| Container smoke test | An image that builds but doesn't serve (`/health` + a real `/predict`) |
| Deploy | A bad rollout: `kubectl rollout status` with a 180 s timeout, then `rollout undo` if it fails |

### Run it locally
See **[jenkins/README.md](../jenkins/README.md)**: `make jenkins-up && make jenkins-build` starts a pre-configured
Jenkins and runs this pipeline against your local Kubernetes. Verified green in 3 min 40 s.

### Setup on a shared Jenkins server
1. Use a Jenkins agent that has Python 3.12+, Docker and kubectl.
2. Add the credentials:
   * `docker-registry-creds`: username/password for your registry (e.g. Docker Hub)
   * `kubeconfig`: a secret file for the target cluster
3. Create a **Pipeline from SCM** job that points at this repo.
4. Build parameters: `REGISTRY` (e.g. `docker.io/<user>`), `DEPLOY` (push + deploy), and `RETRAIN`
   (set it to false to reuse the models in the workspace).

### How Airflow and Jenkins work together
Airflow owns the **data and model lifecycle** (scheduled retraining on new data). Jenkins owns the **code and release
lifecycle** (every commit is tested and shipped). When Airflow produces models that pass the gates, its `deploy`
task triggers the Jenkins job, so new models reach production through the same tested path as code changes.
