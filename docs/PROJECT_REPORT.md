# Voyage Analytics — Final Project Report

**Travel MLOps Capstone**

---

## 1. Executive summary

We built a production-style machine-learning system for a travel platform. It delivers three
capabilities through a single REST API and an interactive web app:

1. **Flight price prediction:** an instant fare estimate for any route, cabin class and agency.
2. **Gender classification:** infers male/female from a user's first name.
3. **Hotel recommendation:** a ranked list of hotels for each traveller.

The models are only part of the work. The rest of the project is about running them reliably:
* every experiment is tracked and every model is versioned (MLflow);
* the service is packaged once and runs anywhere (Docker), scaling with traffic (Kubernetes);
* retraining is automated when new data arrives (Airflow);
* every change is tested and shipped by a pipeline with quality gates and automatic rollback (Jenkins).

| Model | Held-out result |
|---|---|
| Flight price (Random Forest) | R² 1.000, RMSE R$0.00 on future flights. Ridge fallback R² 0.88 on unseen routes. |
| Gender (char n-gram logistic regression) | 89.4% accuracy, macro F1 0.894. 76.4% on unseen names. |
| Hotel recommender (tuned hybrid) | hit@3 0.375, vs 0.329 for random and 0.365 for popularity. |

## 2. Business problem

Travel platforms compete on **price transparency** and **personalisation**:
* **Price prediction** lets users see expected fares before searching and lets the business detect mispriced inventory.
* **User profiling** (here, gender) supports audience analytics and segment-level reporting.
* **Recommendations** raise conversion by showing relevant hotels first.

A notebook can prove a model works once. A business needs the model **available in milliseconds,
reproducible, monitored, and refreshed as data changes**. That is why this project treats MLOps as a first-class goal.

## 3. Data

| Dataset | Rows | Key fields | Used for |
|---|---|---|---|
| users | 1,340 | code, company, name, gender, age | Gender model, recommender profiles |
| flights | 271,888 legs (135,944 round trips) | travelCode, userCode, from, to, flightType, price, time, distance, agency, date | Price model, home-city signal |
| hotels | 40,552 stays | travelCode, userCode, name, place, days, price, total, date | Recommender |

**Relationships:** `userCode` joins flights and hotels to `users.code`. `travelCode` joins a hotel stay to
its trip. Each trip has **two flight legs** (outbound and return), and every hotel stay is in the outbound
destination city (100% match).

**Data quality checks** (`src/data.py: validate`, run by Airflow and Jenkins): required columns, nulls,
positive prices, `from ≠ to`, `total = price × days`, and every userCode existing in users.
The capstone data passes all of them.

**Exploratory findings that shaped the design** (charts in `notebooks/travel_capstone_analysis.ipynb`, details in `reports/MODEL_EVALUATION.md` §1):
* The price is constant within each (route, class, agency) group, so it is a deterministic lookup.
* `time` and `distance` are perfectly correlated (r = 1.0), and dates have no effect on price.
* There are only 9 hotels, one per city, each with a fixed price.
* Destinations are nearly uniform for every origin and user segment.
* A third of users have gender "none". Their names are ordinary gendered names, so "none" means "not disclosed".

## 4. Solution design

### 4.1 Flight price regression
* **Features:** `from`, `to`, `flightType`, `agency` (one-hot) and `distance` (standardised). `time` and the
  calendar features were removed. An ablation run shows they change RMSE by only 0.06.
* **Candidates:** Ridge, Random Forest, HistGradientBoosting, plus the ablation run.
* **Evaluation:**
  * a **time-based split** (train before 2021-11-25, test after), which mirrors forecasting future bookings;
  * **unseen-route** 5-fold GroupKFold, which measures generalisation to new routes.
* **Result:** the Random Forest reproduces every known fare. On unseen routes, Ridge generalises best (R² 0.88 vs 0.70),
  so **both are served**: the champion for known routes and Ridge as an automatic fallback for new ones.
* **Interpretation:** permutation importance ranks distance first, then cabin class, origin, destination and agency.

### 4.2 Gender classification
* **Features:** TF-IDF of character 2–4-grams of the first name. These capture sub-word patterns, so the model can score names it has never seen.
* **Experiments logged in MLflow:**
  * 3-class (54% accuracy), which shows "none" cannot be predicted;
  * binary with behaviour features, which shows no reliable gain;
  * **binary name-only (served)**: 89.4% accuracy, 76.4% on unseen names.
* **Responsible-AI note:** the API states that the prediction is name-based and that users may not disclose gender.

### 4.3 Hotel recommendation
* **Signals:**
  * the user's history share;
  * item-item collaborative similarity;
  * global popularity (also the cold-start fallback);
  * fit to the user's usual nightly budget;
  * a penalty for the user's home city.
* **Tuning:** a grid search over 105 weight sets on a per-user **temporal** validation window. The scores are then reported on a later,
  untouched test window, with random tie-breaking so sparse signals cannot earn credit from tied zeros.
* **Result:** the tuned hybrid (collaborative + home penalty) beats random by +4.6 hit@3 points (+14% relative)
  and also beats popularity. The ceiling is low because the data has almost no destination-preference signal.

### 4.4 Serving: REST API (Flask)
* **Request flow:** JSON request → validation (missing fields, `from = to`, bad distance, unknown route) →
  pandas row → sklearn pipeline (preprocessing and model in one object, so training/serving skew is impossible) → JSON response with the
  model version used.
* **Endpoints:** `/predict` (single or batch), `/predict/gender`, `/recommend/<user>`, `/health`, `/metadata` and
  `/metrics` (Prometheus).
* **Server:** gunicorn with preloaded models (2 workers × 4 threads per pod). The test suite has 24 tests.

### 4.5 MLOps pipeline
| Tool | Role in this project |
|---|---|
| **MLflow** | 3 experiments. Every run logs parameters, metrics, figures and the model artifact with its signature. The Model Registry versions each model, and versions that pass the gates get the `@champion` alias. |
| **Docker** | Multi-stage, non-root API image with a healthcheck and the models baked in. Image tag = build + git SHA, so it identifies an exact code + model version. |
| **Kubernetes** | Deployment with rolling updates (`maxUnavailable: 0`), readiness/liveness probes, resource limits, HPA (2–10 pods on CPU/memory), PodDisruptionBudget and Ingress. Verified on k3s (Colima): under in-cluster load the HPA scaled 2→10 pods in 40 s and served 35,087 batch requests in 150 s. |
| **Airflow** | Weekly DAG: ingest → validate → drift → 3 parallel trainers → quality gate → deploy or alert. Tested end to end on Airflow 3.3.2 (61 s). |
| **Jenkins** | lint → data validation → train with gates → tests → parallel image builds → container smoke test → push → deploy with automatic rollback. Run end to end on a local Jenkins defined as code (3 min 40 s, 24/24 tests). The deployed image tag `1-<git sha>` ties the running pods to the exact commit. |
| **Monitoring** | PSI drift report (all features < 0.04 PSI, so stable), API `/metrics`, and a Streamlit monitoring tab. |

### 4.6 Streamlit application
Five tabs:
* **Recommendations:** user picker, budget filter, "only new hotels", reasons, history;
* **Travel insights:** KPIs, trends, price structure, origin→destination heatmap, spend by segment;
* **Flight price;**
* **Gender;**
* **Model monitoring:** model comparisons, drift chart, evaluation figures.

The app can call the API (`API_URL`) so the UI and the model service scale independently.

## 5. Challenges and how they were solved

| Challenge | Solution |
|---|---|
| Three linked datasets, with round trips duplicating `travelCode` in flights | Explicit join keys. `outbound_legs()` isolates the leg that reflects the user's choice. |
| A "perfect" price model that could hide poor generalisation | Added a time-based split and an unseen-route test. Serve Ridge for unseen routes. |
| Sparse one-hot output crashed HistGradientBoosting on the full data | Dense one-hot encoding (`sparse_output=False`). Found by running the real data end to end. |
| Recommender metrics inflated by tied zero scores | Random tie-breaking in ranking evaluation. The inflated hit@3 of 0.45 corrected to an honest 0.375. |
| MLflow 3 default (skops) serializer rejects tree internals | Logged with cloudpickle for these trusted, self-produced models. |
| The API container crashed on start: pandas 3 needs `pyarrow`, which locally only arrived through MLflow | Separate serving requirements with `pyarrow` pinned. A clean container build caught what the dev environment hid. |
| Port 5000 is taken by the macOS AirPlay receiver | Moved the API to port 8000 everywhere. |
| Parallel Airflow trainers writing one metrics file | One metrics file per model, so there is no write race. |
| Keeping Airflow's environment separate from the ML stack | Training runs as subprocesses in the project's virtualenv. |

## 6. Reliability and reproducibility

* **Version control:**
  * code is versioned in git;
  * models in the MLflow registry;
  * images by an immutable tag;
  * data snapshots are archived by the ingest task.
* **Repeatability:** a fixed random seed and pinned dependencies. `make train` rebuilds every artifact from the raw CSVs.
* **Maintainability:**
  * a single `config.py` with env-var overrides;
  * shared `src/` code used by training, the API, the UI and Airflow;
  * flake8 clean;
  * per-component READMEs.
* **Safety nets:**
  * quality gates (R² ≥ 0.90, F1 ≥ 0.80);
  * data validation;
  * a container smoke test;
  * Kubernetes probes;
  * automatic rollback.

## 7. Learnings

* **Training a model is the easy 20%.** Most of the work was in validation, packaging, automation and monitoring.
* **Evaluation design matters more than the algorithm.** The same model scores R² 1.0 or 0.70 depending
  on whether the test asks the right question.
* **Look at the data before trusting a metric.** Deterministic prices, a "none" label meaning "not disclosed",
  and uniform destinations all changed the design.
* **Simple, honest baselines** (random, popularity) are what make a recommender's numbers meaningful.

## 8. Future improvements

* **Monitoring:** Prometheus and Grafana dashboards on `/metrics`, alerting on the PSI threshold, and logging predictions for
  delayed-label evaluation.
* **Retraining:** trigger the DAG on drift or new data instead of a fixed schedule. Shadow-deploy new models and A/B test them before promotion.
* **Security:** API keys or OAuth on the API, rate limiting, TLS at the ingress, and secrets in a vault.
* **Better recommendations:** implicit feedback (clicks), contextual features (trip purpose, season), and learning-to-rank
  once real interaction data exists.
* **Cloud:** a managed Kubernetes service (EKS/GKE/AKS), an MLflow server with S3/GCS artifacts and a Postgres backend, managed
  Airflow (MWAA/Composer), and blue-green or canary releases.
