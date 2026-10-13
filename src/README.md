# `src/` — data, training, tracking, monitoring

| Module | Role |
|---|---|
| `config.py` | Every path, the MLflow URI and the quality-gate thresholds. All can be overridden with env vars. |
| `data.py` | Loading, validation, feature lists, helpers (`outbound_legs`, `route_key`, `first_name`). |
| `train_flight_price.py` | Regression: 4 candidate runs, time-based and unseen-route evaluation, permutation importance, champion + Ridge fallback. |
| `train_gender.py` | Classification: 3 experiment runs, stratified and unseen-name CV, confusion matrices. |
| `recommender.py` | `HotelRecommender`: hybrid scoring (history, collaborative similarity, popularity, budget, home-city penalty). |
| `train_recommender.py` | Temporal 60/20/20 split, 105-combination weight grid search, baselines, NDCG/MRR/hit@k. |
| `tracking.py` | MLflow helpers: experiment setup, log and register model, promote the `@champion` alias, write metrics. |
| `monitor.py` | PSI data-drift report. |
| `plots.py` | Evaluation figures for `reports/figures/`, also logged to MLflow. |
| `pipeline.py` | Validate → drift → train all models (`make train`). |
| `validate_cli.py` | Data-quality gate for Airflow and Jenkins (non-zero exit on failure). |

## MLflow usage

```bash
make train        # logs runs to sqlite:///mlflow.db (artifacts in ./mlruns)
make mlflow-ui    # http://localhost:5001
```

* **Experiments:** `travel-flight-price`, `travel-gender`, `travel-hotel-recommender`.
* **Per run:** parameters (model type, hyper-parameters, features, split), metrics, figures,
  classification reports, grid-search table and the model artifact with its input signature.
* **Model Registry:** `flight-price-regressor` and `gender-classifier`. Each training run creates a new
  version. A version that passes its quality gate gets the alias **`@champion`**:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/flight-price-regressor@champion")
```

* **Roll back:** point the alias at an earlier version (MLflow UI → Models → Aliases, or
  `MlflowClient().set_registered_model_alias(name, "champion", "3")`).
* **Shared tracking server:** set `MLFLOW_TRACKING_URI=http://mlflow:5000`. No code changes are needed.
