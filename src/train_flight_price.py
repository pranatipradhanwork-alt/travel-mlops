"""Objective 1 + 7: flight price regression, tracked and versioned in MLflow.

Evaluation protocol
  * Primary  - time-based holdout: train on the oldest 80% of flights, test on
               the newest 20%. This mirrors production (predicting future bookings).
  * Robustness - 5-fold GroupKFold over undirected routes: every test route is
               unseen in training. Shows how each model copes with a new route.
Candidates: Ridge, Random Forest, HistGradientBoosting, plus an ablation run that
adds `time` and calendar features to justify dropping them.

The champion (best time-holdout RMSE) is refit on all data and registered in
MLflow. Ridge is also saved as a fallback for routes never seen in training,
because it generalizes better to unseen routes than the tree models.

    python -m src.train_flight_price
"""
import sys
import time

import joblib
import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import plots, tracking
from src.config import FLIGHT_MODEL_NAME, FLIGHT_MODEL_PATH, MIN_FLIGHT_R2, RANDOM_STATE
from src.data import (DATE_FEATURES, FLIGHT_CATEGORICAL, FLIGHT_NUMERIC,
                      add_date_features, load_flights, route_key)

ROUTE_CV_SAMPLE = 60_000  # rows used for the (5x refit) unseen-route check


def build_pipeline(estimator, numeric=FLIGHT_NUMERIC):
    pre = ColumnTransformer([
        # Dense output: HistGradientBoosting does not accept sparse input.
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FLIGHT_CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


ABLATION_NUMERIC = FLIGHT_NUMERIC + ["time"] + DATE_FEATURES
CANDIDATES = {
    # name: (estimator, numeric features, eligible to be champion)
    "ridge": (Ridge(alpha=1.0), FLIGHT_NUMERIC, True),
    "random_forest": (RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                            random_state=RANDOM_STATE), FLIGHT_NUMERIC, True),
    "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                                             random_state=RANDOM_STATE), FLIGHT_NUMERIC, True),
    "hgb_with_time_and_date": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                                             random_state=RANDOM_STATE), ABLATION_NUMERIC, False),
}


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def unseen_route_cv(pipe, df, features):
    sample = df.sample(min(ROUTE_CV_SAMPLE, len(df)), random_state=RANDOM_STATE)
    scores = []
    for tr, te in GroupKFold(n_splits=5).split(sample, groups=sample["route"]):
        m = clone(pipe).fit(sample.iloc[tr][features], sample.iloc[tr]["price"])
        scores.append(evaluate(sample.iloc[te]["price"], m.predict(sample.iloc[te][features])))
    return {f"unseen_route_{k}": float(np.mean([s[k] for s in scores])) for k in ("rmse", "mae", "r2")}


def main():
    tracking.setup("flight-price")
    df = add_date_features(load_flights())
    df["route"] = [route_key(a, b) for a, b in zip(df["from"], df["to"])]
    cutoff = df["date"].quantile(0.8)
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    print(f"train {len(train):,} rows (< {cutoff.date()}), test {len(test):,} rows")

    results = {}
    for name, (est, numeric, eligible) in CANDIDATES.items():
        features = FLIGHT_CATEGORICAL + numeric
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(clone(est), numeric)
            t0 = time.time()
            pipe.fit(train[features], train["price"])
            fit_s = time.time() - t0
            pred = pipe.predict(test[features])
            metrics = {**evaluate(test["price"], pred), **unseen_route_cv(pipe, df, features),
                       "train_seconds": round(fit_s, 2)}
            mlflow.log_params({"model_type": name, "features": ",".join(features),
                               "split": "time-based 80/20", "n_train": len(train), "n_test": len(test),
                               **{k: v for k, v in est.get_params().items()
                                  if isinstance(v, (int, float, str, bool))}})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("eligible_champion", str(eligible))
            results[name] = {"run_id": run.info.run_id, "pipe": pipe, "pred": pred, "features": features,
                             "eligible": eligible, **metrics}
            print(f"{name:24s} time-holdout rmse={metrics['rmse']:7.2f} r2={metrics['r2']:.4f} | "
                  f"unseen-route r2={metrics['unseen_route_r2']:.3f} mae={metrics['unseen_route_mae']:.0f}")

    best_name = min((k for k in results if results[k]["eligible"]), key=lambda k: results[k]["rmse"])
    best = results[best_name]
    print(f"champion: {best_name}")

    # Figures for the report and MLflow.
    figs = [
        plots.predicted_vs_actual(test["price"], best["pred"], f"{best_name} - time holdout",
                                  "flight_pred_vs_actual.png"),
        plots.grouped_bars(list(results), {
            "time-holdout R²": [results[k]["r2"] for k in results],
            "unseen-route R²": [results[k]["unseen_route_r2"] for k in results],
        }, "Flight price models: known vs unseen routes", "R²", "flight_model_comparison.png"),
    ]
    sample = test.sample(min(5000, len(test)), random_state=RANDOM_STATE)
    imp = permutation_importance(best["pipe"], sample[best["features"]], sample["price"], n_repeats=5,
                                 random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
    importances = dict(zip(best["features"], imp.importances_mean.round(2)))
    figs.append(plots.bar(list(importances), list(importances.values()),
                          f"Permutation importance ({best_name})", "RMSE increase when shuffled",
                          "flight_feature_importance.png"))

    # Refit champion and fallback on all data for serving.
    champion = clone(best["pipe"]).fit(df[best["features"]], df["price"])
    fallback = clone(results["ridge"]["pipe"]).fit(df[results["ridge"]["features"]], df["price"])
    routes = df.groupby(["from", "to"])["distance"].first()
    route_table = {f"{a}|{b}": float(d) for (a, b), d in routes.items()}

    with mlflow.start_run(run_id=best["run_id"]):
        version = tracking.log_sklearn_model(champion, "model", FLIGHT_MODEL_NAME,
                                             input_example=test[best["features"]].head(3))
        mlflow.log_dict(importances, "permutation_importance.json")
        for f in figs:
            mlflow.log_artifact(str(f), "figures")
        mlflow.set_tag("selected", "true")

    passed = best["r2"] >= MIN_FLIGHT_R2
    if passed:
        tracking.promote(FLIGHT_MODEL_NAME, version)
        joblib.dump({"pipeline": champion, "features": best["features"], "model_type": best_name,
                     "fallback": fallback, "fallback_features": results["ridge"]["features"],
                     "routes": route_table, "mlflow_version": version,
                     "metrics": {k: best[k] for k in ("rmse", "mae", "r2", "mape", "unseen_route_r2")},
                     "train_stats": {"price_mean": float(df["price"].mean()),
                                     "distance": df["distance"].describe().to_dict()}},
                    FLIGHT_MODEL_PATH)
        print(f"saved {FLIGHT_MODEL_PATH} (registry version {version} -> champion)")
    keys = ("rmse", "mae", "r2", "mape", "unseen_route_rmse", "unseen_route_mae", "unseen_route_r2")
    tracking.write_metrics("flight_price", {
        "best_model": best_name, "registry_version": version, "passed_gate": passed,
        "split": f"time-based, test >= {cutoff.date()}", "permutation_importance": importances,
        "candidates": {k: {m: v[m] for m in keys} for k, v in results.items()},
    })
    if not passed:
        print(f"quality gate failed: r2={best['r2']:.4f} < {MIN_FLIGHT_R2}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
