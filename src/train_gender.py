"""Objective 8: classify a user's gender.

Findings on the data (all logged as MLflow runs so they are reproducible):
  * About a third of users are labelled "none". Their names are ordinary gendered
    names, so "none" means "not disclosed" and cannot be predicted from the data.
    A 3-class model is logged for comparison.
  * Age, company and travel behaviour carry no gender signal. A run that adds
    them is logged to show this.
  * The served model is a male/female classifier on character n-grams of the first
    name, trained on users who disclosed their gender.

Evaluation: stratified 5-fold CV, plus GroupKFold by first name ("unseen names"),
which shows how the model does on names it has never seen.

    python -m src.train_gender
"""
import sys

import joblib
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import plots, tracking
from src.config import GENDER_MODEL_NAME, GENDER_MODEL_PATH, MIN_GENDER_F1, RANDOM_STATE
from src.data import first_name, load_flights, load_users

BEHAVIOUR = ["age", "n_trips", "avg_flight_price", "first_class_share"]


def name_pipeline():
    pre = ColumnTransformer([("name", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), "first_name")])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=5.0, max_iter=3000))])


def name_plus_behaviour_pipeline():
    pre = ColumnTransformer([
        ("name", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), "first_name"),
        ("company", OneHotEncoder(handle_unknown="ignore"), ["company"]),
        ("num", StandardScaler(), BEHAVIOUR),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=5.0, max_iter=3000))])


def user_features(users, flights):
    df = users.copy()
    df["first_name"] = df["name"].map(first_name)
    fa = flights.groupby("userCode").agg(
        n_trips=("travelCode", "nunique"), avg_flight_price=("price", "mean"),
        first_class_share=("flightType", lambda s: (s == "firstClass").mean()))
    df = df.merge(fa, left_on="code", right_index=True, how="left")
    df[BEHAVIOUR] = df[BEHAVIOUR].fillna(df[BEHAVIOUR].median())
    return df


def cv_metrics(make, X, y, groups):
    strat = cross_val_predict(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    unseen = cross_val_predict(make(), X, y, cv=GroupKFold(5), groups=groups)
    m = {"accuracy": accuracy_score(y, strat), "f1_macro": f1_score(y, strat, average="macro"),
         "precision_macro": precision_score(y, strat, average="macro"),
         "recall_macro": recall_score(y, strat, average="macro"),
         "unseen_names_accuracy": accuracy_score(y, unseen),
         "unseen_names_f1_macro": f1_score(y, unseen, average="macro")}
    return {k: float(v) for k, v in m.items()}, strat


EXPERIMENTS = {
    # run name: (pipeline factory, label set, served?)
    "3class_name_only": (name_pipeline, ["female", "male", "none"], False),
    "binary_name_plus_behaviour": (name_plus_behaviour_pipeline, ["female", "male"], False),
    "binary_name_only": (name_pipeline, ["female", "male"], True),
}


def main():
    tracking.setup("gender")
    users = user_features(load_users(), load_flights())
    print(f"label distribution: {users['gender'].value_counts().to_dict()}")
    results = {}
    for name, (make, labels, served) in EXPERIMENTS.items():
        df = users[users["gender"].isin(labels)].reset_index(drop=True)
        X, y = df, df["gender"]
        with mlflow.start_run(run_name=name):
            metrics, pred = cv_metrics(make, X, y, df["first_name"])
            mlflow.log_params({"labels": ",".join(labels), "n_users": len(df), "model": "logistic_regression",
                               "features": "first_name char 2-4grams" + (" + behaviour" if "behaviour" in name else ""),
                               "cv": "stratified 5-fold + GroupKFold(first_name)"})
            mlflow.log_metrics(metrics)
            report = classification_report(y, pred, zero_division=0)
            mlflow.log_text(report, "classification_report.txt")
            fig = plots.confusion(y, pred, labels, name.replace("_", " "), f"gender_confusion_{name}.png")
            mlflow.log_artifact(str(fig), "figures")
            results[name] = {**metrics, "report": report}
            print(f"{name:28s} acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f} | "
                  f"unseen-names acc={metrics['unseen_names_accuracy']:.3f}")
            if served:
                model = make().fit(X, y)  # final model on all labelled users
                version = tracking.log_sklearn_model(model, "model", GENDER_MODEL_NAME,
                                                     input_example=X[["first_name"]].head(3))
                served_metrics = metrics
    print(results["binary_name_only"]["report"])

    passed = served_metrics["f1_macro"] >= MIN_GENDER_F1
    if passed:
        tracking.promote(GENDER_MODEL_NAME, version)
        joblib.dump({"pipeline": model, "features": ["first_name"], "classes": list(model.classes_),
                     "metrics": served_metrics, "mlflow_version": version}, GENDER_MODEL_PATH)
        print(f"saved {GENDER_MODEL_PATH}")
    tracking.write_metrics("gender", {
        "served_model": "binary_name_only", "registry_version": version, "passed_gate": passed,
        "label_distribution": users["gender"].value_counts().to_dict(),
        "experiments": {k: {m: round(v, 4) for m, v in r.items() if m != "report"} for k, r in results.items()},
        **{k: round(v, 4) for k, v in served_metrics.items()},
    })
    if not passed:
        print(f"quality gate failed: f1_macro={served_metrics['f1_macro']:.3f} < {MIN_GENDER_F1}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
