"""Data-drift monitoring with the Population Stability Index (PSI).

Compares a reference window (data the models were trained on) with a current
window (newly arrived data). Rule of thumb: PSI < 0.1 stable, 0.1-0.2 moderate,
> 0.2 significant drift -> investigate and retrain.

    python -m src.monitor                       # last 90 days vs everything before
    python -m src.monitor --current data/landing/flights.csv
Writes reports/drift_report.json and exits 0 (use --fail-on-drift for CI).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.config import DRIFT_REPORT_PATH, PSI_ALERT
from src.data import load_flights

NUMERIC = ["price", "distance"]
CATEGORICAL = ["flightType", "agency", "from", "to"]


def psi(expected, actual, eps=1e-4):
    e, a = np.asarray(expected, float) + eps, np.asarray(actual, float) + eps
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def categorical_psi(ref, cur):
    cats = sorted(set(ref) | set(cur))
    r, c = ref.value_counts(), cur.value_counts()
    return psi([r.get(k, 0) for k in cats], [c.get(k, 0) for k in cats])


def drift_report(reference, current):
    scores = {c: numeric_psi(reference[c], current[c]) for c in NUMERIC}
    scores.update({c: categorical_psi(reference[c], current[c]) for c in CATEGORICAL})
    drifted = {k: round(v, 4) for k, v in scores.items() if v > PSI_ALERT}
    return {"psi": {k: round(v, 4) for k, v in scores.items()}, "threshold": PSI_ALERT,
            "drifted_features": drifted, "drift_detected": bool(drifted),
            "reference_rows": len(reference), "current_rows": len(current),
            "mean_price_reference": round(float(reference["price"].mean()), 2),
            "mean_price_current": round(float(current["price"].mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", help="CSV of new flights; default = last --days of flights.csv")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--fail-on-drift", action="store_true")
    args = ap.parse_args()

    flights = load_flights()
    if args.current:
        reference, current = flights, pd.read_csv(args.current)
    else:
        cutoff = flights["date"].max() - pd.Timedelta(days=args.days)
        reference, current = flights[flights["date"] <= cutoff], flights[flights["date"] > cutoff]
    report = drift_report(reference, current)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["drift_detected"] and args.fail_on_drift:
        sys.exit(2)


if __name__ == "__main__":
    main()
