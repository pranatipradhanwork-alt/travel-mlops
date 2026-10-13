"""Objective 2: Flask REST API serving all three models.

Endpoints
  GET  /health                 liveness/readiness + loaded model versions
  GET  /metadata               valid values for each flight input field
  GET  /metrics                Prometheus-format request/latency/prediction counters
  POST /predict                flight price (single object or list of objects)
  POST /predict/gender         gender from a user's name
  GET  /recommend/<user_code>  hotel suggestions (?place=&max_price=&k=&exclude_booked=)

Run locally:  python -m api.app      Production: gunicorn -c api/gunicorn.conf.py api.app:app
"""
import logging
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FLIGHT_MODEL_PATH, GENDER_MODEL_PATH, RECOMMENDER_PATH  # noqa: E402
from src.data import first_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("travel-api")

app = Flask(__name__)
MODELS = {}
REQUIRED_FLIGHT_FIELDS = ["from", "to", "flightType", "agency"]

# --------------------------------------------------------------------- monitoring
# Kept in-process to avoid extra dependencies; scrape /metrics with Prometheus.
_lock = threading.Lock()
REQUESTS = Counter()          # (endpoint, status) -> count
LATENCY_SUM = Counter()       # endpoint -> seconds
PREDICTIONS = Counter()       # label -> count (e.g. known vs fallback route)
PRICE_SUM = [0.0, 0]          # running sum/count of predicted prices, to spot output drift


@app.before_request
def _start_timer():
    g.t0 = time.perf_counter()


@app.after_request
def _record(response):
    endpoint = request.url_rule.rule if request.url_rule else "unknown"
    with _lock:
        REQUESTS[(endpoint, response.status_code)] += 1
        LATENCY_SUM[endpoint] += time.perf_counter() - g.get("t0", time.perf_counter())
    return response


def load_models():
    for key, path in [("flight", FLIGHT_MODEL_PATH), ("gender", GENDER_MODEL_PATH),
                      ("recommender", RECOMMENDER_PATH)]:
        if Path(path).exists():
            MODELS[key] = joblib.load(path)
            log.info("loaded %s model from %s", key, path)
        else:
            log.warning("%s model missing at %s", key, path)


load_models()


def bad_request(msg, status=400):
    return jsonify({"error": msg}), status


def flight_categories():
    pre = MODELS["flight"]["pipeline"].named_steps["pre"]
    enc = pre.named_transformers_["cat"]
    return {col: sorted(map(str, cats)) for col, cats in zip(enc.feature_names_in_, enc.categories_)}


# --------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    status = {k: k in MODELS for k in ("flight", "gender", "recommender")}
    ok = status["flight"]
    return jsonify({
        "status": "ok" if ok else "degraded", "models": status,
        "flight_model": {k: MODELS["flight"].get(k) for k in ("model_type", "mlflow_version", "metrics")}
        if ok else None,
    }), 200 if ok else 503


@app.get("/metadata")
def metadata():
    if "flight" not in MODELS:
        return bad_request("flight model not loaded", 503)
    return jsonify({"categories": flight_categories(), "required_fields": REQUIRED_FLIGHT_FIELDS,
                    "optional_fields": ["distance"], "known_routes": len(MODELS["flight"]["routes"])})


@app.get("/metrics")
def metrics():
    lines = ["# TYPE travel_api_requests_total counter"]
    with _lock:
        for (ep, code), n in sorted(REQUESTS.items()):
            lines.append(f'travel_api_requests_total{{endpoint="{ep}",status="{code}"}} {n}')
        lines.append("# TYPE travel_api_latency_seconds_sum counter")
        for ep, s in sorted(LATENCY_SUM.items()):
            lines.append(f'travel_api_latency_seconds_sum{{endpoint="{ep}"}} {s:.6f}')
        lines.append("# TYPE travel_api_predictions_total counter")
        for label, n in sorted(PREDICTIONS.items()):
            lines.append(f'travel_api_predictions_total{{kind="{label}"}} {n}')
        mean = PRICE_SUM[0] / PRICE_SUM[1] if PRICE_SUM[1] else 0
        lines.append("# TYPE travel_api_predicted_price_mean gauge")
        lines.append(f"travel_api_predicted_price_mean {mean:.2f}")
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@app.post("/predict")
def predict_price():
    """Body: {"from", "to", "flightType", "agency", ["distance"]} or a list of them.

    If the route was seen in training, the champion model is used and a missing
    distance is looked up. Unseen routes need a distance and use the Ridge fallback.
    """
    if "flight" not in MODELS:
        return bad_request("flight model not loaded", 503)
    payload = request.get_json(silent=True)
    if payload is None:
        return bad_request("body must be JSON")
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        return bad_request("empty request")
    m = MODELS["flight"]
    rows, known = [], []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            return bad_request(f"record {i}: must be an object")
        missing = [f for f in REQUIRED_FLIGHT_FIELDS if not r.get(f)]
        if missing:
            return bad_request(f"record {i}: missing fields {missing}")
        if r["from"] == r["to"]:
            return bad_request(f"record {i}: 'from' and 'to' must differ")
        route_distance = m["routes"].get(f"{r['from']}|{r['to']}")
        distance = r.get("distance", route_distance)
        if distance is None:
            return bad_request(f"record {i}: unknown route, please provide 'distance' (km)")
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return bad_request(f"record {i}: 'distance' must be a number")
        if distance <= 0:
            return bad_request(f"record {i}: 'distance' must be positive")
        rows.append({**{f: r[f] for f in REQUIRED_FLIGHT_FIELDS}, "distance": distance})
        known.append(route_distance is not None)

    df = pd.DataFrame(rows)
    known_s = pd.Series(known)
    preds = pd.Series(0.0, index=df.index)
    if known_s.any():
        preds[known_s] = m["pipeline"].predict(df.loc[known_s, m["features"]])
    if (~known_s).any():
        preds[~known_s] = m["fallback"].predict(df.loc[~known_s, m["fallback_features"]])

    results = [{"predicted_price": round(float(p), 2), "known_route": bool(k),
                "model": m["model_type"] if k else "ridge_fallback"} for p, k in zip(preds, known)]
    with _lock:
        PREDICTIONS["known_route"] += int(known_s.sum())
        PREDICTIONS["fallback_route"] += int((~known_s).sum())
        PRICE_SUM[0] += float(preds.sum())
        PRICE_SUM[1] += len(preds)
    body = results if isinstance(payload, list) else results[0]
    return jsonify({"predictions": body, "model_version": m.get("mlflow_version")})


@app.post("/predict/gender")
def predict_gender():
    """Body: {"name": "Maria Silva"} -> male/female with class probabilities."""
    if "gender" not in MODELS:
        return bad_request("gender model not loaded", 503)
    body = request.get_json(silent=True) or {}
    if not str(body.get("name", "")).strip():
        return bad_request("field 'name' is required")
    m = MODELS["gender"]
    X = pd.DataFrame([{"first_name": first_name(body["name"])}])
    proba = m["pipeline"].predict_proba(X)[0]
    best = int(proba.argmax())
    with _lock:
        PREDICTIONS[f"gender_{m['classes'][best]}"] += 1
    return jsonify({"gender": m["classes"][best], "confidence": round(float(proba[best]), 4),
                    "probabilities": {c: round(float(p), 4) for c, p in zip(m["classes"], proba)},
                    "note": "Predicted from first name only; users may choose not to disclose gender."})


@app.get("/recommend/<int:user_code>")
def recommend(user_code):
    if "recommender" not in MODELS:
        return bad_request("recommender not loaded", 503)
    k = request.args.get("k", 5, type=int)
    max_price = request.args.get("max_price", type=float)
    exclude = request.args.get("exclude_booked", "false").lower() in ("1", "true", "yes")
    rec = MODELS["recommender"]
    recs = rec.recommend(user_code, place=request.args.get("place"), max_price=max_price,
                         k=max(1, min(k, 20)), exclude_booked=exclude)
    with _lock:
        PREDICTIONS["recommend_known_user" if user_code in rec.user_index else "recommend_cold_start"] += 1
    return jsonify({"user": rec.user_profile(user_code), "recommendations": recs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
