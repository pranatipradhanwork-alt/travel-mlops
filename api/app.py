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
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FLIGHT_MODEL_PATH, GENDER_MODEL_PATH, RECOMMENDER_PATH  # noqa: E402
from src import gender_rules  # noqa: E402
from src.data import first_name  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("travel-api")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # behind Render's proxy: build https:// URLs
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
INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Travel MLOps API</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;line-height:1.5}
code,pre{background:#f3f3f3;border-radius:4px;padding:2px 4px}pre{padding:12px;overflow-x:auto}</style></head>
<body><h1>Travel MLOps API</h1>
<p>Flight price prediction, gender classification and hotel recommendations.</p>
<h2>Try it in the browser</h2>
<ul>
<li><a href="/health">/health</a> &ndash; service status and loaded models</li>
<li><a href="/metadata">/metadata</a> &ndash; cities, flight types and agencies the model knows</li>
<li><a href="/recommend/5?k=3">/recommend/5?k=3</a> &ndash; hotel recommendations for user 5</li>
<li><a href="/metrics">/metrics</a> &ndash; request and prediction counters (Prometheus format)</li>
</ul>
<h2>POST endpoints</h2>
<pre>curl -X POST {base}predict -H "Content-Type: application/json" \\
  -d '{"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass", "agency": "FlyingDrops"}'

curl -X POST {base}predict/gender -H "Content-Type: application/json" -d '{"name": "Maria Silva"}'</pre>
</body></html>"""


@app.get("/")
def index():
    return Response(INDEX_HTML.replace("{base}", request.host_url), mimetype="text/html")

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
    """Body: {"name": "Maria Silva"} -> male/female/uncertain with class probabilities.

    "uncertain" when the first name is not a known name or the model is not
    confident enough (rules in src/gender_rules.py).
    """
    if "gender" not in MODELS:
        return bad_request("gender model not loaded", 503)
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return bad_request("field 'name' is required")
    error = gender_rules.name_error(name)
    if error:
        return bad_request(error)

    m = MODELS["gender"]
    first = first_name(name)
    proba = m["pipeline"].predict_proba(pd.DataFrame([{"first_name": first}]))[0]
    probs = {c: round(float(p), 4) for c, p in zip(m["classes"], proba)}
    gender, note = gender_rules.decide(first, probs)
    with _lock:
        PREDICTIONS[f"gender_{gender}"] += 1
    return jsonify({"gender": gender, "confidence": max(probs.values()), "probabilities": probs, "note": note})


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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
