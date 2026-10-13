# Flight Price API (Flask)

A REST API that serves the flight price, gender and hotel recommendation models.
Models load once at startup from `models/*.joblib`. Under gunicorn, `preload_app` shares them
across all worker processes.

## Run

```bash
make api                                            # dev server  → http://localhost:8000
gunicorn -c api/gunicorn.conf.py api.app:app        # production server (used in Docker)
```

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | 8000 | Bind port |
| `GUNICORN_WORKERS` / `GUNICORN_THREADS` | 2 / 4 | Concurrency per pod |
| `MODEL_DIR` | `./models` | Where the `.joblib` files are loaded from |

## Endpoints

### `POST /predict` — flight price

| Field | Required | Example |
|---|---|---|
| `from`, `to` | ✅ | `"Recife (PE)"`, `"Florianopolis (SC)"` |
| `flightType` | ✅ | `economic` · `premium` · `firstClass` |
| `agency` | ✅ | `CloudFy` · `FlyingDrops` · `Rainbow` |
| `distance` | only for unseen routes | `676.53` |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops"}'
```
```json
{"model_version": 1,
 "predictions": {"known_route": true, "model": "random_forest", "predicted_price": 1434.38}}
```
* **Known route:** the distance is looked up and the Random Forest champion is used.
* **Unseen route:** `distance` is required and the Ridge fallback is used (`"model": "ridge_fallback"`).
  Ridge generalises better to new routes. See `reports/MODEL_EVALUATION.md`.
* **Batch:** send a JSON list to get a list of predictions back.

### `POST /predict/gender`
```bash
curl -X POST localhost:8000/predict/gender -H 'Content-Type: application/json' -d '{"name":"Patricia Carson"}'
```
```json
{"gender": "female", "confidence": 0.92, "probabilities": {"female": 0.92, "male": 0.08}, "note": "..."}
```

### `GET /recommend/<user_code>`
Query parameters: `k` (1–20, default 5), `max_price` (nightly), `place`, `exclude_booked=true`.
```bash
curl "localhost:8000/recommend/0?k=3&max_price=250"
```
Returns the user's profile plus ranked hotels with a `reason`. Unknown users get popularity-based suggestions (cold start).

### Operations
| Endpoint | Use |
|---|---|
| `GET /health` | Liveness/readiness probe. Returns 503 if the flight model failed to load. |
| `GET /metadata` | Valid categories and number of known routes. Useful for building forms. |
| `GET /metrics` | Prometheus text format: requests and latency per endpoint, known vs fallback predictions, mean predicted price. |

## Errors
Every validation error returns **400** with `{"error": "..."}`. Covered cases: missing field, `from == to`,
non-numeric or negative distance, unknown route without a distance, non-JSON body, empty list.
If a model is missing, the endpoint returns **503**.

## Tests
```bash
pytest tests/test_api.py -q
```
