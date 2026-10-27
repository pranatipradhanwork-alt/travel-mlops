"""API and model tests. Run after training: pytest -q"""
import pytest

from api.app import MODELS, app

SAMPLE = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass",
          "agency": "FlyingDrops"}


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_models_loaded():
    assert {"flight", "gender", "recommender"} <= MODELS.keys()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_metadata(client):
    body = client.get("/metadata").json
    assert "firstClass" in body["categories"]["flightType"]
    assert body["known_routes"] > 0


def test_predict_known_route_matches_history(client):
    """The dataset has a single price per (route, class, agency); known routes should reproduce it."""
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    pred = r.json["predictions"]
    assert pred["known_route"] is True
    assert pred["predicted_price"] == pytest.approx(1434.38, abs=5)  # first row of flights.csv


def test_predict_batch_and_class_ordering(client):
    batch = [dict(SAMPLE, flightType=t) for t in ("economic", "premium", "firstClass")]
    prices = [p["predicted_price"] for p in client.post("/predict", json=batch).json["predictions"]]
    assert prices[0] < prices[1] < prices[2], "first class should cost more than economy"


def test_predict_unseen_route_uses_fallback(client):
    r = client.post("/predict", json=dict(SAMPLE, to="Manaus (AM)", distance=2800))
    assert r.status_code == 200
    pred = r.json["predictions"]
    assert pred["known_route"] is False and pred["model"] == "ridge_fallback"
    assert pred["predicted_price"] > 0


def test_predict_unseen_route_requires_distance(client):
    r = client.post("/predict", json=dict(SAMPLE, to="Manaus (AM)"))
    assert r.status_code == 400 and "distance" in r.json["error"]


@pytest.mark.parametrize("bad, msg", [
    ({k: v for k, v in SAMPLE.items() if k != "agency"}, "agency"),
    (dict(SAMPLE, to=SAMPLE["from"]), "differ"),
    (dict(SAMPLE, distance="far"), "number"),
    (dict(SAMPLE, distance=-5), "positive"),
])
def test_predict_validation(client, bad, msg):
    r = client.post("/predict", json=bad)
    assert r.status_code == 400 and msg in r.json["error"]


def test_predict_bad_json(client):
    assert client.post("/predict", data="nope", content_type="text/plain").status_code == 400
    assert client.post("/predict", json=[]).status_code == 400


def test_gender(client):
    r = client.post("/predict/gender", json={"name": "Patricia Carson"})
    assert r.status_code == 200 and r.json["gender"] == "female"
    assert client.post("/predict/gender", json={"name": "Joseph Holsten"}).json["gender"] == "male"
    assert client.post("/predict/gender", json={}).status_code == 400


@pytest.mark.parametrize("name, gender", [("Patricia", "female"), ("Priya Sharma", "female"), ("Rahul", "male")])
def test_gender_names_from_many_countries(client, name, gender):
    r = client.post("/predict/gender", json={"name": name})
    assert r.status_code == 200 and r.json["gender"] == gender


@pytest.mark.parametrize("name", ["abc", "abcabc abcabc", "axhy", "qzvk"])
def test_gender_non_names_are_uncertain(client, name):
    # the instructor's examples, plus random letters the model is confidently wrong on
    r = client.post("/predict/gender", json={"name": name})
    assert r.status_code == 200 and r.json["gender"] == "uncertain"


@pytest.mark.parametrize("name", ["123", "j@ne", "A"])
def test_gender_rejects_bad_input(client, name):
    assert client.post("/predict/gender", json={"name": name}).status_code == 400


def test_gender_allows_two_letter_names(client):
    assert client.post("/predict/gender", json={"name": "Al"}).status_code == 200


def test_recommend_known_user(client):
    r = client.get("/recommend/0?k=3")
    assert r.status_code == 200 and len(r.json["recommendations"]) == 3
    assert r.json["user"]["user_code"] == 0


def test_recommend_filters(client):
    recs = client.get("/recommend/1?max_price=250&exclude_booked=true").json["recommendations"]
    assert all(x["avg_price"] <= 250 and x["times_booked"] == 0 for x in recs)


def test_recommend_cold_start(client):
    r = client.get("/recommend/999999")
    assert r.status_code == 200 and r.json["user"] is None and r.json["recommendations"]


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").get_data(as_text=True)
    assert 'travel_api_requests_total{endpoint="/predict",status="200"}' in text
    assert "travel_api_predictions_total" in text


def test_index_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200 and r.mimetype == "text/html"
    assert b'href="/health"' in r.data and b"/predict/gender" in r.data
