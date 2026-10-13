"""Unit tests for data utilities, the recommender and drift monitoring (no API)."""
import numpy as np
import pandas as pd

from src.data import first_name, outbound_legs, route_key, validate
from src.monitor import categorical_psi, numeric_psi
from src.recommender import HotelRecommender, _scale_rows


def test_dataset_passes_validation():
    counts = validate()
    assert counts["users"] > 0 and counts["flights"] > 0 and counts["hotels"] > 0


def test_helpers():
    assert first_name("  Maria  Silva") == "maria"
    assert route_key("B", "A") == route_key("A", "B") == "A | B"


def test_outbound_legs_keeps_first_leg_per_trip():
    f = pd.DataFrame({"travelCode": [1, 1, 2, 2], "from": ["X", "Y", "Z", "W"], "to": ["Y", "X", "W", "Z"],
                      "date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-02-01", "2020-02-03"])})
    out = outbound_legs(f)
    assert list(out["from"]) == ["X", "Z"]


def test_scale_rows_handles_constant_rows():
    m = _scale_rows(np.array([[1.0, 3.0], [2.0, 2.0]]))
    assert m.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def _toy():
    users = pd.DataFrame({"code": [0, 1], "company": ["A", "A"], "name": ["Ann X", "Bob Y"],
                          "gender": ["female", "male"], "age": [30, 40]})
    hotels = pd.DataFrame({
        "travelCode": [1, 2, 3, 4], "userCode": [0, 0, 1, 1],
        "name": ["H1", "H1", "H2", "H3"], "place": ["P1", "P1", "P2", "P3"],
        "days": [1, 2, 1, 1], "price": [100.0, 100.0, 300.0, 200.0], "total": [100.0, 200.0, 300.0, 200.0],
        "date": pd.to_datetime(["2020-01-01"] * 4)})
    flights = pd.DataFrame({"travelCode": [1, 2, 3, 4], "userCode": [0, 0, 1, 1],
                            "from": ["P3", "P3", "P1", "P1"], "to": ["P1", "P1", "P2", "P3"],
                            "date": pd.to_datetime(["2020-01-01"] * 4)})
    return users, hotels, flights


def test_recommender_home_penalty_and_filters():
    users, hotels, flights = _toy()
    rec = HotelRecommender({"history": 1, "similar": 0, "popularity": 0, "budget": 0,
                            "home_penalty": 5}).fit(hotels, users, flights)
    ranked = [r["place"] for r in rec.recommend(0, k=3)]
    assert ranked[0] == "P1"            # most booked by user 0
    assert ranked[-1] == "P3"           # user 0 flies out of P3 -> penalised
    assert all(r["avg_price"] <= 150 for r in rec.recommend(1, max_price=150))
    assert all(r["times_booked"] == 0 for r in rec.recommend(0, exclude_booked=True))


def test_recommender_cold_start_uses_popularity():
    users, hotels, flights = _toy()
    rec = HotelRecommender().fit(hotels, users, flights)
    recs = rec.recommend(12345, k=1)
    assert recs[0]["place"] == "P1" and rec.user_profile(12345) is None


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert numeric_psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert numeric_psi(a, rng.normal(1.5, 1, 5000)) > 0.2
    assert categorical_psi(pd.Series(list("aabb") * 100), pd.Series(list("aaab") * 100)) > 0.05
