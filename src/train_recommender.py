"""Tune, evaluate and save the hotel recommender.

Temporal protocol (per user, ordered by date, users with >= 5 stays):
  stays  0-60%  -> fit for tuning      stays 60-80%  -> validation (pick weights)
  stays  0-80%  -> fit for testing     stays 80-100% -> test (reported numbers)
Then the final model is fit on all stays.

Metrics: hit@1, hit@3 (is the hotel the user actually booked next in the top-k?),
MRR and NDCG@3. Baselines: random ranking and global popularity.

    python -m src.train_recommender
"""
import itertools

import joblib
import mlflow
import numpy as np
import pandas as pd

from src import plots, tracking
from src.config import RANDOM_STATE, RECOMMENDER_PATH
from src.data import load_flights, load_hotels, load_users
from src.recommender import HotelRecommender

GRID = {
    "history": [0.0, 0.5, 1.0],
    "similar": [0.0, 0.5],
    "popularity": [0.0, 0.5, 1.0],
    "budget": [0.0, 0.5],
    "home_penalty": [0.0, 1.0, 3.0],
}


def temporal_split(hotels, lo, hi, min_stays=5):
    h = hotels.sort_values(["date", "travelCode"]).copy()
    h["pos"] = h.groupby("userCode").cumcount() / h.groupby("userCode")["userCode"].transform("size")
    h = h[h.groupby("userCode")["userCode"].transform("size") >= min_stays]
    return h[h["pos"] < lo], h[(h["pos"] >= lo) & (h["pos"] < hi)]


def rank_metrics(scores, rec, held_out):
    """scores: users x items matrix aligned with rec.user_index / rec.items."""
    item_idx = {hid: j for j, hid in enumerate(rec.items)}
    rows = held_out["userCode"].map(rec.user_index)
    cols = (held_out["name"] + " | " + held_out["place"]).map(item_idx)
    ok = rows.notna() & cols.notna()
    rows, cols = rows[ok].astype(int).to_numpy(), cols[ok].astype(int).to_numpy()
    # Break ties randomly; otherwise sparse signals (many zeros) would get credit for ties.
    jitter = np.random.default_rng(RANDOM_STATE).random(scores.shape) * 1e-9
    s = scores + jitter
    target = s[rows, cols]
    ranks = 1 + (s[rows] > target[:, None]).sum(axis=1)
    return {"hit_at_1": float(np.mean(ranks == 1)), "hit_at_3": float(np.mean(ranks <= 3)),
            "mrr": float(np.mean(1 / ranks)),
            "ndcg_at_3": float(np.mean(np.where(ranks <= 3, 1 / np.log2(ranks + 1), 0))),
            "n_eval": int(len(ranks))}


def flights_before(flights, hotels_subset):
    """Only use flights from trips that are part of the fitting window (no leakage)."""
    return flights[flights["travelCode"].isin(hotels_subset["travelCode"]) |
                   (flights["date"] <= hotels_subset["date"].max())]


def main():
    tracking.setup("hotel-recommender")
    users, flights, hotels = load_users(), load_flights(), load_hotels()
    rng = np.random.default_rng(RANDOM_STATE)

    # 1) tune on validation window
    fit_v, val = temporal_split(hotels, 0.6, 0.8)
    rec_v = HotelRecommender().fit(fit_v, users, flights[flights["travelCode"].isin(fit_v["travelCode"])])
    best, best_w, tried = -1, None, []
    for combo in itertools.product(*GRID.values()):
        w = dict(zip(GRID, combo))
        if not any(w[k] for k in ("history", "similar", "popularity", "budget")):
            continue
        m = rank_metrics(rec_v.score_matrix(w), rec_v, val)
        tried.append({**w, **m})
        if m["mrr"] > best:
            best, best_w = m["mrr"], w
    print(f"grid search: {len(tried)} weight sets, best validation MRR {best:.4f} with {best_w}")

    # 2) report on the untouched test window
    fit_t, test = temporal_split(hotels, 0.8, 1.01)
    rec_t = HotelRecommender(best_w).fit(fit_t, users, flights[flights["travelCode"].isin(fit_t["travelCode"])])
    n_users, n_items = len(rec_t.user_index), len(rec_t.items)
    results = {
        "random": rank_metrics(rng.random((n_users, n_items)), rec_t, test),
        "popularity": rank_metrics(np.tile(rec_t.popularity, (n_users, 1)), rec_t, test),
        "history_only": rank_metrics(rec_t.score_matrix({"history": 1, "similar": 0, "popularity": 0,
                                                         "budget": 0, "home_penalty": 0}), rec_t, test),
        "hybrid_tuned": rank_metrics(rec_t.score_matrix(), rec_t, test),
    }
    for name, m in results.items():
        print(f"{name:14s} hit@1={m['hit_at_1']:.3f} hit@3={m['hit_at_3']:.3f} "
              f"MRR={m['mrr']:.3f} NDCG@3={m['ndcg_at_3']:.3f}  (n={m['n_eval']})")

    fig = plots.grouped_bars(list(results), {
        "hit@1": [r["hit_at_1"] for r in results.values()],
        "hit@3": [r["hit_at_3"] for r in results.values()],
        "MRR": [r["mrr"] for r in results.values()],
    }, "Hotel recommender vs baselines (temporal test window)", "score", "recommender_vs_baselines.png")

    with mlflow.start_run(run_name="hybrid_tuned"):
        mlflow.log_params({f"w_{k}": v for k, v in best_w.items()})
        mlflow.log_params({"split": "per-user temporal 60/20/20", "grid_size": len(tried)})
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items() if k != "n_eval"})
        mlflow.log_table(pd.DataFrame(tried), "grid_search.json")
        mlflow.log_artifact(str(fig), "figures")
        rec = HotelRecommender(best_w).fit(hotels, users, flights)
        joblib.dump(rec, RECOMMENDER_PATH)
        mlflow.log_artifact(str(RECOMMENDER_PATH))

    lift = results["hybrid_tuned"]["hit_at_3"] - results["random"]["hit_at_3"]
    tracking.write_metrics("recommender", {
        "weights": best_w, "validation_mrr": best, "test": results,
        "hit_at_3": results["hybrid_tuned"]["hit_at_3"], "lift_hit_at_3_vs_random": lift,
        "n_hotels": n_items,
    })
    print(f"saved {RECOMMENDER_PATH}")


if __name__ == "__main__":
    main()
