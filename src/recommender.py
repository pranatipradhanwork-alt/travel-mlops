"""Objective 9: hybrid hotel recommender.

For a user u and hotel h the score is a weighted sum of signals, each scaled to [0, 1]:
  history    share of u's past stays that were at h (repeat-visit affinity)
  similar    item-item cosine similarity between h and the hotels u booked (collaborative)
  popularity h's share of all bookings (also the cold-start fallback)
  budget     how close h's nightly price is to u's typical nightly spend
minus a penalty for h being in u's home city (the city u usually flies out of) -
people rarely book a hotel in their own city.

Weights are tuned by grid search on a temporal validation split (src/train_recommender.py).
"""
import numpy as np
import pandas as pd

from src.data import outbound_legs

DEFAULT_WEIGHTS = {"history": 0.4, "similar": 0.2, "popularity": 0.2, "budget": 0.2, "home_penalty": 1.0}
SIGNALS = ["history", "similar", "popularity", "budget"]
REASONS = {
    "history": "You've enjoyed staying here before",
    "similar": "Travellers with similar bookings chose this",
    "popularity": "One of the most booked hotels",
    "budget": "Matches your usual nightly budget",
}


def _scale_rows(m):
    lo, hi = m.min(axis=1, keepdims=True), m.max(axis=1, keepdims=True)
    return np.divide(m - lo, hi - lo, out=np.zeros_like(m, dtype=float), where=(hi - lo) > 0)


class HotelRecommender:
    def __init__(self, weights=None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def fit(self, hotels, users, flights=None):
        h = hotels.copy()
        h["hotel_id"] = h["name"] + " | " + h["place"]
        self.catalog = (h.groupby("hotel_id")
                         .agg(name=("name", "first"), place=("place", "first"), avg_price=("price", "mean"),
                              bookings=("travelCode", "count"), avg_days=("days", "mean"))
                         .reset_index())
        self.items = self.catalog["hotel_id"].tolist()
        prices = self.catalog["avg_price"].to_numpy()

        counts = pd.crosstab(h["userCode"], h["hotel_id"]).reindex(columns=self.items, fill_value=0)
        self.user_items = counts
        self.user_index = {u: i for i, u in enumerate(counts.index)}
        m = counts.to_numpy(dtype=float)
        norms = np.linalg.norm(m, axis=0)
        norms[norms == 0] = 1
        item_sim = (m.T @ m) / np.outer(norms, norms)
        np.fill_diagonal(item_sim, 0)

        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        self.popularity = m.sum(axis=0) / max(m.sum(), 1)
        user_spend = (h.groupby("userCode")["price"].mean().reindex(counts.index)
                       .fillna(prices.mean()).to_numpy())
        budget = np.exp(-np.abs(prices[None, :] - user_spend[:, None]) / user_spend[:, None])

        home = np.zeros_like(m)
        if flights is not None and not flights.empty:
            out = outbound_legs(flights)
            origin = pd.crosstab(out["userCode"], out["from"]).reindex(index=counts.index, fill_value=0)
            origin = origin.div(origin.sum(axis=1).replace(0, 1), axis=0)
            home = np.column_stack([origin[p].to_numpy() if p in origin else np.zeros(len(counts))
                                    for p in self.catalog["place"]])
        self.signals = {
            "history": _scale_rows(share),
            "similar": _scale_rows(share @ item_sim),
            "popularity": _scale_rows(np.tile(self.popularity, (len(m), 1))),
            "budget": _scale_rows(budget),
        }
        self.home = home

        u = users.rename(columns={"code": "userCode"}).set_index("userCode")
        self.users = u
        return self

    # ------------------------------------------------------------------ scoring
    def score_matrix(self, weights=None):
        """Scores for every known user x hotel (used for fast offline evaluation)."""
        w = {**self.weights, **(weights or {})}
        s = sum(w[k] * self.signals[k] for k in SIGNALS)
        return s - w["home_penalty"] * self.home

    def _user_scores(self, user_code):
        i = self.user_index.get(user_code)
        if i is None:  # cold start: popularity only
            pop = self.popularity / self.popularity.max()
            return pop, None
        contrib = {k: self.weights[k] * self.signals[k][i] for k in SIGNALS}
        score = sum(contrib.values()) - self.weights["home_penalty"] * self.home[i]
        return score, contrib

    def recommend(self, user_code=None, place=None, max_price=None, k=5, exclude_booked=False):
        score, contrib = self._user_scores(user_code)
        out = self.catalog.assign(score=score)
        booked = (self.user_items.loc[user_code].to_numpy() if user_code in self.user_index
                  else np.zeros(len(self.items)))
        out["times_booked"] = booked
        if contrib is not None:
            top_signal = np.array(SIGNALS)[np.argmax(np.vstack([contrib[k] for k in SIGNALS]), axis=0)]
            out["reason"] = [REASONS[s] for s in top_signal]
            home_idx = self.home[self.user_index[user_code]] > 0.5
            out.loc[home_idx, "reason"] = "Your home city - ranked lower"
        else:
            out["reason"] = "Popular with all travellers"
        if place:
            out = out[out["place"].str.lower() == str(place).lower()]
        if max_price is not None:
            out = out[out["avg_price"] <= float(max_price)]
        if exclude_booked:
            out = out[out["times_booked"] == 0]
        out = out.sort_values("score", ascending=False).head(k)
        return (out[["hotel_id", "name", "place", "avg_price", "avg_days", "score", "times_booked", "reason"]]
                .round({"avg_price": 2, "avg_days": 2, "score": 4}).astype({"times_booked": int})
                .to_dict("records"))

    def user_profile(self, user_code):
        if user_code not in self.users.index:
            return None
        u = self.users.loc[user_code]
        hist = (self.user_items.loc[user_code] if user_code in self.user_index
                else pd.Series(0, index=self.items))
        return {
            "user_code": int(user_code), "name": u["name"], "gender": u["gender"], "age": int(u["age"]),
            "company": u["company"], "hotel_bookings": int(hist.sum()),
            "top_hotels": {k: int(v) for k, v in hist[hist > 0].sort_values(ascending=False).head(5).items()},
        }
