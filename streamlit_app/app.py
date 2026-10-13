"""Objective 9: Streamlit UI for hotel recommendations, travel insights and model monitoring.

    streamlit run streamlit_app/app.py
If API_URL is set (Docker/K8s), the price and gender tabs call the Flask API;
otherwise they load the model files directly.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DRIFT_REPORT_PATH, FIGURES_DIR, FLIGHT_MODEL_PATH, GENDER_MODEL_PATH,  # noqa: E402
                        RECOMMENDER_PATH)
from src.data import first_name, load_flights, load_hotels, load_users, outbound_legs  # noqa: E402
from src.tracking import read_metrics  # noqa: E402

API_URL = os.getenv("API_URL", "").rstrip("/")
st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")


@st.cache_resource
def get_model(path):
    return joblib.load(path)


@st.cache_data
def get_data():
    users, flights, hotels = load_users(), load_flights(), load_hotels()
    return users, flights, hotels, outbound_legs(flights)


def call_api(method, path, **kw):
    r = requests.request(method, f"{API_URL}{path}", timeout=10, **kw)
    r.raise_for_status()
    return r.json()


users, flights, hotels, outbound = get_data()
rec = get_model(RECOMMENDER_PATH)

st.title("✈️ Voyage Analytics")
st.caption("Flight price prediction · hotel recommendations · gender classification — "
           + (f"predictions served by API at {API_URL}" if API_URL else "predictions from local model files"))
tab_rec, tab_insights, tab_price, tab_gender, tab_models = st.tabs(
    ["🏨 Hotel recommendations", "📊 Travel insights", "💸 Flight price", "👤 Gender", "🧪 Model monitoring"])

# ---------------------------------------------------------------- recommendations
with tab_rec:
    left, right = st.columns([1, 2])
    with left:
        options = users.sort_values("code").apply(lambda r: f"{r.code} - {r['name']}", axis=1).tolist()
        choice = st.selectbox("Traveller", ["New traveller (no history)"] + options, index=1)
        user_code = None if choice.startswith("New") else int(choice.split(" - ")[0])
        max_price = st.slider("Max nightly price (R$)", 50, int(hotels["price"].max()) + 10,
                              int(hotels["price"].max()) + 10, step=10)
        k = st.slider("Number of suggestions", 1, len(rec.items), 5)
        exclude = st.checkbox("Only hotels I haven't stayed at")
    recs = rec.recommend(user_code, max_price=max_price, k=k, exclude_booked=exclude)
    with right:
        profile = rec.user_profile(user_code) if user_code is not None else None
        if profile:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age", profile["age"])
            c2.metric("Company", profile["company"])
            c3.metric("Hotel stays", profile["hotel_bookings"])
            trips = outbound[outbound["userCode"] == user_code]
            c4.metric("Usual departure", trips["from"].mode().iat[0] if not trips.empty else "—")
        if not recs:
            st.info("No hotels match those filters. Try a higher budget.")
        for i, r in enumerate(recs, 1):
            with st.container(border=True):
                a, b, c = st.columns([3, 1, 1])
                stays = f" · stayed {r['times_booked']}x" if r["times_booked"] else ""
                a.markdown(f"**{i}. {r['name']}** — {r['place']}  \n_{r['reason']}{stays}_")
                b.metric("Per night", f"R$ {r['avg_price']:,.0f}")
                c.metric("Typical stay", f"{r['avg_days']:.1f} nights")
    if user_code is not None:
        hist = hotels[hotels["userCode"] == user_code].sort_values("date", ascending=False)
        with st.expander(f"Booking history ({len(hist)} stays)"):
            st.dataframe(hist[["date", "name", "place", "days", "price", "total"]], hide_index=True,
                         use_container_width=True)

# ---------------------------------------------------------------- insights
with tab_insights:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users", f"{users['code'].nunique():,}")
    m2.metric("Round trips", f"{flights['travelCode'].nunique():,}")
    m3.metric("Hotel stays", f"{len(hotels):,}")
    m4.metric("Avg flight price", f"R$ {flights['price'].mean():,.0f}")

    c1, c2 = st.columns(2)
    monthly = outbound.set_index("date").resample("MS")["price"].count().rename("trips").reset_index()
    c1.plotly_chart(px.line(monthly, x="date", y="trips", title="Trips per month"), use_container_width=True)
    c2.plotly_chart(px.box(flights, x="flightType", y="price", color="agency",
                           category_orders={"flightType": ["economic", "premium", "firstClass"]},
                           title="Flight price by class and agency"), use_container_width=True)

    c3, c4 = st.columns(2)
    dest = hotels.groupby("place").agg(stays=("travelCode", "count"), price=("price", "mean")).reset_index()
    c3.plotly_chart(px.bar(dest.sort_values("stays"), x="stays", y="place", orientation="h", color="price",
                           title="Hotel stays by city (colour = nightly price)"), use_container_width=True)
    routes = flights.groupby(["from", "to", "flightType"]).agg(distance=("distance", "first"),
                                                               price=("price", "mean")).reset_index()
    c4.plotly_chart(px.scatter(routes, x="distance", y="price", color="flightType", hover_data=["from", "to"],
                               title="Average price vs distance per route"), use_container_width=True)

    c5, c6 = st.columns(2)
    od = pd.crosstab(outbound["from"], outbound["to"])
    c5.plotly_chart(px.imshow(od, title="Trips: origin → destination", aspect="auto",
                              color_continuous_scale="Blues"), use_container_width=True)
    demo = users.merge(hotels, left_on="code", right_on="userCode")
    demo["age_band"] = pd.cut(demo["age"], [0, 30, 45, 60, 200], labels=["<30", "30-44", "45-59", "60+"])
    c6.plotly_chart(px.histogram(demo, x="age_band", y="total", histfunc="avg", color="company",
                                 barmode="group", title="Average hotel spend per stay by age and company"),
                    use_container_width=True)
    st.info("**Insight:** destinations are spread almost evenly across cities for every origin, age group "
            "and company, which is why personal history adds only a modest lift to recommendations "
            "(see Model monitoring).")

# ---------------------------------------------------------------- flight price
with tab_price:
    cities = sorted(set(flights["from"]) | set(flights["to"]))
    c1, c2, c3, c4 = st.columns(4)
    origin = c1.selectbox("From", cities)
    dest = c2.selectbox("To", [c for c in cities if c != origin])
    ftype = c3.selectbox("Class", ["economic", "premium", "firstClass"])
    agency = c4.selectbox("Agency", sorted(flights["agency"].unique()))
    payload = {"from": origin, "to": dest, "flightType": ftype, "agency": agency}
    route = flights[(flights["from"] == origin) & (flights["to"] == dest)]
    if not route.empty:
        st.caption(f"Known route · {route['distance'].iat[0]:,.0f} km · {route['time'].iat[0]:.2f} h")
    if st.button("Predict price", type="primary"):
        try:
            if API_URL:
                pred = call_api("POST", "/predict", json=payload)["predictions"]
            else:
                m = get_model(FLIGHT_MODEL_PATH)
                X = pd.DataFrame([{**payload, "distance": m["routes"][f"{origin}|{dest}"]}])
                pred = {"predicted_price": float(m["pipeline"].predict(X[m["features"]])[0]),
                        "model": m["model_type"]}
            st.metric("Predicted price", f"R$ {pred['predicted_price']:,.2f}")
            hist = route[(route["flightType"] == ftype) & (route["agency"] == agency)]["price"]
            if not hist.empty:
                st.caption(f"Historical price for this route/class/agency: R$ {hist.mean():,.2f} · "
                           f"model: {pred.get('model')}")
        except Exception as e:  # surface API/network errors in the UI
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- gender
with tab_gender:
    st.write("Predicts **male / female** from a first name. Users who chose not to disclose "
             "their gender (\"none\") cannot be identified from a name, so they are not a predicted class.")
    name = st.text_input("Full name", "Patricia Carson")
    if name.strip():
        try:
            if API_URL:
                res = call_api("POST", "/predict/gender", json={"name": name})
                probs = res["probabilities"]
            else:
                gm = get_model(GENDER_MODEL_PATH)
                p = gm["pipeline"].predict_proba(pd.DataFrame([{"first_name": first_name(name)}]))[0]
                probs = dict(zip(gm["classes"], p))
            best = max(probs, key=probs.get)
            st.metric("Prediction", best, f"{probs[best]:.0%} confidence", delta_color="off")
            st.progress(float(probs.get("female", 0)), text=f"female {probs.get('female', 0):.0%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- monitoring
with tab_models:
    metrics = read_metrics()
    if not metrics:
        st.info("Train the models first: `make train`")
    if "flight_price" in metrics:
        fp = metrics["flight_price"]
        st.subheader("Flight price regression")
        cand = pd.DataFrame(fp["candidates"]).T.reset_index(names="model")
        st.dataframe(cand.round(3), hide_index=True, use_container_width=True)
        st.caption(f"Champion: **{fp['best_model']}** (MLflow v{fp['registry_version']}) · {fp['split']}. "
                   "`unseen_route_*` = 5-fold CV where test routes never appear in training; "
                   "the API uses the Ridge fallback for such routes.")
    cols = st.columns(2)
    if "gender" in metrics:
        gm = metrics["gender"]
        with cols[0]:
            st.subheader("Gender classifier")
            exp = pd.DataFrame(gm["experiments"]).T.reset_index(names="experiment")
            st.dataframe(exp[["experiment", "accuracy", "f1_macro", "unseen_names_accuracy"]],
                         hide_index=True, use_container_width=True)
    if "recommender" in metrics:
        rm = metrics["recommender"]
        with cols[1]:
            st.subheader("Hotel recommender")
            t = pd.DataFrame(rm["test"]).T.reset_index(names="ranking")
            st.dataframe(t[["ranking", "hit_at_1", "hit_at_3", "mrr"]].round(3), hide_index=True,
                         use_container_width=True)
            st.caption(f"Tuned weights: {rm['weights']}")
    if DRIFT_REPORT_PATH.exists():
        st.subheader("Data drift (PSI, last 90 days vs history)")
        d = json.loads(DRIFT_REPORT_PATH.read_text())
        psi = pd.Series(d["psi"], name="PSI").rename_axis("feature").reset_index()
        fig = px.bar(psi, x="feature", y="PSI", title=f"Alert threshold {d['threshold']}")
        fig.add_hline(y=d["threshold"], line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        (st.error if d["drift_detected"] else st.success)(
            f"Drift detected in {list(d['drifted_features'])}" if d["drift_detected"] else "No significant drift")
    figs = sorted(FIGURES_DIR.glob("*.png"))
    if figs:
        with st.expander(f"Evaluation figures ({len(figs)})"):
            for row in range(0, len(figs), 2):
                cs = st.columns(2)
                for c, f in zip(cs, figs[row:row + 2]):
                    c.image(str(f), caption=f.stem.replace("_", " "))
