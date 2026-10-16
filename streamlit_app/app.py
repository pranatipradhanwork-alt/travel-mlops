"""Objective 9: Streamlit UI for hotel recommendations, travel insights and model monitoring.

    streamlit run streamlit_app/app.py
If API_URL is set (Docker/K8s), the price and gender tabs call the Flask API;
otherwise they load the model files directly.
"""
import html
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

PALETTE = ["#2563EB", "#14B8A6", "#F59E0B", "#8B5CF6", "#EF4444", "#0EA5E9"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = "Blues"

st.markdown("""<style>
.block-container {padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.hero {background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #14B8A6 100%); color: #fff;
       border-radius: 18px; padding: 28px 32px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; font-size: 2.3rem; margin: 0; padding: 0;}
.hero p {color: #DBEAFE; margin: 6px 0 18px; font-size: 1.05rem;}
.stats {display: flex; flex-wrap: wrap; gap: 12px;}
.stat {background: rgba(255,255,255,.14); border-radius: 12px; padding: 10px 16px; min-width: 130px;}
.stat b {display: block; font-size: 1.4rem;} .stat span {font-size: .85rem; color: #DBEAFE;}
.card {background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 18px;
       margin-bottom: 10px; display: flex; align-items: center; gap: 16px;
       box-shadow: 0 1px 2px rgba(15,23,42,.04);}
.rank {background: #DBEAFE; color: #1E40AF; font-weight: 700; border-radius: 10px;
       min-width: 38px; height: 38px; display: flex; align-items: center; justify-content: center;}
.card .main {flex: 1;} .card .name {font-weight: 600; font-size: 1.05rem;}
.chip {display: inline-block; background: #CCFBF1; color: #115E59; border-radius: 999px;
       padding: 1px 10px; font-size: .8rem; margin-left: 6px;}
.card .why {color: #64748B; font-size: .88rem; margin-top: 2px;}
.card .num {text-align: right; min-width: 92px;} .card .num b {font-size: 1.15rem; display: block;}
.card .num span {color: #64748B; font-size: .78rem;}
.result {background: #fff; border: 1px solid #E2E8F0; border-left: 6px solid #2563EB;
         border-radius: 14px; padding: 18px 22px;}
.result .big {font-size: 2.2rem; font-weight: 700; color: #1E3A8A;}
.result .sub {color: #64748B;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_model(path):
    return joblib.load(path)


@st.cache_resource  # shared and never copied: the data is ~200 MB in memory and read-only
def get_data():
    users, flights, hotels = load_users(), load_flights(), load_hotels()
    return users, flights, hotels, outbound_legs(flights)


def call_api(method, path, **kw):
    r = requests.request(method, f"{API_URL}{path}", timeout=10, **kw)
    r.raise_for_status()
    return r.json()


users, flights, hotels, outbound = get_data()
rec = get_model(RECOMMENDER_PATH)

n_cities = len(set(flights["from"].unique()) | set(flights["to"].unique()))
st.markdown(f"""<div class="hero"><h1>✈️ Voyage Analytics</h1>
<p>Hotel recommendations, flight price prediction and travel insights for a Brazilian travel agency.</p>
<div class="stats">
<div class="stat"><b>{users['code'].nunique():,}</b><span>travellers</span></div>
<div class="stat"><b>{flights['travelCode'].nunique():,}</b><span>round trips</span></div>
<div class="stat"><b>{len(hotels):,}</b><span>hotel stays</span></div>
<div class="stat"><b>{n_cities}</b><span>cities</span></div>
</div></div>""", unsafe_allow_html=True)
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
            stays = f" · you stayed here {r['times_booked']}x" if r["times_booked"] else ""
            st.markdown(f"""<div class="card"><div class="rank">{i}</div>
<div class="main"><div class="name">{html.escape(r['name'])}<span class="chip">{html.escape(r['place'])}</span></div>
<div class="why">{html.escape(r['reason'])}{stays}</div></div>
<div class="num"><b>R$ {r['avg_price']:,.0f}</b><span>per night</span></div>
<div class="num"><b>{r['avg_days']:.1f}</b><span>nights, typical</span></div></div>""",
                        unsafe_allow_html=True)
    if user_code is not None:
        hist = hotels[hotels["userCode"] == user_code].sort_values("date", ascending=False)
        with st.expander(f"Booking history ({len(hist)} stays)"):
            st.dataframe(hist[["date", "name", "place", "days", "price", "total"]], hide_index=True,
                         width="stretch")

# ---------------------------------------------------------------- insights
@st.cache_data  # small figures; each session gets its own copy, so concurrent renders cannot clash
def insight_figures():
    """Built once per server: Streamlit re-runs the whole script on every click."""
    route_counts = outbound.groupby(["from", "to"]).size()
    busiest = route_counts.idxmax()
    stats = {"flight": flights["price"].mean(), "night": hotels["price"].mean(), "days": hotels["days"].mean(),
             "route": f"{busiest[0].split(' (')[0]} → {busiest[1].split(' (')[0]}", "route_n": route_counts.max()}
    figs = []
    monthly = outbound.set_index("date").resample("MS")["price"].count().rename("trips").reset_index()
    figs.append(px.line(monthly, x="date", y="trips", title="Trips per month"))
    # A sample keeps the box plot's shape without sending ~270k prices to the browser.
    sample = flights.sample(min(len(flights), 5000), random_state=0)
    figs.append(px.box(sample, x="flightType", y="price", color="agency",
                       category_orders={"flightType": ["economic", "premium", "firstClass"]},
                       title="Flight price by class and agency"))
    dest = hotels.groupby("place").agg(stays=("travelCode", "count"), price=("price", "mean")).reset_index()
    figs.append(px.bar(dest.sort_values("stays"), x="stays", y="place", orientation="h", color="price",
                       title="Hotel stays by city (colour = nightly price)"))
    routes = flights.groupby(["from", "to", "flightType"]).agg(distance=("distance", "first"),
                                                               price=("price", "mean")).reset_index()
    figs.append(px.scatter(routes, x="distance", y="price", color="flightType", hover_data=["from", "to"],
                           title="Average price vs distance per route"))
    od = pd.crosstab(outbound["from"], outbound["to"])
    figs.append(px.imshow(od, title="Trips: origin → destination", aspect="auto", color_continuous_scale="Blues"))
    demo = users.merge(hotels, left_on="code", right_on="userCode")
    demo["age_band"] = pd.cut(demo["age"], [0, 30, 45, 60, 200], labels=["<30", "30-44", "45-59", "60+"])
    spend = demo.groupby(["age_band", "company"], observed=True)["total"].mean().reset_index()
    figs.append(px.bar(spend, x="age_band", y="total", color="company", barmode="group",
                       title="Average hotel spend per stay by age and company"))
    return stats, figs


with tab_insights:
    stats, figs = insight_figures()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg flight price", f"R$ {stats['flight']:,.0f}")
    m2.metric("Avg hotel night", f"R$ {stats['night']:,.0f}")
    m3.metric("Avg hotel stay", f"{stats['days']:.1f} nights")
    m4.metric(f"Busiest route: {stats['route']}", f"{stats['route_n']:,} trips")
    for left_fig, right_fig in zip(figs[::2], figs[1::2]):
        c1, c2 = st.columns(2)
        c1.plotly_chart(left_fig, width="stretch")
        c2.plotly_chart(right_fig, width="stretch")
    st.info("**Insight:** destinations are spread almost evenly across cities for every origin, age group "
            "and company, which is why personal history adds only a modest lift to recommendations "
            "(see Model monitoring).")

# ---------------------------------------------------------------- flight price
with tab_price:
    cities = sorted(set(flights["from"].unique()) | set(flights["to"].unique()))
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
        classes = ["economic", "premium", "firstClass"]
        try:
            batch = [{**payload, "flightType": c} for c in classes]
            if API_URL:
                preds = call_api("POST", "/predict", json=batch)["predictions"]
            else:
                m = get_model(FLIGHT_MODEL_PATH)
                X = pd.DataFrame([{**b, "distance": m["routes"][f"{origin}|{dest}"]} for b in batch])
                preds = [{"predicted_price": float(p), "model": m["model_type"]}
                         for p in m["pipeline"].predict(X[m["features"]])]
            by_class = {c: p["predicted_price"] for c, p in zip(classes, preds)}
            pred = preds[classes.index(ftype)]
            hist = route[(route["flightType"] == ftype) & (route["agency"] == agency)]["price"]
            hist_txt = f"Historical average for this trip: R$ {hist.mean():,.2f} · " if not hist.empty else ""
            left, right = st.columns([1, 1.3])
            left.markdown(f"""<div class="result"><div class="sub">Predicted fare · {html.escape(origin)}
→ {html.escape(dest)} · {ftype} · {html.escape(agency)}</div>
<div class="big">R$ {pred['predicted_price']:,.2f}</div>
<div class="sub">{hist_txt}model: {pred.get('model')}</div></div>""", unsafe_allow_html=True)
            comp = pd.DataFrame({"class": classes, "price": [by_class[c] for c in classes]})
            fig = px.bar(comp, x="class", y="price", text_auto=",.0f", color="class",
                         title=f"Same route with {agency}, by class")
            fig.update_layout(showlegend=False, height=300, margin=dict(t=50, b=10), yaxis_title="R$")
            right.plotly_chart(fig, width="stretch")
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
        st.dataframe(cand.round(3), hide_index=True, width="stretch")
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
                         hide_index=True, width="stretch")
    if "recommender" in metrics:
        rm = metrics["recommender"]
        with cols[1]:
            st.subheader("Hotel recommender")
            t = pd.DataFrame(rm["test"]).T.reset_index(names="ranking")
            st.dataframe(t[["ranking", "hit_at_1", "hit_at_3", "mrr"]].round(3), hide_index=True,
                         width="stretch")
            st.caption(f"Tuned weights: {rm['weights']}")
    if DRIFT_REPORT_PATH.exists():
        st.subheader("Data drift (PSI, last 90 days vs history)")
        d = json.loads(DRIFT_REPORT_PATH.read_text())
        psi = pd.Series(d["psi"], name="PSI").rename_axis("feature").reset_index()
        fig = px.bar(psi, x="feature", y="PSI", title=f"Alert threshold {d['threshold']}")
        fig.add_hline(y=d["threshold"], line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
        (st.error if d["drift_detected"] else st.success)(
            f"Drift detected in {list(d['drifted_features'])}" if d["drift_detected"] else "No significant drift")
    figs = sorted(FIGURES_DIR.glob("*.png"))
    if figs:
        with st.expander(f"Evaluation figures ({len(figs)})"):
            for row in range(0, len(figs), 2):
                cs = st.columns(2)
                for c, f in zip(cs, figs[row:row + 2]):
                    c.image(str(f), caption=f.stem.replace("_", " "))

st.divider()
st.caption("Voyage Analytics · ML models trained on the agency's flight, hotel and user data · "
           + (f"predictions served by the API at {API_URL}" if API_URL else "predictions from the bundled models"))
