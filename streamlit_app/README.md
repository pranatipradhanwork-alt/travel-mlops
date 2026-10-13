# Streamlit App — Voyage Analytics

```bash
make ui                                   # http://localhost:8501, uses local model files
API_URL=http://localhost:5000 make ui     # price and gender predictions go through the Flask API
```

| Tab | What you can do |
|---|---|
| 🏨 Hotel recommendations | Pick a traveller (or "new traveller" for cold start). Filter by nightly budget and "only hotels I haven't stayed at". See ranked hotels with the reason for each, the user's profile, usual departure city and booking history. |
| 📊 Travel insights | KPIs, trips per month, price by class/agency, stays per city, price vs distance per route, an origin→destination heatmap, and spend by age and company. |
| 💸 Flight price | Choose route, class and agency to get a predicted price, compared with the historical price. |
| 👤 Gender | Type a name to get male/female with a confidence score. |
| 🧪 Model monitoring | Candidate-model comparison, gender experiments, recommender vs baselines, PSI drift chart, and all evaluation figures. |

Why Streamlit: the whole UI is pure Python, so the same `src/` code that trains the models also renders them,
without a separate frontend stack. Data and models are cached with `st.cache_data` / `st.cache_resource`.

In Docker and Kubernetes, `API_URL` points at the API service, so the UI and the API scale independently.
