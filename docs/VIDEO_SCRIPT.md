# Video Script (~15 minutes) + Interview Prep

Each section lists **what to show on screen** and **what to say**. The spoken text is a suggestion, so rephrase it
in your own words. Speaking naturally matters more than reading word for word.

**Before recording:**
- run `make train`, `make api`, `make ui` and `make mlflow-ui`;
- open VS Code on the repo;
- have tabs ready for Streamlit (8501), MLflow (5001), and a terminal for `curl`.

---

## 1. Introduction — 1.5 min
**Show:** the README architecture diagram.

> "Hi, I'm ___. In this project I built Voyage Analytics, an end-to-end machine-learning system for the travel
> and tourism domain. It solves three business problems: predicting flight prices, classifying a user's gender, and
> recommending hotels.
>
> This isn't just a set of models in a notebook. It's a production-style system. The models are written in Python
> with scikit-learn and served through a Flask REST API. The API is packaged with Docker and deployed on Kubernetes,
> so it can scale with traffic. Apache Airflow automates retraining, Jenkins runs CI/CD, MLflow tracks every
> experiment and model version, and a Streamlit app gives business users an interactive front end.
>
> Over the next fifteen minutes I'll walk through the problem, the data, the models, how they're served, and the MLOps
> pipeline around them."

## 2. Problem understanding — 2 min
**Show:** the "Business problem" section of the project report.

> "Why machine learning here? A travel platform lives on two things: price transparency and personalisation. If I can
> estimate a fare instantly, users can plan without searching every combination, and the business can spot mispriced
> inventory. Understanding who the users are supports segment analytics, and recommending the right hotel raises
> conversion.
>
> So the core tasks are prediction, which is a regression for price; classification for gender; recommendation for
> hotels; and automation, to keep all of it fresh.
>
> Why isn't a notebook enough? A notebook proves a model worked once, on my laptop. The business needs an answer in
> milliseconds, from a service that's always up, that handles more traffic at peak times, that can be retrained when
> data changes, and where we always know which model version produced which prediction. That's what
> productionisation adds, and it's most of the work in this project."

## 3. Data understanding & use-case mapping — 2 min
**Show:** Streamlit → Travel insights tab, then the origin→destination heatmap.

> "There are three datasets. Users has 1,340 people with company, name, gender and age. Flights has about 272,000
> flight legs, which is around 136,000 round trips, with origin, destination, cabin class, agency, distance, time, date
> and price. Hotels has about 40,000 stays with hotel, city, nights, nightly price and total.
>
> They connect through two keys: userCode links flights and hotels to the user, and travelCode links a hotel stay
> to its trip. Something I noticed early is that every trip appears twice in flights, an outbound and a return leg
> sharing the same travelCode. So when I measure where people choose to go, I only use the outbound leg.
>
> Mapping data to features: flights drive the price model, users drive the gender model, and hotels plus flights drive
> the recommender.
>
> Exploring the data changed my design. Price turned out to be a deterministic function of route, class and agency.
> Time and distance are perfectly correlated, and dates don't affect price. And this heatmap shows destinations are
> almost evenly spread for every origin, which tells me in advance that recommendation gains will be modest."

## 4. Model development & serving design — 2.5 min
**Show:** `src/train_flight_price.py`, the MLflow comparison, then the terminal with `curl`.

> "For flight price, I selected five features: origin, destination, class, agency and distance. I dropped time and
> date features, and an ablation run in MLflow confirms they add nothing. I compared Ridge, Random Forest and gradient
> boosting.
>
> Evaluation is where I was careful. I used a time-based split: train on older flights, test on newer ones, just like
> production. Random Forest gets R² of 1.0, because the price really is a lookup. To check it wasn't just
> memorising, I also tested on routes the model had never seen. There, Ridge generalises best, 0.88 against 0.70.
> So the API uses the Random Forest for known routes and automatically falls back to Ridge for new ones.
>
> For gender, I use character n-grams of the first name with logistic regression: 89% accuracy, and 76% on names
> it has never seen. A third of users are labelled 'none'. Their names are ordinary names, so that label means
> 'not disclosed' and can't be predicted. I serve a male/female model and log the three-class experiment for
> transparency.
>
> Serving: the API receives JSON, validates it, builds a row, and passes it through a scikit-learn pipeline that
> contains both the preprocessing and the model. That's important, because it means training and serving can't
> drift apart." *(run the curl)* "Here's a real request. The response includes the price, whether the route was known,
> which model answered, and the model version from the registry."

## 5. MLOps pipeline & deployment — 2 min
**Show:** `Dockerfile` → `k8s/api-hpa.yaml` → `Jenkinsfile` stages → Airflow DAG graph → MLflow Models page.

> "Docker packages the API with its exact dependencies and models into one image: a multi-stage build that runs as a
> non-root user and has a health check. The tag is the build number plus the git commit, so every image is one exact
> version.
>
> Kubernetes runs that image. It does rolling updates with zero downtime and readiness probes, so a pod only gets traffic
> once its models are loaded. A Horizontal Pod Autoscaler scales from 2 to 10 pods when CPU passes 60%.
>
> Jenkins is the CI/CD pipeline. On every commit it lints, validates the data, retrains with quality gates, runs 24
> tests, builds the images, smoke-tests the container, pushes, and deploys. If the rollout isn't healthy it rolls
> back automatically.
>
> Airflow handles the data side. Weekly, or when new files land, it ingests, validates, checks for drift, trains all
> three models in parallel, and if they pass the gates it triggers Jenkins to ship them.
>
> And MLflow ties it together. Every run logs parameters, metrics and plots, and the model registry versions each
> model. The version that passes its gate gets the 'champion' alias, so rolling back is just moving that alias."

## 6. Recommendation layer, app & monitoring — 2 min
**Show:** Streamlit → Hotel recommendations (pick a user, change the budget) → Model monitoring tab.

> "The recommender is a hybrid. It scores each hotel using the user's history, collaborative similarity to what they
> booked, popularity, budget fit, and a penalty for their home city, since people rarely book a hotel where they live.
> I tuned the weights with a grid search on a time-based validation window and report results on a later, untouched
> window. It beats both random and popularity baselines. The gain is modest, but that's honest, given how little
> preference signal this data contains.
>
> I chose Streamlit because the whole UI is Python, so it reuses the same code as training. Business users can explore
> recommendations with reasons, filter by budget, see travel insights, and check predictions.
>
> For monitoring, this tab shows every model's metrics and a drift chart using the Population Stability Index. All
> features are stable right now. The API also exposes Prometheus metrics: request counts, latency, how often the
> fallback model is used, and the average predicted price, which is an early signal of output drift."

## 7. Reliability, challenges & optimisation — 1.5 min
**Show:** the "Challenges" table in PROJECT_REPORT.md.

> "A few challenges. First, multiple datasets with duplicate trip codes, which I solved with explicit outbound-leg
> logic. Second, the gradient-boosting model crashed on the full data because of sparse encoding, which I only found
> by running the real data end to end. Third, my first recommender evaluation was inflated, because tied scores counted
> as hits. Fixing that took hit@3 from a misleading 0.45 to an honest 0.375. And fourth, the integration work itself:
> keeping Airflow's environment separate from the ML stack, and avoiding parallel tasks writing one metrics file.
>
> Optimisations: better feature selection backed by an ablation, a single shared code base for training, API, UI and
> Airflow, and scalable deployment with autoscaling.
>
> For reliability: code in git, models in the registry, immutable image tags, archived data snapshots, a fixed random seed
> for repeatability, and quality gates at every stage."

## 8. Learnings & future improvements — 1.5 min
> "My biggest learning is that training the model is maybe 20% of the work. Deploying a system means thinking about
> validation, packaging, scaling, automation and monitoring. And evaluation design matters more than the algorithm:
> the same model scores 1.0 or 0.7 depending on the question you ask.
>
> Next steps: Prometheus and Grafana dashboards with alerts, retraining triggered by drift instead of a schedule,
> canary or A/B releases for new models, authentication and rate limiting on the API, learning-to-rank once real
> click data exists, and moving to managed cloud services like EKS and a hosted MLflow.
>
> Thank you for watching."

---

# Follow-up interview questions — prepared answers

**1. How would you productionize an ML model for real-time travel pricing?**
Wrap preprocessing and model in one pipeline object so serving equals training. Serve it behind a stateless REST/gRPC API
with input validation and versioned responses, packaged in a container and run on Kubernetes with autoscaling. Cache
hot routes, and keep p99 latency low by loading models once (gunicorn preload). Track every model in a registry, roll out
with canary releases, and monitor latency, error rate, input drift (PSI) and prediction drift. When real fares arrive,
compare them with predictions. For real pricing, add demand, seasonality and lead-time features and retrain often.

**2. Why did you use Docker and Kubernetes?**
Docker gives **reproducibility and portability**: the same image, with the same dependencies and models, runs on my laptop,
in CI and in production, and the tag identifies the exact version. Kubernetes gives **operability at scale**: self-healing
(probes restart bad pods), horizontal autoscaling under load, zero-downtime rolling updates, easy rollback, and
declarative config stored in git.

**3. How does MLflow help in managing model versions?**
Every training run records parameters, metrics, artifacts and the model with its input signature, so any result can be
reproduced and compared. The **Model Registry** gives each model sequential versions, and aliases such as `@champion` mark
what's in production. Promotion and rollback become a metadata change, and the API reports which version answered.
I also log the losing candidates and the ablation runs, which documents why a model was chosen.

**4. How would you handle model retraining in production?**
Retrain on a schedule *and* on triggers: new data landing, a PSI drift alert, or a drop in live accuracy once labels
arrive. Run the pipeline in Airflow: validate → train → evaluate on a time-based holdout → compare against the current champion
→ promote only if the new model is better and passes the gates → ship through CI/CD. Go further with shadow or canary deployment
and an A/B test before full rollout. Keep old versions so rollback is instant.

**5. How would you scale this for a large travel platform?**
* **Serving:** run more replicas behind a load balancer with an HPA, cache popular route predictions (Redis), batch requests,
  and keep the API stateless.
* **Data:** move from CSV to a warehouse or lakehouse, and use Spark or Dask for feature pipelines.
* **Consistency:** add a feature store for training/serving consistency.
* **Recommendations:** precompute candidates offline and re-rank online.
* **Infrastructure:** use managed Kubernetes across zones, a hosted MLflow on Postgres + S3, and managed Airflow.
* **Operations:** observability with Prometheus/Grafana and alerting, plus SLOs on latency and availability.
