# Model Evaluation & Analysis

All numbers in this report come from `make train` on the full capstone dataset
(1,340 users · 271,888 flight legs / 135,944 round trips · 40,552 hotel stays).
Every number is logged in MLflow (`make mlflow-ui`) and in `models/metrics/*.json`.
The figures are in `reports/figures/`.

---

## 1. What the data tells us before modelling

| Finding | Evidence | Consequence |
|---|---|---|
| Flights are **round trips**. The outbound and return legs share a `travelCode`. | Every `travelCode` appears exactly twice. | Destination preferences use the outbound leg only (`src/data.py: outbound_legs`). |
| Flight **price is a deterministic function** of (from, to, flightType, agency). | Standard deviation of price within each of the 490 groups = **0.00**. | A tree model can reach R² ≈ 1.0 on known routes, so a random split proves little. We add an **unseen-route** test. |
| `time` and `distance` carry the same information. | Pearson r = **1.000**, and each route has exactly one distance. | `time` is dropped. Feature selection is confirmed by an ablation run. |
| Prices have **no time effect**. | Time-based and random splits give the same R² for every model. | Calendar features are dropped. Confirmed by the same ablation run. |
| Class and agency drive price. | Mean price: economic R$658 · premium R$920 · firstClass R$1,181. FlyingDrops is ~29% more expensive than CloudFy/Rainbow. | Both are kept as categorical features. |
| **One hotel per city** (9 hotels), each with one fixed nightly price. | 9 hotel names, 9 places, 9 (name, place) pairs. Price std = 0. | Recommending a hotel is the same as recommending a destination. |
| **Destinations are close to uniformly random.** | For every origin, each other city gets 12–13% of trips. A gradient-boosting classifier using origin, month, weekday, age, company, gender, class and agency reaches 12.3% accuracy, against 12.5% for a uniform guess. | Personalisation can only give a modest lift. The recommender is judged against random and popularity baselines. |
| Gender `none` is **one third** of users (440 of 1,340). These users have ordinary gendered names (e.g. "Debbie Helms", "Jack Sabo"). | Name-based 3-class accuracy is only 54%. | `none` means "not disclosed". The served model predicts male/female only. |
| Trip volume **declines steadily** from 2019 to 2023. | Trips per month (Streamlit → Travel insights). | The time-based test set (newest 20% of flights) is smaller. Drift is watched with PSI. |

---

## 2. Flight price regression

**Protocol.**
- **Primary test (time-based holdout):** train on flights before 2021-11-25 (80%, 217,230 rows) and test on the newest 20% (54,658 rows). This mirrors production, where the model predicts future bookings.
- **Robustness test (unseen routes):** 5-fold `GroupKFold` on undirected routes, so every route in the test fold is missing from training.

| Model | Features | RMSE | MAE | R² | MAPE | Unseen-route R² | Unseen-route MAE |
|---|---|---|---|---|---|---|---|
| Ridge | selected | 102.16 | 80.40 | 0.921 | 10.2% | **0.882** | **97.3** |
| **Random Forest** ✅ | selected | **0.00** | **0.00** | **1.000** | **0.0%** | 0.698 | 154.8 |
| HistGradientBoosting | selected | 3.39 | 2.49 | 1.000 | 0.31% | 0.770 | 134.2 |
| HGB + `time` + date features (ablation) | all | 3.33 | 2.47 | 1.000 | 0.30% | 0.771 | 134.1 |

*Selected features: `from`, `to`, `flightType`, `agency`, `distance`.*

![model comparison](figures/flight_model_comparison.png)

**Interpretation**
* **Random Forest reproduces every known price exactly.** Because the price is a lookup over
  490 (route × class × agency) combinations, a forest with enough depth memorises each
  one. This is correct behaviour for this data, not leakage: the test rows are *later in time*
  than the training rows.
* **Feature selection is justified by the ablation.** Adding `time` and four calendar features changes
  RMSE by 0.06 (3.39 → 3.33) and unseen-route R² by 0.001.
* **Tree models generalise worse to new routes.** Ridge learns additive city effects plus a distance
  slope, so it extrapolates to new routes (R² 0.88). The trees memorise route identity (0.70–0.77).
  **Design decision:** the API serves the Random Forest for known routes and switches
  automatically to the Ridge fallback when a route was never seen in training. The response reports which
  one was used (`known_route`, `model`).
* **Permutation importance** (RMSE increase when a feature is shuffled): distance 394 · flightType 325 ·
  from 181 · to 117 · agency 42. This matches the business intuition: distance and cabin class set the price.

![importance](figures/flight_feature_importance.png)
![pred vs actual](figures/flight_pred_vs_actual.png)

**Quality gate:** time-holdout R² ≥ 0.90 (`MIN_FLIGHT_R2`). Passed.

---

## 3. Gender classification

**Protocol.**
- **Stratified 5-fold cross-validation**, reporting accuracy and macro precision, recall and F1.
- **Unseen names:** `GroupKFold` grouped by first name, so the model is tested only on first names it has never seen.
- **World names:** 5-fold held-out accuracy on 31,297 labelled first names from many countries, and on the
  1,220 of them used in India. A test name is always removed from all training data, including `users.csv`.

| Experiment | Training names | Accuracy | Macro F1 | Unseen-names accuracy | World held-out | Indian held-out |
|---|---|---|---|---|---|---|
| Name only, female / male / none | users | 0.543 | 0.530 | 0.477 | – | – |
| Name + age, company, travel behaviour | users | 0.900 | 0.900 | 0.753 | – | – |
| Name only (served until Oct 2026) | users | 0.894 | 0.894 | 0.764 | 0.744 | 0.685 |
| **Name only** ✅ (served) | **users + world names** | **0.931** | **0.931** | **0.886** | **0.891** | **0.868** |

Per class (served model): female precision 0.95 / recall 0.91 · male precision 0.91 / recall 0.95.

![confusion](figures/gender_confusion_binary_name_plus_world_names.png)

**World names (`data/world_names.csv`).** 31,297 first names labelled clearly male (16,083) or female (15,214),
taken from Jörg Michael's `nam_dict` (via the `gender-guesser` package, GNU FDL 1.2+). Unisex names are dropped.
The old model learned from only 484 distinct first names, so it missed patterns from other countries: it
called *Aarti* male and was 68.5% accurate on Indian names. Adding the world names keeps the same model
(TF-IDF character 2–4-grams + logistic regression, C=5) and raises never-seen-name accuracy by 12 points and
Indian names by 18 points. We also tried 1–5-grams and C = 20 or 50. All six settings were within about 1 point
of each other, so we kept the original setting and changed only the data.

**Interpretation**
* **`none` cannot be predicted.** Adding it as a third class drops accuracy from 89% to 54%, and the
  3-class confusion matrix shows `none` spread across all predictions. Those users simply did not disclose a
  gender.
* **Behavioural features add nothing reliable.** They gain +0.6 points on regular CV but lose 1.1 points
  on unseen names, and the averages per gender are almost identical (age 42.3–43.2, first-class share 42–43%).
  Following Occam's razor, we serve the simpler name-only model.
* **More names beat more features.** The model generalises to new names at 89% accuracy. It learns sub-word
  patterns (e.g. names ending in *-a*, *-ine*) through character 2–4-grams, and more varied names teach it more patterns.
* **Ethics note.** Inferring gender from a name is unreliable and gender is not binary. The API response says so,
  and the prediction should never drive decisions about individuals.

**Quality gate:** macro F1 ≥ 0.80 (`MIN_GENDER_F1`). Passed.

### 3.1 Abstaining with "uncertain" (non-names and close calls)

**Problem.** The model always picks male or female, even for input that is not a name.
With the old model, "abc" returned *male* at 0.501 (a coin flip), "abcabc abcabc" *female* at 0.633, and random
letters like "qzvk" *male* at 0.93. The new model is even surer on gibberish ("abc" → male 0.92), so a
confidence cutoff alone cannot fix this.

**Fix (`src/gender_rules.py`, used by the API and the Streamlit app).**
1. **Input check → HTTP 400:** only letters, spaces, hyphens and apostrophes are allowed, and the first name
   needs at least 2 letters. The minimum is 2, not 3, because real names such as *Al*, *Ed* and *Jo* exist.
2. **Known-names check → "uncertain":** the first name must be in a list of 34,484 first names from many
   countries (`src/first_names.txt`: US Census 1990, plus `nam_dict` with names from 50+ countries including
   India, plus every name in `users.csv`). This is what stops gibberish.
3. **Confidence cutoff → "uncertain":** the top probability must be at least **0.60**.

**Choosing the cutoff (served model).** Out-of-fold probabilities: users scored with `GroupKFold(5)` by first
name, and world names with 5-fold. Every name is scored by a model that never saw it.

| Cutoff | Users: names → uncertain | Users: accuracy on the rest | Users: wrong answers caught | World: → uncertain / accuracy on rest | Indian: → uncertain / accuracy on rest |
|---|---|---|---|---|---|
| none | 0% | 88.6% | 0 / 103 | 0% / 89.1% | 0% / 86.8% |
| 0.55 | 3.3% | 90.7% | 22 / 103 | 3.2% / 90.4% | 3.6% / 88.5% |
| **0.60** ✅ | **8.0%** | **91.8%** | **35 / 103** | **6.5% / 91.4%** | **7.4% / 89.8%** |
| 0.65 | 11.7% | 92.3% | 42 / 103 | 9.7% / 92.4% | 10.7% / 91.1% |
| 0.70 | 15.6% | 93.3% | 52 / 103 | 13.5% / 93.3% | 14.9% / 92.2% |

We picked **0.60**: it removes a third of the wrong answers while giving no answer for only about 1 name in 13.
Higher cutoffs gain about 0.5–1.5 points of accuracy for every 4 more points of real names left unanswered.
With the old model, 0.60 left 18.4% of never-seen names unanswered (79.4% accuracy on the rest), so the
retrained model abstains less than half as often.

The names check handles the gibberish: none of the 26 test strings ("abc", "abcabc", "axhy", "qzvk", "qwerty",
"user", …) is in the list, so all of them return "uncertain".

**Limitations.** Some names stay hard from spelling alone. *Pranati* gets female 0.54 → "uncertain", and
*Aarti* still comes out male 0.66 even though it is labelled female in the training data, because the model learns
patterns, not single names. Rare names missing from the list get "uncertain" rather than an error.

---

## 4. Hotel recommendation

**Protocol.** Per-user temporal split for users with at least 5 stays:
- Weights are **tuned** by fitting on the first 60% of each user's stays and validating on the next 20%.
- Results are **reported** by fitting on the first 80% and testing on the last 20% (7,607 held-out stays).

The test window is never seen during tuning. A grid search over 105 weight combinations optimises MRR.
Ties are broken randomly, so sparse signals cannot score hits from tied zeros.

| Ranking | hit@1 | hit@3 | MRR | NDCG@3 |
|---|---|---|---|---|
| Random | 0.105 | 0.329 | 0.310 | 0.232 |
| Global popularity | 0.126 | 0.365 | 0.333 | 0.261 |
| User history only | 0.123 | 0.367 | 0.333 | 0.261 |
| **Hybrid (tuned)** ✅ | **0.126** | **0.375** | **0.339** | **0.266** |

The tuned weights are collaborative similarity 0.5 and home-city penalty 3.0. History, popularity and budget all get weight 0.

![recommender](figures/recommender_vs_baselines.png)

**Interpretation**
* With 9 hotels, a random ranking already gets hit@3 = 3/9 ≈ 0.33. The tuned hybrid adds **+4.6 points
  hit@3 (+14% relative)** over random, and beats popularity and history-only on every metric.
* The **home-city penalty** is the strongest signal: people rarely book a hotel in the city they fly
  out of. Item-item collaborative similarity adds a small further gain.
* The gain is modest because, as section 1 shows, **the data contains almost no destination preference
  signal**. A model cannot beat the information in the data. On real booking data, the same pipeline (signals →
  weight grid search → temporal validation) would automatically give more weight to history and budget.

---

## 5. Monitoring

* `src/monitor.py` computes the **Population Stability Index** (PSI) between the training history and the last 90 days
  for price, distance, flightType, agency, from and to. The latest run shows every PSI below 0.04, so there is no drift.
  The alert threshold is 0.2.
* Airflow runs this check on every retraining (`check_drift`), and Streamlit plots it (Model monitoring tab).
* The API exposes `/metrics` in Prometheus format. It reports request counts and latency per endpoint, known-route vs fallback
  predictions, the mean predicted price (to spot output drift), and cold-start vs known-user recommendations.

## 6. Limitations & next steps

* The dataset is synthetic: prices are deterministic and destinations random. The flight model's perfect
  score would not hold on real fares. Real fares would need demand, seasonality and lead-time features.
* There is no user feedback loop. Click and booking logs would allow online evaluation (A/B tests) of the recommender.
* A gender model trained only on names inherits cultural bias in naming conventions.
