"""Loading, validation and feature engineering shared by training, API and UI.

Dataset facts that drive the design (see reports/MODEL_EVALUATION.md):
  * flights.csv holds round trips: outbound and return legs share a travelCode.
  * Flight price is fully determined by (from, to, flightType, agency); `time`
    and `distance` are perfectly correlated, and every route has one distance.
  * hotels.csv has one hotel per city, and each stay matches the outbound leg's destination.
"""
import pandas as pd

from src.config import FLIGHTS_CSV, HOTELS_CSV, USERS_CSV

REQUIRED_COLUMNS = {
    "users": {"code", "company", "name", "gender", "age"},
    "flights": {"travelCode", "userCode", "from", "to", "flightType", "price",
                "time", "distance", "agency", "date"},
    "hotels": {"travelCode", "userCode", "name", "place", "days", "price", "total", "date"},
}

# Selected features. `time` is dropped (r = 1.0 with distance) and calendar
# features are dropped (no effect on price - see the ablation run in MLflow).
FLIGHT_CATEGORICAL = ["from", "to", "flightType", "agency"]
FLIGHT_NUMERIC = ["distance"]
FLIGHT_FEATURES = FLIGHT_CATEGORICAL + FLIGHT_NUMERIC
DATE_FEATURES = ["month", "day_of_week", "year", "is_weekend"]


def load_users(path=USERS_CSV):
    return pd.read_csv(path)


def load_flights(path=FLIGHTS_CSV):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def load_hotels(path=HOTELS_CSV):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def validate():
    """Basic data-quality checks; raises ValueError with every problem found."""
    problems = []
    frames = {"users": load_users(), "flights": load_flights(), "hotels": load_hotels()}
    for name, df in frames.items():
        missing = REQUIRED_COLUMNS[name] - set(df.columns)
        if missing:
            problems.append(f"{name}: missing columns {sorted(missing)}")
        if df.empty:
            problems.append(f"{name}: empty")
        nulls = df[list(REQUIRED_COLUMNS[name] & set(df.columns))].isna().sum()
        if nulls.any():
            problems.append(f"{name}: nulls {nulls[nulls > 0].to_dict()}")
    f, h = frames["flights"], frames["hotels"]
    if (f["price"] <= 0).any():
        problems.append("flights: non-positive prices")
    if (f["from"] == f["to"]).any():
        problems.append("flights: origin equals destination")
    unknown = set(f["userCode"]) - set(frames["users"]["code"])
    if unknown:
        problems.append(f"flights: {len(unknown)} userCodes not in users")
    bad_total = (h["total"] - h["price"] * h["days"]).abs() > 0.05
    if bad_total.any():
        problems.append(f"hotels: {int(bad_total.sum())} rows where total != price * days")
    if problems:
        raise ValueError("; ".join(problems))
    return {k: len(v) for k, v in frames.items()}


def add_date_features(df):
    """Accepts a frame with a `date` column (str or datetime) and adds calendar features."""
    out = df.copy()
    d = pd.to_datetime(out["date"], format="mixed")
    out["month"] = d.dt.month
    out["day_of_week"] = d.dt.dayofweek
    out["year"] = d.dt.year
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return out


def outbound_legs(flights):
    """First leg of every round trip - the leg that reflects where the user chose to go."""
    return flights.sort_values(["date", "travelCode"]).groupby("travelCode").head(1)


def route_key(origin, destination):
    """Undirected route id, used to hold out whole routes during evaluation."""
    return " | ".join(sorted([origin, destination]))


def first_name(full_name):
    return str(full_name).strip().split(" ")[0].lower()
