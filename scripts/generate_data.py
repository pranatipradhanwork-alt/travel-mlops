"""Generate synthetic users / flights / hotels CSVs that match the capstone schema.

WARNING: overwrites data/*.csv. Use this only when the real datasets are not available
(e.g. a quick demo). If you have the real
files, copy them to data/users.csv, data/flights.csv and data/hotels.csv and
skip this script.

    python scripts/generate_data.py --users 1340 --trips 30000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR  # noqa: E402

CITIES = {
    # city: (lat, lon)
    "Florianopolis (SC)": (-27.59, -48.55),
    "Sao Paulo (SP)": (-23.55, -46.63),
    "Salvador (BH)": (-12.97, -38.50),
    "Brasilia (DF)": (-15.79, -47.88),
    "Recife (PE)": (-8.05, -34.88),
    "Natal (RN)": (-5.79, -35.21),
    "Rio de Janeiro (RJ)": (-22.91, -43.17),
    "Campo Grande (MS)": (-20.44, -54.65),
    "Aracaju (SE)": (-10.91, -37.07),
}
FLIGHT_TYPES = {"economic": 1.0, "premium": 1.55, "firstClass": 2.4}
AGENCIES = {"Rainbow": 0.95, "CloudFy": 1.0, "FlyingDrops": 1.1}
COMPANIES = ["4You", "Acme Factory", "Wonka Company", "Monsters CYA", "Umbrella LTDA"]
HOTELS = {
    # place: list of (hotel name, base nightly price)
    city: [(f"Hotel {chr(65 + i)}", float(p)) for i, p in enumerate(prices)]
    for city, prices in zip(
        CITIES,
        [
            (180, 250, 320), (210, 290, 410), (140, 220, 300), (160, 240, 330),
            (130, 190, 260), (120, 175, 245), (230, 310, 440), (110, 165, 230),
            (100, 150, 215),
        ],
    )
}
MALE = ("Roy Charles Jose Paulo Carlos Daniel Pedro Lucas Mateus Rafael Bruno Gabriel "
        "Thiago Felipe Joao Marcos Andre Eduardo Ricardo Gustavo Leonardo Samuel Victor "
        "Henrique Diego Fernando Rodrigo Antonio Luiz Miguel Arthur David").split()
FEMALE = ("Ana Maria Juliana Fernanda Patricia Aline Camila Amanda Bruna Leticia Mariana "
          "Beatriz Larissa Gabriela Carolina Rafaela Vanessa Natalia Isabela Luana Helena "
          "Sofia Laura Alice Valentina Julia Manuela Clara Lorena Tatiane Debora Renata").split()
UNISEX = "Alex Sam Jordan Kim Ariel Dani Rene Cris".split()
SURNAMES = ("Silva Santos Oliveira Souza Lima Pereira Costa Ferreira Almeida Ribeiro "
            "Carvalho Gomes Martins Rocha Barbosa Araujo Melo Cardoso Teixeira Dias").split()


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (*a, *b))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def make_users(n, rng):
    rows = []
    for code in range(n):
        g = rng.choice(["male", "female", "none"], p=[0.48, 0.48, 0.04])
        pool = MALE if g == "male" else FEMALE if g == "female" else UNISEX
        # A small share of names are ambiguous to keep the classifier honest.
        if rng.random() < 0.05:
            pool = UNISEX
        name = f"{rng.choice(pool)} {rng.choice(SURNAMES)}"
        rows.append(dict(code=code, company=rng.choice(COMPANIES), name=name,
                         gender=g, age=int(rng.integers(21, 66))))
    return pd.DataFrame(rows)


def make_trips(users, n_trips, rng):
    cities = list(CITIES)
    # Each user has a home city and a few favourite destinations -> learnable preferences.
    home = {u: rng.choice(cities) for u in users.code}
    favs = {u: rng.choice([c for c in cities if c != home[u]], size=3, replace=False) for u in users.code}
    hotel_taste = {u: rng.choice(["budget", "mid", "luxury"], p=[0.4, 0.4, 0.2]) for u in users.code}
    tier_idx = {"budget": 0, "mid": 1, "luxury": 2}
    age = users.set_index("code").age.to_dict()

    flights, hotels = [], []
    start = pd.Timestamp("2019-09-26")
    for travel_code in range(n_trips):
        u = int(rng.choice(users.code))
        dest = rng.choice(favs[u]) if rng.random() < 0.75 else rng.choice([c for c in cities if c != home[u]])
        date = start + pd.Timedelta(days=int(rng.integers(0, 1460)))
        older = age[u] > 45
        ftype = rng.choice(list(FLIGHT_TYPES), p=[0.35, 0.35, 0.30] if older else [0.5, 0.3, 0.2])
        agency = rng.choice(list(AGENCIES))
        dist = haversine_km(CITIES[home[u]], CITIES[dest]) * 1.08
        hours = dist / 680 + 0.45
        seasonal = 1.12 if date.month in (1, 7, 12) else 1.0
        price = (180 + 0.42 * dist) * FLIGHT_TYPES[ftype] * AGENCIES[agency] * seasonal
        price *= rng.normal(1, 0.03)
        flights.append(dict(travelCode=travel_code, userCode=u, **{"from": home[u], "to": dest},
                            flightType=ftype, price=round(price, 2), time=round(hours, 2),
                            distance=round(dist, 2), agency=agency, date=date.strftime("%m/%d/%Y")))
        if rng.random() < 0.7:  # not every trip books a hotel
            taste = tier_idx[hotel_taste[u]]
            idx = taste if rng.random() < 0.7 else int(rng.integers(0, 3))
            hname, base = HOTELS[dest][idx]
            days = int(rng.integers(1, 5))
            nightly = round(base * seasonal, 2)
            hotels.append(dict(travelCode=travel_code, userCode=u, name=hname, place=dest,
                               days=days, price=nightly, total=round(nightly * days, 2),
                               date=date.strftime("%m/%d/%Y")))
    return pd.DataFrame(flights), pd.DataFrame(hotels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=1340)
    ap.add_argument("--trips", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    users = make_users(args.users, rng)
    flights, hotels = make_trips(users, args.trips, rng)
    users.to_csv(DATA_DIR / "users.csv", index=False)
    flights.to_csv(DATA_DIR / "flights.csv", index=False)
    hotels.to_csv(DATA_DIR / "hotels.csv", index=False)
    print(f"users={len(users)} flights={len(flights)} hotels={len(hotels)} -> {DATA_DIR}")


if __name__ == "__main__":
    main()
