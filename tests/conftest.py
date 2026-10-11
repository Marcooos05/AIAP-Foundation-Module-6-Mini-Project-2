import numpy as np
import pandas as pd
import pytest

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
ROOM_BANDS_SGD = {  # (branch, room) -> (min, max), as found in EDA §4.1
    ("Changi", "Single"): (400, 600), ("Changi", "Queen"): (600, 800), ("Changi", "King"): (800, 1000),
    ("Orchard", "Single"): (600, 900), ("Orchard", "Queen"): (900, 1200), ("Orchard", "King"): (1200, 1500),
}


@pytest.fixture
def base_config():
    """A valid pipeline config with small grids, shared across test modules."""
    return {
        "file_path": "./data/mini_project_2_data.csv",
        "target_column": "no_show",
        "positive_class": "1",
        "val_test_size": 0.3,
        "validation_size": 0.5,
        "random_state": 42,
        "cv": 2,
        "scoring": "roc_auc",
        "n_jobs": 1,
        "usd_to_sgd": 1.37,
        "onehot_min_frequency": 1,
        "imbalance_strategies": ["none", "class_weight", "undersample", "oversample"],
        "columns_to_drop": ["booking_id"],
        "numeric_features": ["lead_time_months", "stay_nights", "price_sgd", "num_adults", "num_children"],
        "nominal_features": ["country", "branch", "first_time", "arrival_month", "room"],
        "ordinal_features": [],
        "ordinal_categories": {},
        "param_grids": {
            "logistic_regression": {"classifier__C": [1.0]},
            "knn": {"classifier__n_neighbors": [5]},
            "decision_tree": {"classifier__max_depth": [3]},
            "random_forest": {"classifier__n_estimators": [10], "classifier__max_depth": [3]},
            "gradient_boosting": {"classifier__n_estimators": [10], "classifier__max_depth": [2]},
        },
    }


@pytest.fixture
def synthetic_raw_df():
    """
    ~300 raw rows shaped like mini_project_2_data.csv, with the same data-quality
    issues (mixed-case months, word-form adults, negative checkout days, SGD / USD
    prices, missing room or price, one empty row) and a target that depends on
    country so models can learn something.
    """
    rng = np.random.default_rng(0)
    n = 300
    branch = rng.choice(["Changi", "Orchard"], n)
    room = rng.choice(["Single", "Queen", "King"], n)
    arrival = pd.Timestamp("2000-01-01") + pd.to_timedelta(rng.integers(0, 366, n), unit="D")
    checkout = arrival + pd.to_timedelta(rng.integers(1, 6, n), unit="D")
    booking_month_idx = (arrival.month.values - 1 - rng.integers(0, 12, n)) % 12
    country = rng.choice(["China", "Japan", "India"], n)
    price_sgd = np.array([rng.uniform(*ROOM_BANDS_SGD[(b, r)]) for b, r in zip(branch, room)])
    currency_usd = rng.random(n) < 0.5
    price = np.where(
        currency_usd,
        [f"USD$ {p / 1.37:.2f}" for p in price_sgd],
        [f"SGD$ {p:.2f}" for p in price_sgd],
    ).astype(object)

    room = room.astype(object)
    missing = rng.random(n)
    room[missing < 0.1] = np.nan                       # room missing, price known
    price[(missing >= 0.1) & (missing < 0.2)] = np.nan  # price missing, room known

    checkout_day = checkout.day.values.astype(float)
    checkout_day[rng.random(n) < 0.1] *= -1            # sign error (EDA §6.3)
    arrival_month = np.array([MONTHS[m - 1] for m in arrival.month])
    arrival_month[::7] = np.char.upper(arrival_month[::7].astype(str))  # casing variants (EDA §6.1)
    no_show = (rng.random(n) < np.where(country == "China", 0.7, 0.2)).astype(float)

    df = pd.DataFrame({
        "booking_id": np.arange(1, n + 1),
        "no_show": no_show,
        "branch": branch,
        "booking_month": [MONTHS[m] for m in booking_month_idx],
        "arrival_month": arrival_month,
        "arrival_day": arrival.day.values.astype(float),
        "checkout_month": [MONTHS[m - 1] for m in checkout.month],
        "checkout_day": checkout_day,
        "country": country,
        "first_time": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "room": room,
        "price": price,
        "platform": rng.choice(["Website", "Email", "Agent", "Phone"], n),
        "num_adults": rng.choice(["1", "2", "one", "two"], n),
        "num_children": rng.integers(0, 3, n).astype(float),
    })
    empty_row = {col: np.nan for col in df.columns} | {"booking_id": n + 1}
    return pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
