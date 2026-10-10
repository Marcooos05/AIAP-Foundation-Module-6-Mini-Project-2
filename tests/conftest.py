import numpy as np
import pandas as pd
import pytest

# TODO: once the real dataset's columns are known, reshape these fixtures to match it.


@pytest.fixture
def base_config():
    """A valid pipeline config with small grids, shared across test modules."""
    return {
        "file_path": "./data/data.csv",
        "target_column": "target",
        "positive_class": "yes",
        "val_test_size": 0.3,
        "validation_size": 0.5,
        "random_state": 42,
        "cv": 2,
        "scoring": "roc_auc",
        "n_jobs": 1,
        "class_weight": None,
        "onehot_min_frequency": 1,
        "columns_to_drop": ["id"],
        "numeric_features": ["num_a", "num_b"],
        "nominal_features": ["cat_a"],
        "ordinal_features": [],
        "ordinal_categories": {"size": ["S", "M", "L"]},
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
    ~200 raw rows with numeric, nominal and ordinal columns, a few missing
    values, and a target that depends on `num_a` so models can learn something.
    """
    rng = np.random.default_rng(0)
    n = 200
    num_a = rng.normal(0, 1, n)
    num_b = rng.uniform(0, 100, n)
    num_b[rng.random(n) < 0.05] = np.nan
    cat_a = rng.choice(["red", "green", "blue"], n).astype(object)
    cat_a[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "num_a": num_a,
        "num_b": num_b,
        "cat_a": cat_a,
        "size": rng.choice(["S", "M", "L"], n),
        "target": np.where(num_a + rng.normal(0, 0.5, n) > 0, "yes", "no"),
    })
