"""Validation for the pipeline's YAML configuration."""

from typing import Any, Dict

REQUIRED_KEYS = {
    "file_path": str,
    "target_column": str,
    "positive_class": (str, int),
    "val_test_size": float,
    "validation_size": float,
    "random_state": int,
    "param_grids": dict,
    "cv": int,
    "scoring": str,
    "n_jobs": int,
    "usd_to_sgd": (int, float),
    "onehot_min_frequency": int,
    "imbalance_strategies": list,
    "columns_to_drop": list,
    "numeric_features": list,
    "nominal_features": list,
    "ordinal_features": list,
    "ordinal_categories": dict,
}

FEATURE_ROLE_KEYS = ("numeric_features", "nominal_features", "ordinal_features")

# `scoring` is used both by GridSearchCV and to pick the best model from the
# validation metrics, so it must be a metric the pipeline computes.
SUPPORTED_SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")

SUPPORTED_IMBALANCE_STRATEGIES = ("none", "class_weight", "undersample", "oversample")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the pipeline configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
    Raises:
        ValueError: If a required key is missing, has the wrong type, or an
            out-of-range value, or if the feature roles are inconsistent.
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")

    for key, expected_type in REQUIRED_KEYS.items():
        value = config[key]
        # bool is a subclass of int, so reject it explicitly.
        if not isinstance(value, expected_type) or isinstance(value, bool):
            types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
            raise ValueError(
                f"Config key '{key}' must be of type {' or '.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__}."
            )

    for key in ("val_test_size", "validation_size"):
        value = config[key]
        if not 0.0 < value < 1.0:
            raise ValueError(f"Config key '{key}' must be in (0, 1), got {value}.")

    if config["cv"] < 2:
        raise ValueError(f"Config key 'cv' must be >= 2, got {config['cv']}.")

    if config["scoring"] not in SUPPORTED_SCORING:
        raise ValueError(
            f"Config key 'scoring' must be one of {list(SUPPORTED_SCORING)}, got '{config['scoring']}'."
        )

    if config["n_jobs"] == 0 or config["n_jobs"] < -1:
        raise ValueError(f"Config key 'n_jobs' must be -1 (all cores) or >= 1, got {config['n_jobs']}.")

    if config["usd_to_sgd"] <= 0:
        raise ValueError(f"Config key 'usd_to_sgd' must be > 0, got {config['usd_to_sgd']}.")

    strategies = config["imbalance_strategies"]
    if not strategies or any(s not in SUPPORTED_IMBALANCE_STRATEGIES for s in strategies):
        raise ValueError(
            f"Config key 'imbalance_strategies' must be a non-empty list from "
            f"{list(SUPPORTED_IMBALANCE_STRATEGIES)}, got {strategies}."
        )

    if config["onehot_min_frequency"] < 1:
        raise ValueError(
            f"Config key 'onehot_min_frequency' must be >= 1, got {config['onehot_min_frequency']}."
        )

    for key in ("columns_to_drop", *FEATURE_ROLE_KEYS):
        if not all(isinstance(item, str) for item in config[key]):
            raise ValueError(f"Config key '{key}' must be a list of strings.")

    seen_roles: Dict[str, str] = {}
    for key in FEATURE_ROLE_KEYS:
        for feature in config[key]:
            if feature in seen_roles:
                raise ValueError(
                    f"Feature '{feature}' is listed in both '{seen_roles[feature]}' and '{key}'."
                )
            seen_roles[feature] = key

    if not seen_roles:
        raise ValueError("Config must list at least one feature in numeric/nominal/ordinal_features.")

    dropped_features = sorted(set(seen_roles) & set(config["columns_to_drop"]))
    if dropped_features:
        raise ValueError(f"Feature(s) {dropped_features} are also listed in 'columns_to_drop'.")

    target = config["target_column"]
    if target in seen_roles or target in config["columns_to_drop"]:
        raise ValueError(f"Target column '{target}' must not be a feature or in 'columns_to_drop'.")

    for feature in config["ordinal_features"]:
        categories = config["ordinal_categories"].get(feature)
        if not isinstance(categories, list) or not categories:
            raise ValueError(
                f"Ordinal feature '{feature}' needs a non-empty category list in 'ordinal_categories'."
            )

    for model_name, grid in config["param_grids"].items():
        if not isinstance(grid, dict):
            raise ValueError(
                f"Config key 'param_grids.{model_name}' must be a dict of hyperparameter lists."
            )
