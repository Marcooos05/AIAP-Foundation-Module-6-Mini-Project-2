import pytest

from src.config_validation import validate_config


def test_valid_config_passes(base_config):
    validate_config(base_config)


def test_missing_required_key_raises(base_config):
    del base_config["cv"]
    with pytest.raises(ValueError, match="missing required key"):
        validate_config(base_config)


def test_wrong_type_raises(base_config):
    base_config["random_state"] = "42"
    with pytest.raises(ValueError, match="must be of type int"):
        validate_config(base_config)


def test_integer_positive_class_passes(base_config):
    base_config["positive_class"] = 1
    validate_config(base_config)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_split_size_out_of_range_raises(base_config, value):
    base_config["val_test_size"] = value
    with pytest.raises(ValueError, match="val_test_size"):
        validate_config(base_config)


def test_cv_below_minimum_raises(base_config):
    base_config["cv"] = 1
    with pytest.raises(ValueError, match="cv"):
        validate_config(base_config)


def test_unsupported_scoring_raises(base_config):
    base_config["scoring"] = "r2"
    with pytest.raises(ValueError, match="scoring"):
        validate_config(base_config)


@pytest.mark.parametrize("value", [0, -2])
def test_invalid_n_jobs_raises(base_config, value):
    base_config["n_jobs"] = value
    with pytest.raises(ValueError, match="n_jobs"):
        validate_config(base_config)


def test_invalid_class_weight_raises(base_config):
    base_config["class_weight"] = "balanced_subsample"
    with pytest.raises(ValueError, match="class_weight"):
        validate_config(base_config)


def test_onehot_min_frequency_below_one_raises(base_config):
    base_config["onehot_min_frequency"] = 0
    with pytest.raises(ValueError, match="onehot_min_frequency"):
        validate_config(base_config)


def test_feature_list_with_non_string_raises(base_config):
    base_config["numeric_features"] = ["num_a", 123]
    with pytest.raises(ValueError, match="numeric_features"):
        validate_config(base_config)


def test_no_features_raises(base_config):
    base_config["numeric_features"] = []
    base_config["nominal_features"] = []
    with pytest.raises(ValueError, match="at least one feature"):
        validate_config(base_config)


def test_feature_in_two_roles_raises(base_config):
    base_config["ordinal_features"] = ["cat_a"]
    base_config["ordinal_categories"]["cat_a"] = ["red"]
    with pytest.raises(ValueError, match="listed in both"):
        validate_config(base_config)


def test_dropped_feature_raises(base_config):
    base_config["columns_to_drop"] = ["num_a"]
    with pytest.raises(ValueError, match="columns_to_drop"):
        validate_config(base_config)


def test_target_as_feature_raises(base_config):
    base_config["numeric_features"].append("target")
    with pytest.raises(ValueError, match="Target column"):
        validate_config(base_config)


def test_ordinal_feature_without_categories_raises(base_config):
    base_config["ordinal_features"] = ["grade"]
    with pytest.raises(ValueError, match="ordinal_categories"):
        validate_config(base_config)


def test_ordinal_feature_with_categories_passes(base_config):
    base_config["ordinal_features"] = ["size"]
    validate_config(base_config)


def test_malformed_param_grid_raises(base_config):
    base_config["param_grids"] = {"knn": ["not", "a", "dict"]}
    with pytest.raises(ValueError, match="param_grids"):
        validate_config(base_config)
