import pytest
from sklearn.linear_model import LogisticRegression

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


@pytest.fixture
def prepared(base_config, synthetic_raw_df):
    data_preparation = DataPreparation(base_config)
    df_cleaned = data_preparation.clean_data(synthetic_raw_df)
    model_training = ModelTraining(base_config, data_preparation.create_preprocessor)
    return model_training, df_cleaned


def test_split_is_stratified(prepared):
    model_training, df_cleaned = prepared
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)

    assert len(X_train) + len(X_val) + len(X_test) == len(df_cleaned)
    assert set(y_train.unique()) == {0, 1}
    for y_split in (y_val, y_test):
        assert y_split.mean() == pytest.approx(y_train.mean(), abs=0.05)


def test_end_to_end_pipeline_runs(base_config, prepared):
    model_training, df_cleaned = prepared
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)

    baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(
        X_train, y_train, X_val, y_val
    )
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    assert set(all_models) == {
        "dummy_most_frequent", "logistic_regression_baseline",
        "logistic_regression", "knn", "decision_tree", "random_forest", "gradient_boosting",
    }
    for metrics in all_metrics.values():
        for key in METRIC_KEYS:
            assert 0.0 <= metrics[key] <= 1.0
        assert metrics["imbalance_strategy"] in {"none", "class_weight", "undersample", "oversample"}
    assert all_metrics["dummy_most_frequent"]["roc_auc"] == pytest.approx(0.5)
    assert all("best_params" in tuned_metrics[name] for name in tuned_models)
    # KNN and Gradient Boosting cannot use class weights.
    assert tuned_metrics["knn"]["imbalance_strategy"] != "class_weight"
    assert tuned_metrics["gradient_boosting"]["imbalance_strategy"] != "class_weight"

    best_name = max(all_metrics, key=lambda name: all_metrics[name]["roc_auc"])
    threshold = model_training.tune_threshold(all_models[best_name], X_val, y_val)
    assert 0.0 < threshold["threshold"] < 1.0
    assert threshold["validation_f1"] >= threshold["validation_f1_at_0.5"]

    refit_model = model_training.refit_on_train_val(all_models[best_name], X_train, X_val, y_train, y_val)
    final_metrics = model_training.evaluate_final_model(
        refit_model, X_test, y_test, best_name, threshold["threshold"],
    )
    assert set(METRIC_KEYS) <= set(final_metrics)
    assert final_metrics["threshold"] == threshold["threshold"]

    importance = model_training.explain_model(refit_model, X_test, y_test, best_name)
    assert set(importance) == {*base_config["numeric_features"], *base_config["nominal_features"]}


@pytest.mark.parametrize("strategy", ["none", "class_weight", "undersample", "oversample"])
def test_each_imbalance_strategy_trains(base_config, synthetic_raw_df, strategy):
    base_config["imbalance_strategies"] = [strategy]
    base_config["param_grids"] = {"logistic_regression": {"classifier__C": [1.0]}}
    data_preparation = DataPreparation(base_config)
    model_training = ModelTraining(base_config, data_preparation.create_preprocessor)
    X_train, X_val, _, y_train, y_val, _ = model_training.split_data(data_preparation.clean_data(synthetic_raw_df))

    grid = model_training._with_imbalance_strategies("logistic_regression", base_config["param_grids"]["logistic_regression"])
    assert len(grid) == 1
    pipeline = model_training._build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.set_params(**{key: values[0] for key, values in grid[0].items()})
    pipeline.fit(X_train, y_train)
    assert pipeline.predict_proba(X_val).shape == (len(X_val), 2)


def test_missing_param_grid_raises(base_config, prepared):
    del base_config["param_grids"]["knn"]
    model_training, df_cleaned = prepared
    X_train, X_val, _, y_train, y_val, _ = model_training.split_data(df_cleaned)

    with pytest.raises(ValueError, match="knn"):
        model_training.train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val)
