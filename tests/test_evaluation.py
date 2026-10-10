import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import (
    classification_report_dict,
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)


class FixedPredictionModel:
    """Stand-in classifier returning preset predictions, so expected metrics are known exactly."""

    def __init__(self, y_pred, y_score):
        self.y_pred = np.asarray(y_pred)
        self.y_score = np.asarray(y_score)

    def predict(self, X):
        return self.y_pred

    def predict_proba(self, X):
        return np.column_stack([1 - self.y_score, self.y_score])


@pytest.fixture
def known_case():
    # y_true: 3 positives, 3 negatives. Predictions: TP=2, FN=1, FP=1, TN=2.
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    model = FixedPredictionModel(
        y_pred=[1, 1, 0, 1, 0, 0],
        y_score=[0.9, 0.8, 0.4, 0.6, 0.3, 0.1],
    )
    X = pd.DataFrame({"x": range(6)})
    return model, X, y_true


@pytest.fixture
def fitted_pipeline():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    y = pd.Series((X["x1"] + rng.normal(0, 0.5, 60) > 0).astype(int))
    model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())]).fit(X, y)
    return model, X, y


def test_compute_metrics_returns_expected_values(known_case):
    model, X, y = known_case
    metrics = compute_metrics(model, X, y)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(8 / 9)  # 8 of 9 positive/negative pairs ranked correctly
    assert metrics["average_precision"] == pytest.approx((1 + 1 + 3 / 4) / 3)  # precision at each positive
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]


def test_compute_metrics_handles_no_positive_predictions():
    y = pd.Series([1, 0, 1, 0])
    model = FixedPredictionModel(y_pred=[0, 0, 0, 0], y_score=[0.5, 0.5, 0.5, 0.5])
    metrics = compute_metrics(model, pd.DataFrame({"x": range(4)}), y)
    assert metrics["precision"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_classification_report_uses_target_names(known_case):
    model, X, y = known_case
    report = classification_report_dict(model, X, y, ["Negative", "Positive"])
    assert report["Positive"]["recall"] == pytest.approx(2 / 3)
    assert report["Negative"]["support"] == 3


def test_plot_roc_curves_writes_png(fitted_pipeline, tmp_path):
    model, X, y = fitted_pipeline
    path = tmp_path / "figures" / "roc.png"
    plot_roc_curves({"model_a": model, "model_b": model}, X, y, path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_precision_recall_curves_writes_png(fitted_pipeline, tmp_path):
    model, X, y = fitted_pipeline
    path = tmp_path / "figures" / "pr.png"
    plot_precision_recall_curves({"model_a": model, "model_b": model}, X, y, path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_confusion_matrix_writes_png(fitted_pipeline, tmp_path):
    model, X, y = fitted_pipeline
    path = tmp_path / "cm.png"
    plot_confusion_matrix(model, X, y, path, labels=["Negative", "Positive"])
    assert path.exists() and path.stat().st_size > 0
