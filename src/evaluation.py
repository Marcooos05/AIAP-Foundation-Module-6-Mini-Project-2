"""Classification metrics and evaluation plots for fitted model pipelines."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')  # render to files only, so the pipeline runs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def predict_with_threshold(model: Pipeline, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Predicts the positive class (1) when its predicted probability is at least `threshold`.

    Args:
        model (Pipeline): A fitted pipeline exposing `predict_proba`.
        X (pd.DataFrame): Features.
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        np.ndarray: Predicted labels (0/1).
    """
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes classification metrics for a fitted binary classifier.

    Precision, recall and F1 are reported for the positive class (1), using
    `threshold` on the predicted probability. ROC-AUC and average precision
    (area under the precision-recall curve) are computed from predicted
    probabilities, so they do not depend on the threshold. Average precision
    is the more informative of the two when the positive class is rare.

    Args:
        model (Pipeline): A fitted pipeline exposing `predict_proba`.
        X (pd.DataFrame): Features.
        y (pd.Series): True binary labels (0/1).
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        Dict[str, Any]: threshold, accuracy, precision, recall, f1, roc_auc,
            average_precision, and the confusion matrix as a nested list
            [[TN, FP], [FN, TP]].
    """
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_score)),
        'average_precision': float(average_precision_score(y, y_score)),
        'confusion_matrix': confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }


def classification_report_dict(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    target_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Builds scikit-learn's per-class classification report as a dictionary.

    Args:
        model (Pipeline): A fitted classifier pipeline.
        X (pd.DataFrame): Features.
        y (pd.Series): True binary labels (0/1).
        target_names (List[str]): Display names for classes 0 and 1.
        threshold (float): Decision threshold on the positive-class probability.
    Returns:
        Dict[str, Any]: Precision, recall, F1 and support per class, plus averages.
    """
    return classification_report(
        y, predict_with_threshold(model, X, threshold), labels=[0, 1], target_names=target_names,
        output_dict=True, zero_division=0,
    )


def plot_model_comparison(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    """
    Plots every model's validation ROC-AUC, average precision and F1 side by side and saves the figure.

    Args:
        metrics (Dict[str, Dict[str, Any]]): Model name -> validation metrics.
        path (Path): Output image path.
    """
    scores = pd.DataFrame(metrics).T[['roc_auc', 'average_precision', 'f1']].astype(float)
    scores = scores.sort_values('roc_auc')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (column, title) in zip(axes, [('roc_auc', 'ROC-AUC'), ('average_precision', 'PR-AUC'), ('f1', 'F1 (threshold 0.5)')]):
        ax.barh(scores.index, scores[column])
        for y_pos, value in enumerate(scores[column]):
            ax.text(value, y_pos, f' {value:.3f}', va='center', fontsize='small')
        ax.set_title(f'{title} (validation)')
        ax.set_xlim(0, 1)
    _save_figure(fig, path)


def plot_roc_curves(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """
    Plots the ROC curves of one or more fitted models on a shared axes and saves the figure.

    Args:
        models (Dict[str, Pipeline]): Model name -> fitted classifier pipeline.
        X (pd.DataFrame): Features to evaluate on.
        y (pd.Series): True binary labels (0/1).
        path (Path): Output image path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (model_name, model) in enumerate(models.items()):
        RocCurveDisplay.from_estimator(
            model, X, y, name=model_name, ax=ax, plot_chance_level=(i == len(models) - 1),
        )
    ax.set_title('ROC curves')
    ax.legend(loc='lower right', fontsize='small')
    _save_figure(fig, path)


def plot_precision_recall_curves(models: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """
    Plots the precision-recall curves of one or more fitted models on a shared axes and saves the figure.

    Args:
        models (Dict[str, Pipeline]): Model name -> fitted classifier pipeline.
        X (pd.DataFrame): Features to evaluate on.
        y (pd.Series): True binary labels (0/1).
        path (Path): Output image path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (model_name, model) in enumerate(models.items()):
        PrecisionRecallDisplay.from_estimator(
            model, X, y, name=model_name, ax=ax, plot_chance_level=(i == len(models) - 1),
        )
    ax.set_title('Precision-recall curves')
    ax.legend(loc='upper right', fontsize='small')
    _save_figure(fig, path)


def plot_confusion_matrix(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
    labels: List[str],
    title: str = 'Confusion matrix',
    threshold: float = 0.5,
) -> None:
    """
    Plots the confusion matrix of a fitted model and saves the figure.

    Args:
        model (Pipeline): A fitted classifier pipeline.
        X (pd.DataFrame): Features to evaluate on.
        y (pd.Series): True binary labels (0/1).
        path (Path): Output image path.
        labels (List[str]): Display names for classes 0 and 1.
        title (str): Figure title.
        threshold (float): Decision threshold on the positive-class probability.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, predict_with_threshold(model, X, threshold), labels=[0, 1], display_labels=labels,
        cmap='Blues', values_format='d', ax=ax,
    )
    ax.set_title(title)
    _save_figure(fig, path)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure to %s", path)
