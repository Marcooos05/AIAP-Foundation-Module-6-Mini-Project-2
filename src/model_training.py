"""Build, tune and train classification model pipelines, including class-imbalance handling."""

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_preparation import RoomPriceImputer
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Class for classification model training and evaluation.

    Every model is an imbalanced-learn pipeline:
    `room_price_imputer` -> `preprocessor` -> `sampler` -> `classifier`.
    The sampler only runs while fitting, so resampling touches the training
    folds only; validation and test data are never resampled.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for model training.
        preprocessor_factory (Callable[[], ColumnTransformer]): Builds a fresh,
            unfitted preprocessor. Called once per model pipeline so every
            pipeline owns its own independent preprocessor instance.
    """

    TUNED_MODEL_NAMES = ('logistic_regression', 'knn', 'decision_tree', 'random_forest', 'gradient_boosting')

    # Models that accept class_weight; KNN and Gradient Boosting do not.
    CLASS_WEIGHT_MODELS = ('logistic_regression', 'decision_tree', 'random_forest')

    def __init__(self, config: Dict[str, Any], preprocessor_factory: Callable[[], ColumnTransformer]):
        """
        Initializes the ModelTraining class with the given configuration and preprocessor factory.

        Args:
            config (Dict[str, Any]): Configuration dictionary for model training.
            preprocessor_factory (Callable[[], ColumnTransformer]): Callable that
                returns a new, unfitted ColumnTransformer each time it's called.
        """
        self.config = config
        self.preprocessor_factory = preprocessor_factory

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Encodes the target as 1 (positive class) / 0 (every other class) and splits the data into
        training, validation, and testing sets, stratified on the target so
        every split keeps the same class ratio.

        Args:
            df (pd.DataFrame): Input DataFrame containing cleaned data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Training, validation, and testing sets for features and target variable.
        Raises:
            ValueError: If the positive class does not split the target into two classes.
        """
        logger.info("Starting data splitting process.")
        target = self.config['target_column']
        X = df.drop(columns=[target])
        # Compare as strings so `positive_class: "1"` matches an integer 0/1 target column.
        y = (df[target].astype(str) == str(self.config['positive_class'])).astype(int)
        if y.nunique() < 2:
            raise ValueError(
                f"Positive class '{self.config['positive_class']}' must match some, but not all, "
                f"values of '{target}'."
            )

        random_state = self.config['random_state']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config['val_test_size'], random_state=random_state, stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config['validation_size'], random_state=random_state, stratify=y_temp,
        )

        for split_name, y_split in (('train', y_train), ('validation', y_val), ('test', y_test)):
            logger.info(
                "%s set: %d rows, positive class share %.3f", split_name, len(y_split), y_split.mean(),
            )
        logger.info("Data splitting process completed.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Trains and evaluates baseline models (a most-frequent-class dummy
        classifier and a default Logistic Regression, both without imbalance
        handling) on the training and validation sets. They set the floor the
        tuned models must beat.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Trained baseline models and their evaluation metrics on the validation set.
        """
        logger.info("Starting baseline model training and evaluation.")
        models = {
            'dummy_most_frequent': DummyClassifier(strategy='most_frequent'),
            'logistic_regression_baseline': LogisticRegression(max_iter=1000),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            logger.info("Training %s.", model_name)
            pipeline = self._build_pipeline(model)
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline

            logger.info("Evaluating %s on validation set.", model_name)
            metrics[model_name] = self._evaluate_model(pipeline, X_val, y_val, model_name)
            metrics[model_name]['imbalance_strategy'] = 'none'

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
        """
        Tunes Logistic Regression, K-Nearest Neighbours, Decision Tree, Random
        Forest and Gradient Boosting classifiers with GridSearchCV. Each search
        covers the model's hyperparameter grid from the config crossed with the
        configured imbalance strategies, so the best way to handle the class
        imbalance is chosen per model. Each best estimator is then evaluated on
        the validation set.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]: Trained tuned models and their evaluation metrics on the validation set.
        Raises:
            ValueError: If a model has no entry in the config's `param_grids`.
        """
        logger.info("Starting tuned model training and evaluation.")
        tuned_models = {}
        tuned_metrics = {}
        param_grids = self.config['param_grids']
        random_state = self.config['random_state']

        models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
        }

        for model_name, model in models.items():
            if model_name not in param_grids:
                raise ValueError(f"Config 'param_grids' is missing an entry for model '{model_name}'.")

            logger.info("Training %s with hyperparameter tuning.", model_name)
            grid_search = GridSearchCV(
                estimator=self._build_pipeline(model),
                param_grid=self._with_imbalance_strategies(model_name, param_grids[model_name]),
                cv=self.config['cv'],
                scoring=self.config['scoring'],
                n_jobs=self.config['n_jobs'],
            )
            grid_search.fit(X_train, y_train)
            best_params = self._describe_params(grid_search.best_params_)
            logger.info(
                "%s best CV %s: %.4f with %s",
                model_name, self.config['scoring'], grid_search.best_score_, best_params,
            )

            tuned_models[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (Tuned)",
            )
            tuned_metrics[model_name]['imbalance_strategy'] = best_params.pop('imbalance_strategy')
            tuned_metrics[model_name]['best_params'] = best_params
            tuned_metrics[model_name]['cv_score'] = float(grid_search.best_score_)

        logger.info("Tuned model training and evaluation completed.")
        return tuned_models, tuned_metrics

    def tune_threshold(self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Chooses the decision threshold that maximises F1 for the positive class
        on the validation set. The default 0.5 is rarely best for an imbalanced
        target, and resampling or class weights shift the predicted
        probabilities, so the threshold is tuned after model selection.

        Args:
            model (Pipeline): The selected model, fitted on the training split only.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
        Returns:
            Dict[str, float]: The best threshold and its validation F1, plus the F1 at 0.5.
        """
        y_score = model.predict_proba(X_val)[:, 1]
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        f1_scores = [f1_score(y_val, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
        best = int(np.argmax(f1_scores))
        result = {
            'threshold': float(thresholds[best]),
            'validation_f1': float(f1_scores[best]),
            'validation_f1_at_0.5': float(f1_score(y_val, (y_score >= 0.5).astype(int), zero_division=0)),
        }
        logger.info(
            "Tuned threshold %.2f: validation F1 %.4f (F1 at 0.5: %.4f).",
            result['threshold'], result['validation_f1'], result['validation_f1_at_0.5'],
        )
        return result

    def refit_on_train_val(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Refits a clone of the selected model on the combined training and
        validation data, so the final test evaluation reflects a model trained
        on as much data as possible once model/hyperparameter selection is finished.

        Args:
            model (Pipeline): The selected model pipeline (fitted on X_train only).
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series): Training target variable.
            y_val (pd.Series): Validation target variable.
        Returns:
            Pipeline: A new pipeline, refit on the combined train+val data.
        """
        logger.info("Refitting selected model on combined train+val data.")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        refit_model = clone(model)
        refit_model.fit(X_train_val, y_train_val)
        return refit_model

    def evaluate_final_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluates the final selected model on the testing set.

        Args:
            model (Pipeline): The final selected model pipeline.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target variable.
            model_name (str): Name of the model being evaluated.
            threshold (float): Decision threshold on the positive-class probability.
        Returns:
            Dict[str, Any]: Evaluation metrics for the final model on the testing set.
        """
        logger.info("Evaluating final model on testing set (threshold %.2f).", threshold)
        return self._evaluate_model(model, X_test, y_test, model_name, threshold)

    def explain_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimates feature importance by permutation: each configured input
        feature is shuffled in turn and the drop in the configured score
        (`scoring`) is measured. Works for every model type (including KNN) and
        reports importance per original feature rather than per one-hot column,
        so it can be compared directly with the EDA's feature ranking.

        Args:
            model (Pipeline): The final fitted model pipeline.
            X (pd.DataFrame): Held-out features to measure importance on.
            y (pd.Series): Held-out target variable.
            model_name (str): Name of the model being explained.
        Returns:
            Dict[str, Dict[str, float]]: Feature name -> {'mean', 'std'} of the
                score drop, sorted by descending mean.
        """
        features = [
            *self.config['numeric_features'],
            *self.config['nominal_features'],
            *self.config['ordinal_features'],
        ]
        result = permutation_importance(
            model, X[features], y,
            scoring=self.config['scoring'],
            n_repeats=5,
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs'],
        )
        importance = {
            feature: {'mean': float(mean), 'std': float(std)}
            for feature, mean, std in zip(features, result.importances_mean, result.importances_std)
        }
        importance = dict(sorted(importance.items(), key=lambda item: item[1]['mean'], reverse=True))
        logger.info(
            "Permutation importance for %s: %s",
            model_name, {feature: round(values['mean'], 4) for feature, values in importance.items()},
        )
        return importance

    def _build_pipeline(self, model: Any) -> Pipeline:
        """Wraps a classifier with its own imputer, freshly built preprocessor and a (default: no-op) sampler."""
        return Pipeline(steps=[
            ('room_price_imputer', RoomPriceImputer()),
            ('preprocessor', self.preprocessor_factory()),
            ('sampler', 'passthrough'),
            ('classifier', model),
        ])

    def _with_imbalance_strategies(self, model_name: str, grid: Dict[str, List[Any]]) -> List[Dict[str, List[Any]]]:
        """
        Expands a hyperparameter grid into one sub-grid per configured
        imbalance strategy, so GridSearchCV compares the strategies as well:

        - none: no sampler, no class weights.
        - class_weight: class_weight='balanced' (only for models that support it).
        - undersample / oversample: a random sampler on the training folds.
        """
        random_state = self.config['random_state']
        samplers = {
            'none': 'passthrough',
            'undersample': RandomUnderSampler(random_state=random_state),
            'oversample': RandomOverSampler(random_state=random_state),
        }
        sub_grids = []
        for strategy in self.config['imbalance_strategies']:
            if strategy == 'class_weight':
                if model_name in self.CLASS_WEIGHT_MODELS:
                    sub_grids.append({**grid, 'sampler': ['passthrough'], 'classifier__class_weight': ['balanced']})
            else:
                sub_grid = {**grid, 'sampler': [samplers[strategy]]}
                if model_name in self.CLASS_WEIGHT_MODELS:
                    sub_grid['classifier__class_weight'] = [None]
                sub_grids.append(sub_grid)
        return sub_grids

    @staticmethod
    def _describe_params(best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Replaces the sampler object in GridSearchCV's best_params_ with a readable imbalance strategy name."""
        params = {key: value for key, value in best_params.items() if key != 'sampler'}
        sampler = best_params.get('sampler', 'passthrough')
        if isinstance(sampler, RandomUnderSampler):
            strategy = 'undersample'
        elif isinstance(sampler, RandomOverSampler):
            strategy = 'oversample'
        elif params.get('classifier__class_weight') == 'balanced':
            strategy = 'class_weight'
        else:
            strategy = 'none'
        params['imbalance_strategy'] = strategy
        return params

    def _evaluate_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluates a given model and calculates classification metrics.

        Args:
            model (Pipeline): The model pipeline to be evaluated.
            X (pd.DataFrame): Evaluation features.
            y (pd.Series): Evaluation target variable.
            model_name (str): Name of the model being evaluated.
            threshold (float): Decision threshold on the positive-class probability.
        Returns:
            Dict[str, Any]: Threshold, accuracy, precision, recall, F1, ROC-AUC, average precision and confusion matrix.
        """
        metrics = compute_metrics(model, X, y, threshold)
        logger.info(
            "Evaluation metrics for %s: %s",
            model_name, {k: round(v, 4) for k, v in metrics.items() if k != 'confusion_matrix'},
        )
        if metrics['roc_auc'] < 0.5:
            logger.warning(
                "%s has ROC-AUC %.4f, i.e. it ranks worse than random guessing.", model_name, metrics['roc_auc'],
            )
        return metrics
