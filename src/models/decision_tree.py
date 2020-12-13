from typing import Dict, Optional, List
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold, KFold

from algorithms.greedy_search import GreedySearch
from algorithms.pruner import Pruner
from metrics.classification_metrics import CLASSIFICATION_CRITERION_LOOKUP
from metrics.regression_metrics import REGRESSION_METRICS_LOOKUP
from visualization.tree_graph import TreeGraph

logger = logging.getLogger(__name__)


class DecisionTree:

    def __init__(
            self,
            is_classifier: bool,
            max_depth: int = 5,
            min_parent_size: int = 10,
            training_criterion: str = "gini",
            validation_criterion: str = "roc_auc",
            n_folds: int = 5,
            class_weights: Optional[Dict[int, float]] = None,
            use_balanced_class_weights: bool = False,
            refit: bool = True,
            verbose: bool = False
    ):
        self.is_classifier = is_classifier
        self.max_depth = max_depth
        self.min_parent_size = min_parent_size
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion
        self.n_folds = n_folds
        self.class_weights = class_weights
        self.use_balanced_class_weights = use_balanced_class_weights
        self.verbose = verbose
        self.refit = refit
        self.training_criterion_callback_ = None
        self.validation_criterion_callback_ = None
        self.complexity_param_cv_scores_ = None
        self.complexity_param_grid_ = None
        self.complexity_param_ = None
        self.score_ = None
        self.tree_ = None

    def _assign_criterion_callbacks(self) -> None:
        if self.is_classifier:
            self.training_criterion_callback_ = CLASSIFICATION_CRITERION_LOOKUP[self.training_criterion]
            self.validation_criterion_callback_ = CLASSIFICATION_CRITERION_LOOKUP[self.validation_criterion]
        else:
            self.training_criterion_callback_ = REGRESSION_METRICS_LOOKUP[self.training_criterion]
            self.validation_criterion_callback_ = REGRESSION_METRICS_LOOKUP[self.validation_criterion]

    def _assign_sample_weights(self, targets: np.ndarray) -> np.ndarray:
        if self.use_balanced_class_weights:
            self.class_weights = {k: 1 / (targets == k).mean() for k in np.unique(targets)}
        if self.class_weights:
            return np.array([self.class_weights[k] for k in targets])
        return np.ones(targets.shape[0])

    def _initialize_greedy_search(self) -> GreedySearch:
        return GreedySearch(
            is_classifier=self.is_classifier,
            criterion=self.training_criterion_callback_,
            is_soft=False
        )

    def _fit_and_prune_tree(
            self,
            train_inputs: np.ndarray,
            train_targets: np.ndarray,
            train_sample_weights: np.ndarray,
            validation_inputs: Optional[np.ndarray] = None,
            validation_targets: Optional[np.ndarray] = None,
            validation_sample_weights: Optional[np.ndarray] = None,
    ) -> Pruner:

        search = self._initialize_greedy_search()
        tree = search.fit(
            inputs=train_inputs,
            targets=train_targets,
            sample_weights=train_sample_weights
        )
        pruner = Pruner(
            base_tree=tree,
            criterion=self.validation_criterion_callback_
        )
        pruner.fit(
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_sample_weights=train_sample_weights,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            validation_sample_weights=validation_sample_weights
        )
        return pruner

    def _select_complexity_parameter(
            self,
            validation_scores: List[np.ndarray],
            critical_values: List[np.ndarray]
    ) -> None:

        # Initialize grids
        self.complexity_param_grid_ = np.unique(np.concatenate(critical_values))
        self.complexity_param_cv_scores_ = np.zeros((self.complexity_param_grid_.shape[0], self.n_folds))

        # Perform interpolation
        for fold in range(self.n_folds):
            self.complexity_param_cv_scores_[:, fold] = interp1d(
                x=critical_values[fold],
                y=validation_scores[fold],
                kind="nearest",
                fill_value="extrapolate"
            )(self.complexity_param_grid_)

        # Select complexity param based on conservative estimate relative to argmax
        argmax = self.complexity_param_cv_scores_.mean(axis=1).argmax()
        round_tol = int(np.ceil(-np.log10(self.complexity_param_cv_scores_[argmax, :].std() / self.n_folds)))
        selected = np.flip(np.round(self.complexity_param_cv_scores_.mean(axis=1), round_tol)).argmax()
        self.complexity_param_ = np.flip(self.complexity_param_grid_)[selected]
        self.score_ = np.flip(self.complexity_param_cv_scores_, axis=0)[selected].mean()

    def _fit_cv(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> None:

        critical_values = []
        validation_scores = []
        for i, (train, val) in enumerate(
                (StratifiedKFold if self.is_classifier else KFold)(n_splits=self.n_folds).split(inputs, targets)
        ):
            pruner = self._fit_and_prune_tree(
                train_inputs=inputs[train],
                train_targets=targets[train],
                train_sample_weights=sample_weights[train],
                validation_inputs=inputs[val],
                validation_targets=targets[val],
                validation_sample_weights=sample_weights[val]
            )
            critical_values.append(pruner.critical_values_)
            validation_scores.append(pruner.validation_scores_)
            if self.verbose:
                logger.info(f"Completed CV fold {i + 1}/{self.n_folds}")

        self._select_complexity_parameter(
            critical_values=critical_values,
            validation_scores=validation_scores
        )
        if not self.refit:
            self.tree_ = pruner.pruned_trees_[np.argmin(np.abs(pruner.critical_values_ - self.complexity_param_))]

    def _refit_tree(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> None:
        pruner = self._fit_and_prune_tree(
            train_inputs=inputs,
            train_targets=targets,
            train_sample_weights=sample_weights
        )
        self.tree_ = pruner.pruned_trees_[np.argmin(np.abs(pruner.critical_values_ - self.complexity_param_))]

    def fit(self, inputs: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:

        # Assign criterion callback
        self._assign_criterion_callbacks()

        # Assign sample weights based on class weights
        if sample_weights is None:
            sample_weights = self._assign_sample_weights(targets=targets)
        else:
            if self.class_weights or self.use_balanced_class_weights:
                logger.warning(f"Overriding class weights with provided sample weights")

        # Perform cross validation
        self._fit_cv(
            inputs=inputs,
            targets=targets,
            sample_weights=sample_weights
        )

        # Refit on all data with selected complexity parameter
        if self.refit:
            if self.verbose:
                logger.info("Refitting model")
            self._refit_tree(
                inputs=inputs,
                targets=targets,
                sample_weights=sample_weights
            )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.tree_.predict(inputs=inputs)

    def plot_tree(
            self,
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None,
            **kwargs
    ) -> TreeGraph:
        return TreeGraph(
            tree=self.tree_,
            feature_names=feature_names,
            class_names=class_names,
            **kwargs
        )

    def plot_complexity_parameter_search(self):
        plt.figure(figsize=(8, 6))
        plt.errorbar(
            self.complexity_param_grid_,
            self.complexity_param_cv_scores_.mean(axis=1),
            1.96 * self.complexity_param_cv_scores_.std(axis=1) / np.sqrt(self.n_folds),
            color="dodgerblue",
            marker="o",
            alpha=0.5
        )
        plt.axvline(
            self.complexity_param_,
            color="coral",
            alpha=0.5, linewidth=2,
            linestyle="--",
            label=f"Selected complexity param: {'{0:.1e}'.format(self.complexity_param_)}"
        )
        plt.xscale("log")
        plt.legend(fontsize=12)
        plt.xlabel("Complexity parameter (log scale)", fontsize=14)
        plt.ylabel(f"Validation score ({self.validation_criterion})", fontsize=14)



