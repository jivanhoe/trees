
import logging
from typing import Dict, Optional, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algorithms.local_search import LocalSearch
from algorithms.pruning import Pruner
from metrics.classification_metrics import accuracy, roc_auc, average_precision
from metrics.regression_metrics import mean_squared_error, r_squared
from visualization.tree_graph import TreeGraph

logger = logging.getLogger(__name__)


class OptimizedTree:

    def __init__(
            self,
            is_classifier: bool,
            criterion: Optional[str] = None,
            max_depth: int = 10,
            min_leaf_size: int = 1,
            complexity_param: float = 1e-3,
            n_restarts: int = 100,
            validation_size: float = 0.2,
            tuning_batch_size: float = 0.2,
            tune_complexity_param: bool = True,
            retrain_after_tuning: bool = True,
            verbose: bool = False,
            local_search_params: Optional[Dict[str, any]] = None
    ):
        self.criterion = criterion
        self.criterion_callback = None
        self.is_classifier = is_classifier
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.complexity_param = complexity_param
        self.n_restarts = n_restarts
        self.validation_size = validation_size
        self.tuning_batch_size = tuning_batch_size
        self.tune_complexity_param = tune_complexity_param
        self.retrain_after_tuning = retrain_after_tuning
        self.verbose = verbose
        self.local_search_params = local_search_params
        self.trees = []
        self.scores = []
        self.best_tree = None

    def _set_criterion_callback(self):
        if self.is_classifier:
            if self.criterion is None or self.criterion == "accuracy":
                self.criterion = "accuracy"
                self.criterion_callback = accuracy
            elif self.criterion == "auc":
                self.criterion_callback = roc_auc
            elif self.criterion == "precision":
                self.criterion_callback = average_precision
            else:
                raise NotImplementedError("Invalid scoring criterion - use 'accuracy', 'auc' or 'precision'")
        else:
            if self.criterion is None or self.criterion == "mse":
                self.criterion = "mse"
                self.criterion_callback = mean_squared_error
            elif self.criterion == "r2":
                self.criterion_callback = r_squared
            else:
                raise NotImplementedError("Invalid scoring criterion - use 'mse' or 'r2'")

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def _train_batch(self, inputs: np.ndarray, targets: np.ndarray) -> None:

        # Repeat for number of specified random restarts
        for i in range(self.n_restarts):

            self._log(f"Training trees - restart {i + 1}/{self.n_restarts}")

            # Run local search
            local_search = LocalSearch(
                criterion=self.criterion_callback,
                is_classifier=self.is_classifier,
                min_leaf_size=self.min_leaf_size,
                max_depth=self.max_depth,
                random_state=i,
                *(self.local_search_params if self.local_search_params else {})
            )
            tree, score = local_search.optimize_tree(inputs=inputs, targets=targets)

            # Store tree and corresponding objective value
            self.trees.append(tree)
            self.scores.append(score)

    def _tune_complexity_parameter(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        validation_inputs: np.ndarray,
        validation_targets: np.ndarray,
    ) -> None:

        # Initialize grids
        grid_size = 1000
        batch_size = int(np.round(len(self.trees) * self.tuning_batch_size))
        complexity_param_grid = np.logspace(-5, 0, grid_size)
        scoring_grid = np.zeros((batch_size, grid_size))
        sorted_restarts = np.argsort(-np.array(self.scores))

        # Iterate over each tree in the tuning batch
        for i in range(batch_size):

            # Performing pruning procedure on tree
            pruner = Pruner(base_tree=self.trees[sorted_restarts[i]], criterion=self.criterion_callback)
            pruner.prune_tree(
                train_inputs=train_inputs,
                train_targets=train_targets,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets
            )

            # Calculate validation objective value as a function of complexity parameter for tree
            validation_scores, critical_values = np.array(pruner.validation_scores), np.array(pruner.critical_values)
            validation_score_diffs = np.diff(validation_scores, prepend=0)
            for j in range(grid_size):
                scoring_grid[i, j] = (validation_score_diffs * (complexity_param_grid[j] >= critical_values)).sum()

        # Select the complexity parameter with the best mean validation objective value
        mean_scores = scoring_grid.mean(axis=0)
        argmax_set = np.argwhere(mean_scores == mean_scores.max())
        self.complexity_param = (complexity_param_grid[argmax_set.min()] + complexity_param_grid[argmax_set.max()]) / 2

    def _set_best_tree(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        pruner = Pruner(
            base_tree=self.trees[np.argmax(self.scores)],
            criterion=self.criterion_callback,
            complexity_param=self.complexity_param
        )
        self.best_tree = pruner.prune_tree(train_inputs=inputs, train_targets=targets)

    def _fit_without_tuning(self, inputs: np.ndarray, targets: np.ndarray) -> None:

        # Get batch of trees using local search
        self._train_batch(inputs=inputs, targets=targets)

        # Prune best tree in batch and set it as best tree
        self._log(f"Using pre-specified complexity parameter - pruning best tree")
        self._set_best_tree(inputs=inputs, targets=targets)

    def _fit_with_tuning(self, inputs: np.ndarray, targets: np.ndarray) -> None:

        # Split data in training and validation sets
        train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(
            inputs,
            targets,
            test_size=self.validation_size,
            stratify=targets
        )

        # Get batch of trees using local search
        self._train_batch(inputs=train_inputs, targets=train_targets)

        # Tune complexity parameter
        self._log(f"Tuning complexity parameter")
        self._tune_complexity_parameter(
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets
        )

        # Retrain with all data if desired, else prune best existing tree
        if self.retrain_after_tuning:
            self._log(f"Complexity parameter selected - retraining trees")
            self._fit_without_tuning(inputs=inputs, targets=targets)
        else:
            self._log(f"Complexity parameter selected - pruning best existing tree")
            self._set_best_tree(inputs=train_targets, targets=train_targets)

    def fit(self, inputs: np.ndarray, targets: np.ndarray):
        targets = LabelEncoder().fit_transform(targets)
        self._set_criterion_callback()
        if self.tune_complexity_param:
            self._fit_with_tuning(inputs=inputs, targets=targets)
        else:
            self._fit_without_tuning(inputs=inputs, targets=targets)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        assert self.best_tree is not None, "Cannot make predictions for unfitted model"
        predictions = self.best_tree.predict(inputs=inputs)
        return predictions.argmax(1) if self.is_classifier else predictions

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        assert self.best_tree is not None, "Cannot make predictions for unfitted model"
        assert self.is_classifier, "Cannot predict probabilities for regression model"
        return self.best_tree.predict(inputs=inputs)

    def plot(
            self,
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None,
            colors: Optional[List[str]] = None,
            font: str = "verdana",
            fontsize: float = 10.0,
            max_depth_to_plot: int = 5
    ) -> TreeGraph:
        assert self.best_tree is not None, "Cannot make plot tree for unfitted model"
        return TreeGraph(
            tree=self.best_tree,
            feature_names=feature_names,
            class_names=class_names,
            colors=colors,
            font=font,
            fontsize=fontsize,
            max_depth_to_plot=max_depth_to_plot
        )

