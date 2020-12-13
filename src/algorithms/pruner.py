import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from core.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pruner:

    def __init__(
            self,
            base_tree: Tree,
            criterion: Callable
    ):
        self.base_tree = base_tree
        self.criterion = criterion
        self._base_score = None
        self.critical_values_ = []
        self.pruned_trees_ = []
        self.validation_scores_ = None
        self.best_complexity_param_ = None
        self.best_tree_ = None

    @staticmethod
    def _prune_parent(tree: Tree, parent: Tree) -> Optional[Tree]:
        if not tree.is_leaf():
            if np.isclose(tree.data.key, parent.data.key).all():
                tree.delete_children()
            else:
                for child in tree.get_children():
                    Pruner._prune_parent(tree=child, parent=parent)
        return tree

    def _calculate_critical_value(
            self,
            tree: Tree,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> float:
        score = self.criterion(
            predicted=tree.predict(inputs),
            targets=targets,
            sample_weights=sample_weights
        )
        return (self._base_score - score) / (self.base_tree.get_node_count() - tree.get_node_count())

    def _get_pruned_trees(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> None:

        # Initialize
        tree = deepcopy(self.base_tree)
        self.pruned_trees_.append(tree)
        self.critical_values_.append(0)
        self._base_score = self.criterion(
            predicted=self.base_tree.predict(inputs),
            targets=targets,
            sample_weights=sample_weights
        )

        while tree.get_max_depth() > 1:
            min_critical_value = np.inf
            for pruned_tree in [
                Pruner._prune_parent(tree=deepcopy(tree), parent=leaf_parent)
                for leaf_parent in tree.get_leaf_parents()
            ]:
                critical_value = self._calculate_critical_value(
                    tree=pruned_tree,
                    inputs=inputs,
                    targets=targets,
                    sample_weights=sample_weights
                )
                if critical_value < min_critical_value:
                    min_critical_value = critical_value
                    tree = deepcopy(pruned_tree)

            self.pruned_trees_.append(tree)
            self.critical_values_.append(min_critical_value)

    def _get_pruning_data(
            self,
            train_inputs: np.ndarray,
            train_targets: np.ndarray,
            train_sample_weights: np.ndarray,
            validation_inputs: Optional[np.ndarray] = None,
            validation_targets: Optional[np.ndarray] = None,
            validation_sample_weights: Optional[np.ndarray] = None
    ) -> None:

        # Get pruned trees and corresponding critical values
        self._get_pruned_trees(inputs=train_inputs, targets=train_targets, sample_weights=train_sample_weights)

        # Get validation errors
        if (validation_inputs is not None) and (validation_targets is not None):
            self.validation_scores_ = np.array([
                self.criterion(
                    predicted=tree.predict(inputs=validation_inputs),
                    targets=validation_targets,
                    sample_weights=validation_sample_weights
                ) for tree in self.pruned_trees_
            ])

    def fit(
            self,
            train_inputs: np.ndarray,
            train_targets: np.ndarray,
            train_sample_weights: np.ndarray,
            validation_inputs: Optional[np.ndarray] = None,
            validation_targets: Optional[np.ndarray] = None,
            validation_sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        self._get_pruning_data(
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_sample_weights=train_sample_weights,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            validation_sample_weights=validation_sample_weights
        )
        if self.validation_scores_ is not None:
            i = np.argwhere(self.validation_scores_ == np.max(self.validation_scores_)).max()
            self.best_complexity_param_ = (
               self.critical_values_[i] + self.critical_values_[min(i + 1, len(self.critical_values_) - 1)]
            ) / 2
            self.best_tree_ = self.pruned_trees_[i]


