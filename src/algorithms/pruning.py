import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from models.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pruner:

    def __init__(
            self,
            base_tree: Tree,
            criterion: Callable,
            complexity_param: float = 1e-2,
    ):
        self.base_tree = base_tree
        self.criterion = criterion
        self.complexity_param = complexity_param
        self.pruned_trees = []
        self.validation_scores = []
        self.critical_values = []
        self.validation_scores = None

    @staticmethod
    def _prune_parent(tree: Tree, parent: Tree) -> Optional[Tree]:
        if not tree.is_leaf():
            if np.all(tree.data.key == parent.data.key):
                tree.delete_children()
            else:
                for child in tree.get_children():
                    Pruner._prune_parent(tree=child, parent=parent)
        return tree

    def _calculate_critical_value(self, tree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        complexity_change = tree.get_node_count() - self.base_tree.get_node_count()
        score_change = self.criterion(predicted=tree.predict(inputs), targets=targets) - self.criterion(
            predicted=self.base_tree.predict(inputs), targets=targets)
        return score_change / complexity_change

    def _get_pruned_trees(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        tree = deepcopy(self.base_tree)
        self.pruned_trees.append(tree)
        self.critical_values.append(0)
        while tree.get_max_depth() > 0:
            min_critical_value = np.inf
            for pruned_tree in [Pruner._prune_parent(tree=deepcopy(tree), parent=leaf_parent)
                                for leaf_parent in tree.get_leaf_parents()]:
                critical_value = self._calculate_critical_value(tree=pruned_tree, inputs=inputs, targets=targets)
                if critical_value < min_critical_value:
                    min_critical_value = critical_value
                    tree = deepcopy(pruned_tree)

            self.pruned_trees.append(tree)
            self.critical_values.append(min_critical_value)

    def _get_pruning_data(
            self,
            train_inputs: np.ndarray,
            train_targets: np.ndarray,
            validation_inputs: Optional[np.ndarray],
            validation_targets: Optional[np.ndarray]
    ) -> None:

        # Get pruned trees and corresponding critical values
        self._get_pruned_trees(inputs=train_inputs, targets=train_targets)

        # Get validation errors
        if (validation_inputs is not None) and (validation_targets is not None):
            self.validation_scores = np.array([
                self.criterion(predicted=tree.predict(inputs=validation_inputs), targets=validation_targets)
                for tree in self.pruned_trees
            ])

    def prune_tree(
            self,
            train_inputs: np.ndarray,
            train_targets: np.ndarray,
            validation_inputs: Optional[np.ndarray] = None,
            validation_targets: Optional[np.ndarray] = None,
    ) -> Tree:
        self._get_pruning_data(
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets
        )
        if self.validation_scores is not None:
            selected_index = np.argwhere(self.validation_scores == np.max(self.validation_scores)).max()
            self.complexity_param = (self.critical_values[selected_index] + self.critical_values[selected_index + 1])/2
        else:
            selected_index = np.argmax(np.array(self.critical_values) > self.complexity_param)
        return self.pruned_trees[selected_index]
