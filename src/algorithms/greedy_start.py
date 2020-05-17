from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from models.tree import Tree


class GreedyStart:

    def __init__(
            self,
            loss_criterion: Callable,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            max_features: Optional[float] = None,
            max_examples: Optional[float] = None,
            random_state: int = 0
    ):
        self.loss_criterion = loss_criterion
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_examples = max_examples
        self.random_state = random_state
        self.tree = Tree()

    def _make_greedy_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarry) -> None:

        # Initialize loss and best split parameters
        min_loss, best_split_feature, best_split_threshold = np.inf, None, None

        # Initialize children
        subtree.left, subtree.right = Tree(), Tree()

        # Select features to consider for split
        features = np.arange(inputs.shape[1])
        np.random.shuffle(features)
        features = features[:int(np.ceil(self.max_features * features.shape[0]))]

        for split_feature in features:

            # Get reference values for candidate split thresholds for feature
            values = inputs[subtree.data, split_feature].flatten()
            values.sort()

            for i in range(values.shape[0] - 1):

                # Update root split
                split_threshold = (values[i] + values[i + 1]) / 2
                subtree.update_splits(inputs=inputs, split_feature=split_feature, split_threshold=split_threshold)

                # Check if split is feasible
                if self.min_leaf_size <= min([leaf_data.sum() for leaf_data in subtree.get_leaf_data()]):

                    # Check if split improves objective and update best split if so
                    loss = self.loss_criterion(subtree=subtree, targets=targets)
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

        # Reset root split to best split
        subtree.update_splits(inputs=inputs, split_feature=best_split_feature, split_threshold=best_split_threshold)

    def _grow(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarry) -> None:

        # Check if subtree is leaf
        if subtree.is_leaf():

            # Expand the leaf with the best greedy split
            self._make_greedy_split(subtree=subtree, inputs=inputs, targets=targets)
            self.tree.update_depth()

            # Repeat recursively if further splits are feasible
            for child in subtree.get_children():
                if (child.data.sum() > self.min_leaf_size) and (child.root_depth < self.max_depth):
                    self._grow(subtree=child, inputs=inputs, targets=targets)

    def build_tree(self, inputs: np.ndarray, targets: np.ndarry) -> Tree:

        # Set seed
        np.random.seed(self.random_state)

        # If max features
        if self.max_features is None:
            self.max_features = 1 / np.sqrt(inputs.shape[1])

        # If max examples is specified, select a stratified sample of training data
        if self.max_examples:
            inputs, targets, _, _ = train_test_split(inputs, targets, shuffle=True, stratify=targets,
                                                     test_size=1-self.max_examples)

        # Initialize tree data
        self.tree.data = np.ones(inputs.shape[0], dtype=bool)

        # Grow tree and return
        self._grow(self.tree, inputs=inputs, targets=targets)
        return self.tree
