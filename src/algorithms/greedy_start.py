from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

from models.tree import Tree, NodeData


class GreedyStart:

    def __init__(
            self,
            criterion: Callable,
            is_classifier: bool,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            max_features: Optional[float] = None,
            max_samples: Optional[float] = None,
            random_state: int = 0
    ):
        """
        Initialize a greedy search object used to a build feasible tree.
        :param criterion: a callable object that accepts the arguments 'tree', a Tree object, and 'targets', a
        numpy array of shape (n_samples,), and returns a float representing the objective the tree on the data
        :param is_classifier: a boolean that specifies if the tree is being used for a classification  problem, else it
        is assumed to be for a regression problem
        :param min_leaf_size: an integer hyperparameter specifying the minimum number of training examples in any leaf
        (default 1)
        :param max_depth:  an integer hyperparameter specifying the maximum depth of any leaf (default 10)
        :param max_features: a float hyperparameter in range (0, 1] the specifies the proportion of features considered
        for each greedy split (default None)
        :param max_samples: a float hyperparameter in range (0, 1] that specifies the proportion of samples from the
        training data used to inform the greedy splits when growing the tree (default None)
        :param random_state: an integer used to set the numpy seed to control the random behavior of the algorithm
        (default 0)
        """
        self.criterion = criterion
        self.is_classifier = is_classifier
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.tree = None

    def _make_greedy_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Split a leaf node to minimize the loss criterion amongst its newly created children.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: None
        """

        # Initialize loss and best split parameters
        min_loss = -self.criterion(tree=subtree, targets=targets)
        best_split_feature, best_split_threshold = None, None

        # Initialize children
        subtree.left, subtree.right = Tree(data=deepcopy(subtree.data)), Tree(data=deepcopy(subtree.data))

        # Select features to consider for split
        features = np.arange(inputs.shape[1])
        np.random.shuffle(features)
        features = features[:int(np.ceil(self.max_features * features.shape[0]))]

        for split_feature in features:

            # Get reference values for candidate split thresholds for feature
            feature_values = np.unique(inputs[subtree.data.key, split_feature].flatten())

            for split_threshold in (feature_values[:-1] + feature_values[1:]) / 2:

                # Update root split
                subtree.update_splits(
                    inputs=inputs,
                    targets=targets,
                    split_feature=split_feature,
                    split_threshold=split_threshold
                )

                # Check if split is feasible
                if min([leaf_data.key.sum() for leaf_data in subtree.get_leaf_data()]) >= self.min_leaf_size:

                    # Check if split improves objective and update best split if so
                    loss = -self.criterion(tree=subtree, targets=targets)
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

        # Reset root split to best split
        if best_split_feature is None:
            subtree.left, subtree.right = None, None
        subtree.update_splits(
            inputs=inputs,
            targets=targets,
            split_feature=best_split_feature,
            split_threshold=best_split_threshold
        )

    def _grow(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Recursively grow a tree using greedy splits until max depth or min leaf size is reached.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: None
        """

        # Check if subtree is leaf
        if subtree.is_leaf():

            # Expand the leaf with the best greedy split
            self._make_greedy_split(subtree=subtree, inputs=inputs, targets=targets)
            self.tree.update_depth()

            # Repeat recursively if further splits are feasible
            if not subtree.is_leaf():
                for child in subtree.get_children():
                    if (child.data.key.sum() > self.min_leaf_size) and (child.root_depth < self.max_depth):
                        self._grow(subtree=child, inputs=inputs, targets=targets)

    def build_tree(self, inputs: np.ndarray, targets: np.ndarray) -> Tree:
        """
        Build a feasible (and sensible) tree using a greedy splitting heuristic.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a Tree object that is feasible for the given data and hyperparameters
        """

        # Set seed
        np.random.seed(self.random_state)

        # If max features
        if self.max_features is None:
            self.max_features = 1 / np.sqrt(inputs.shape[1])

        # If max examples is specified, select a stratified sample of training data
        if self.max_samples:
            inputs, targets, _, _ = train_test_split(inputs, targets, shuffle=True, stratify=targets,
                                                     test_size=1-self.max_samples)

        # Initialize tree data
        num_samples = inputs.shape[0]
        self.tree = Tree(
            data=NodeData(
                key=np.ones(num_samples, dtype=bool),
                value=np.unique(targets, return_counts=True)[1] / num_samples if self.is_classifier else targets.mean()
            )
        )

        # Grow tree and return
        self._grow(self.tree, inputs=inputs, targets=targets)
        return self.tree
