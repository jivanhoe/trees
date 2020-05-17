from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from models.tree import Tree


class GreedyStart:

    def __init__(
            self,
            criterion: Callable,
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
        :param min_leaf_size: an integer hyperparameter specifying the minimum number of training examples in any leaf
        :param max_depth:  an integer hyperparameter specifying the maximum depth of any leaf
        :param max_features: a float hyperparameter in range (0, 1] the specifies the proportion of features considered
        for each greedy split
        :param max_samples: a float hyperparameter in range (0, 1] that specifies the proportion of samples from the
        training data used to inform the greedy splits when growing the tree
        :param random_state: an integer used to set the numpy seed to control the random behavior of the algorithm
        """
        self.criterion = criterion
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.tree = Tree()

    def _make_greedy_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Blah
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: None
        """

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
                    loss = -self.criterion(tree=subtree, targets=targets)
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

        # Reset root split to best split
        subtree.update_splits(inputs=inputs, split_feature=best_split_feature, split_threshold=best_split_threshold)

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
            for child in subtree.get_children():
                if (child.data.sum() > self.min_leaf_size) and (child.root_depth < self.max_depth):
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
        self.tree.data = np.ones(inputs.shape[0], dtype=bool)

        # Grow tree and return
        self._grow(self.tree, inputs=inputs, targets=targets)
        return self.tree
