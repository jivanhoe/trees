from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from models.tree import Tree
from algorithms.greedy_start import GreedyStart
from metrics.classification_metrics import weighted_gini_impurity
from metrics.regression_metrics import mean_squared_error


class LocalSearch:

    def __init__(
            self,
            criterion: Callable,
            is_classifier: bool,
            tree: Optional[Tree] = None,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            complexity_param: float = 0.0,
            tol: float = 1e-5,
            random_state: int = 0
    ):
        """

        :param criterion: a callable object that accepts the arguments 'tree', a Tree object, and 'targets', a
        numpy array of shape (n_samples,), and returns a float representing the loss the tree on the data
        :param is_classifier: a boolean that specifies if the tree is being used for a classification  problem, else it
        is assumed to be for a regression problem
        :param tree: an optional Tree object, which if provided is used as the feasible incumbent solution upon which
        the local search begins improving (default None)
        :param min_leaf_size: an integer hyperparameter that specifies the minimum number of training examples in any
        leaf (default 1)
        :param max_depth: an integer hyperparameter the specifies the maximum depth of any leaf (default 10)
        :param complexity_param: a float hyperparameter that controls the trade-off between the ability of the tree to
        fit the training data and its
        :param tol: a
        :param random_state: an integer used to set the numpy seed to control the random behavior of the algorithm
        """
        self.criterion = criterion
        self.is_classifier = is_classifier
        self.tree = tree
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.complexity_param = complexity_param
        self.tol = tol
        self.random_state = random_state

    def _regularized_loss(self, subtree: Tree, targets: np.ndarray) -> float:
        """
        Compute the regularized loss of a subtree based on the loss criterion and subtree complexity (i.e. node count).
        :param subtree: a Tree object
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the regularized loss of the subtree
        """
        return -self.criterion(tree=subtree, targets=targets) + self.complexity_param * subtree.get_node_count()

    def _optimize_subtree_root_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize the split of the root node of a subtree using a brute-force approach.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the regularized loss of the subtree with the optimal root split
        """

        # Initialize loss and best split parameters
        min_loss = self._regularized_loss(subtree=subtree, targets=targets)
        best_split_feature, best_split_threshold = subtree.split_feature, subtree.split_threshold

        # Add children if subtree is leaf
        if subtree.is_leaf():
            subtree.left, subtree.right = Tree(), Tree()

        for split_feature in range(inputs.shape[1]):

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
                    loss = self._regularized_loss(subtree=subtree, targets=targets)
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

        # Remove children if best root is leaf
        if best_split_feature is None:
            subtree.left, subtree.right = None, None

        # Reset root split to best split and return min loss
        subtree.update_splits(inputs=inputs, split_feature=best_split_feature, split_threshold=best_split_threshold)
        return min_loss

    def _optimize_subtree_root_node(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize a subtree by modifying or deleting the root node split.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the regularized loss of the subtree with the optimized root node
        """

        # Copy children
        left, right = map(deepcopy, subtree.get_children())

        # Optimize root split if split does not exceed max depth and calculate min loss
        if subtree.root_depth < self.max_depth:
            min_loss = self._optimize_subtree_root_split(subtree=subtree, inputs=inputs, targets=targets)
        else:
            min_loss = self._regularized_loss(subtree=subtree, targets=targets)

        # Check if root deletion improves objective and replace with child if so
        for child in (left, right):
            if child:
                child.data = subtree.data
                child.update_splits(inputs=inputs)
                loss = self._regularized_loss(subtree=child, targets=targets)
                if loss < min_loss:
                    min_loss = loss
                    subtree.left, subtree.right = child.left, child.right
                    subtree.update_splits(inputs=inputs, split_feature=child.split_feature,
                                          split_threshold=child.split_threshold)
        return min_loss

    def optimize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> Tree:
        """
        Use a local search heuristic to iteratively improve a feasible tree until a local optimum is reached.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a Tree object that is locally optimal for the given data
        """

        # Set seed
        np.random.seed(self.random_state)

        # Initialize tree
        if self.tree is None:
            greedy_start = GreedyStart(criterion=(weighted_gini_impurity if self.is_classifier else mean_squared_error),
                                       min_leaf_size=self.min_leaf_size, max_depth=self.max_depth,
                                       random_state=self.random_state)
            self.tree = greedy_start.build_tree(inputs=inputs, targets=targets)
        self.tree.data = np.ones(inputs.shape[0], dtype=bool)
        self.tree.update_splits(inputs=inputs)

        # Initialize loss
        prev_loss, loss = np.inf, self._regularized_loss(subtree=self.tree, targets=targets)

        # Iterate until improvement is less than cut-off tolerance
        while np.abs(loss - prev_loss) > self.tol:

            # Iterate through subtrees is random order and optimize each subtree root node
            subtrees = self.tree.get_subtrees()
            np.random.shuffle(subtrees)
            for subtree in subtrees:
                self._optimize_subtree_root_node(subtree=subtree, inputs=inputs, targets=targets)
            self.tree.update_depth()

            # Recalculate loss
            prev_loss, loss = loss, self._regularized_loss(subtree=self.tree, targets=targets)

        # Return locally optimal tree
        return self.tree
