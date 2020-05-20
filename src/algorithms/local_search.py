import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from algorithms.greedy_start import GreedyStart
from metrics.classification_metrics import weighted_gini_purity
from metrics.regression_metrics import mean_squared_error
from models.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalSearch:

    def __init__(
            self,
            criterion: Callable,
            is_classifier: bool,
            tree: Optional[Tree] = None,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            tol: float = 1e-5,
            max_iterations: int = 10,
            random_state: int = 0
    ):
        """
        Initialize a local search object used to a improve upon a feasible tree.
        :param criterion: a callable object that accepts the arguments 'tree', a Tree object, and 'targets', a
        numpy array of shape (n_samples,), and returns a float representing the loss the tree on the data
        :param is_classifier: a boolean that specifies if the tree is being used for a classification  problem, else it
        is assumed to be for a regression problem
        :param tree: an optional Tree object, which if provided is used as the feasible incumbent solution upon which
        the local search begins improving (default None)
        :param min_leaf_size: an integer hyperparameter that specifies the minimum number of training examples in any
        leaf (default 1)
        :param max_depth: an integer hyperparameter the specifies the maximum depth of any leaf (default 10)
        :param tol:
        :param max_iterations:
        :param random_state: an integer used to set the numpy seed to control the random behavior of the algorithm
        (default 0)
        """
        self.criterion = criterion
        self.is_classifier = is_classifier
        self.tree = tree
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.tol = tol
        self.max_iterations = max_iterations
        self.random_state = random_state

    def _optimize_subtree_root_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize the split of the root node of a subtree using a brute-force approach.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the regularized loss of the subtree with the optimal root split
        """

        # Initialize loss and best split parameters
        min_loss = -self.criterion(tree=subtree, targets=targets)
        best_split_feature, best_split_threshold = subtree.split_feature, subtree.split_threshold

        # Add children if subtree is leaf
        if subtree.is_leaf():
            subtree.left, subtree.right = Tree(data=deepcopy(subtree.data)), Tree(data=deepcopy(subtree.data))

        for split_feature in range(inputs.shape[1]):

            # Get reference values for candidate split thresholds for feature
            values = inputs[subtree.data.key, split_feature].flatten()
            values = np.unique(values[self.min_leaf_size:-self.min_leaf_size])

            for split_threshold in (values[:-1] + values[1:]) / 2:

                # Update root split
                subtree.update_splits(
                    inputs=inputs,
                    targets=targets,
                    split_feature=split_feature,
                    split_threshold=split_threshold,
                    update_leaf_values_only=True
                )

                # Check if split is feasible
                if min([leaf_data.key.sum() for leaf_data in subtree.get_leaf_data()]) >= self.min_leaf_size:

                    # Check if split improves objective and update best split if so
                    loss = -self.criterion(tree=subtree, targets=targets)
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

        # Remove children if best root is leaf
        if best_split_feature is None:
            subtree.left, subtree.right = None, None

        # Reset root split to best split and return min loss
        subtree.update_splits(
            inputs=inputs,
            targets=targets,
            split_feature=best_split_feature,
            split_threshold=best_split_threshold,
            update_leaf_values_only=True
        )
        subtree.update_depth()
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
        if subtree.root_depth < self.max_depth and not subtree.is_pure():
            min_loss = self._optimize_subtree_root_split(subtree=subtree, inputs=inputs, targets=targets)
        else:
            min_loss = -self.criterion(tree=subtree, targets=targets)

        # Check if root deletion improves objective and replace with child if so
        for child in (left, right):
            if child:
                child.data = subtree.data
                child.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=True)
                loss = -self.criterion(tree=child, targets=targets)
                if loss < min_loss:
                    min_loss = loss
                    subtree.left, subtree.right = child.left, child.right
                    subtree.update_splits(
                        inputs=inputs,
                        targets=targets,
                        split_feature=child.split_feature,
                        split_threshold=child.split_threshold,
                        update_leaf_values_only=True
                    )
        subtree.update_depth()
        return min_loss

    def _initialize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        If no tree provided, initialize a feasible solution one using a greedy splitting heuristic. Update splits for
        the full training data set.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: None
        """

        if self.tree is None:
            greedy_start = GreedyStart(
                criterion=(weighted_gini_purity if self.is_classifier else mean_squared_error),
                is_classifier=self.is_classifier,
                min_leaf_size=self.min_leaf_size,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.tree = greedy_start.build_tree(inputs=inputs, targets=targets)

        self.tree.data.keys = np.ones(inputs.shape[0], dtype=bool)
        self.tree.update_splits(inputs=inputs, targets=targets)

    def optimize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> Tree:
        """
        Use a local search heuristic to iteratively improve a feasible tree until a local optimum is reached.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a Tree object that is locally optimal for the given data
        """

        # Set seed
        np.random.seed(self.random_state)

        # Initialize tree and loss
        self._initialize_tree(inputs=inputs, targets=targets)
        prev_loss, loss = np.inf, -self.criterion(tree=self.tree, targets=targets)
        logger.info(f"Iteration: {0} \t Objective value: {'{0:.3f}'.format(-loss)}")

        # Iterate until improvement is less than cut-off tolerance or max iterations reached
        for i in range(self.max_iterations):

            # Iterate through subtrees is random order and optimize each subtree root node
            subtrees = self.tree.get_subtrees()
            np.random.shuffle(subtrees)
            for subtree in subtrees:
                self._optimize_subtree_root_node(subtree=subtree, inputs=inputs, targets=targets)

            # Recalculate loss
            prev_loss, loss = loss, -self.criterion(tree=self.tree, targets=targets)
            logger.info(f"Iteration: {i + 1} \t Objective value: {'{0:.3f}'.format(-loss)}")

            if np.abs(loss - prev_loss) < self.tol:
                logger.info(f"No improvement found - terminating search.")
                break

        # Return locally optimal tree
        self.tree.update_splits(inputs=inputs, targets=targets)
        return self.tree
