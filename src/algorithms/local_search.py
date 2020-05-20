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
            min_leaf_size: int = 1,
            max_depth: int = 10,
            split_tol: float = np.inf,
            search_tol: float = 1e-3,
            max_iterations: int = 10,
            random_state: int = 0,
            tree: Optional[Tree] = None,
    ):
        """
        Initialize a local search object used to a improve upon a feasible tree.
        :param criterion: a callable object that accepts the arguments 'tree', a Tree object, and 'targets', a
        numpy array of shape (n_samples,), and returns a float representing the loss the tree on the data
        :param is_classifier: a boolean that specifies if the tree is being used for a classification  problem, else it
        is assumed to be for a regression problem
        :param min_leaf_size: an integer hyperparameter that specifies the minimum number of training examples in any
        leaf (default 1)
        :param max_depth: an integer hyperparameter the specifies the maximum depth of any leaf (default 10)
        :param split_tol: a float that specifies the maximum deterioration in the loss before a split threshold search
        is terminated (default inf)
        :param search_tol: a float that specifies the cut-off tolerance for terminating the local search
        :param max_iterations: an integer that specifies the maximum number of iterations of local search performed
        :param random_state: an integer used to set the numpy seed to control the random behavior of the algorithm
        (default 0)
        :param tree: an optional Tree object, which if provided is used as the feasible incumbent solution upon which
        the local search begins improving (default None)
        """
        self.criterion = criterion
        self.is_classifier = is_classifier
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.split_tol = split_tol
        self.search_tol = search_tol
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.tree = tree

    def _optimize_subtree_root_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize the split of the root node of a subtree using a brute-force approach.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the regularized loss of the subtree with the optimal root split
        """

        # Initialize min loss
        min_loss = -self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])

        # Add children if subtree is leaf and initialize best split parameters
        if subtree.is_leaf():
            subtree.left, subtree.right = Tree(data=deepcopy(subtree.data)), Tree(data=deepcopy(subtree.data))
            best_split_feature, best_split_threshold = None, None
        else:
            best_split_feature, best_split_threshold = subtree.split_feature, subtree.split_threshold

        # Iterate over all features
        for split_feature in range(inputs.shape[1]):

            # Initialize min loss for feature
            min_loss_for_feature = np.inf

            # Get reference values for candidate split thresholds for feature
            feature_values = inputs[subtree.data.key, split_feature].flatten()
            feature_values = np.unique(feature_values[self.min_leaf_size:-self.min_leaf_size])
            candidate_split_thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            # Iterate over all candidate splits
            for split_threshold in candidate_split_thresholds:

                # Update root split
                subtree.update_splits(
                    inputs=inputs,
                    targets=targets,
                    split_feature=split_feature,
                    split_threshold=split_threshold,
                    update_leaf_values_only=True
                )

                # Check if split is feasible
                if subtree.get_min_leaf_size() >= self.min_leaf_size:

                    # Check if split improves objective and update best split if so
                    loss = -self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])
                    if loss < min_loss:
                        min_loss, best_split_feature, best_split_threshold = loss, split_feature, split_threshold

                    # Check if split has deteriorated sufficiently to end threshold search for feature
                    if loss < min_loss_for_feature:
                        min_loss_for_feature = loss
                    elif loss - min_loss_for_feature > self.split_tol:
                        break

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
            min_loss = -self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])

        # Check if root deletion improves objective and replace with child if so
        for child in (left, right):
            if child:
                child.data = subtree.data
                child.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=True)
                if child.get_min_leaf_size() >= self.min_leaf_size:
                    loss = -self.criterion(predicted=child.predict(), targets=targets[child.data.key])
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
        If no tree provided, initialize a feasible solution one using a greedy splitting heuristic. Else check provided
        tree is feasible.
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
                max_features=None,
                random_state=self.random_state
            )
            self.tree = greedy_start.build_tree(inputs=inputs, targets=targets)
        else:
            assert (self.tree.get_max_depth() <= self.max_depth) and (
                    self.tree.get_min_leaf_size() >= self.min_leaf_size), "Error - initial tree is infeasible."

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
        prev_loss, loss = np.inf, -self.criterion(predicted=self.tree.predict(inputs=inputs), targets=targets)
        logger.info(f"Iteration: {0} \t Objective value: {'{0:.3f}'.format(-loss)}")

        # Iterate until improvement is less than cut-off tolerance or max iterations reached
        for i in range(self.max_iterations):

            # Iterate through subtrees is random order and optimize each subtree root node
            subtrees = self.tree.get_subtrees()
            np.random.shuffle(subtrees)
            for subtree in subtrees:
                self._optimize_subtree_root_node(subtree=subtree, inputs=inputs, targets=targets)

            # Recalculate loss
            self.tree.update_splits(inputs=inputs)
            prev_loss, loss = loss, -self.criterion(predicted=self.tree.predict(inputs=inputs), targets=targets)
            logger.info(f"Iteration: {i + 1} \t Objective value: {'{0:.3f}'.format(-loss)}")

            if np.abs(loss - prev_loss) < self.search_tol:
                logger.info(f"No improvement found - terminating search.")
                break

        # Return locally optimal tree
        self.tree.update_splits(inputs=inputs, targets=targets)
        return self.tree
