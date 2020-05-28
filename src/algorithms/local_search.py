import logging
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np

from algorithms.greedy_start import GreedyStart
from metrics.classification_metrics import weighted_gini_purity
from metrics.regression_metrics import mean_squared_error
from models.tree import Tree

logger = logging.getLogger(__name__)


class LocalSearch:

    def __init__(
            self,
            criterion: Callable,
            is_classifier: bool,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            split_tol: float = np.inf,
            search_tol: float = 1e-5,
            max_candidates_per_split: int = 100,
            max_iterations: int = 10,
            random_state: int = 0,
            verbose: bool = False,
            tree: Optional[Tree] = None,
    ):
        """
        Initialize a local search object used to a improve upon a feasible tree.
        :param criterion: a callable scoring criterion accepts predictions and targets as numpy arrays and returns a
        float representing the score the tree on the data
        :param is_classifier: a boolean that specifies if the tree is being used for a classification  problem, else it
        is assumed to be for a regression problem
        :param min_leaf_size: an integer hyperparameter that specifies the minimum number of training examples in any
        leaf (default 1)
        :param max_depth: an integer hyperparameter the specifies the maximum depth of any leaf (default 10)
        :param split_tol: a float that specifies the maximum deterioration in the objective before a split threshold
        search is terminated (default inf)
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
        self.max_candidates_per_split = max_candidates_per_split
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.tree = tree

    def _get_candidate_split_thresholds(self, subtree: Tree, inputs: np.ndarray, split_feature: int) -> np.ndarray:
        """
        Get candidate split thresholds for feature.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param split_feature: an integer that specifies the column of the feature being considered for the split
        :return: a numpy array of candidate split thresholds
        """
        # Get all possible candidate split thresholds based on data
        feature_values = inputs[subtree.data.key, split_feature].flatten()
        feature_values = np.unique(feature_values[self.min_leaf_size:-self.min_leaf_size])
        candidate_split_thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        # If more candidates than the max amount
        n_candidates = candidate_split_thresholds.shape[0]
        if n_candidates > self.max_candidates_per_split:
            candidate_split_thresholds = candidate_split_thresholds[np.round(
                np.arange(self.max_candidates_per_split) * n_candidates / self.max_candidates_per_split).astype(
                int)]

        # If a split search termination toleration is specified, randomly select whether the search is performed from
        # left to right or right to left
        if (np.random.rand() > 0.5) and (self.split_tol < np.inf):
            candidate_split_thresholds = candidate_split_thresholds[::-1]

        return candidate_split_thresholds

    def _optimize_subtree_root_split(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize the split of the root node of a subtree using a brute-force approach.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the objective value (i.e. score) of the subtree with the optimal root split
        """

        # Initialize objective value
        max_score = self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])

        # Add children if subtree is leaf and initialize best split parameters
        if subtree.is_leaf():
            subtree.left, subtree.right = Tree(data=deepcopy(subtree.data)), Tree(data=deepcopy(subtree.data))
            best_split_feature, best_split_threshold = None, None
        else:
            best_split_feature, best_split_threshold = subtree.split_feature, subtree.split_threshold

        # Iterate over all features
        for split_feature in range(inputs.shape[1]):

            # Initialize objective value for feature
            max_score_for_feature = -np.inf

            # Iterate over all candidate splits
            for split_threshold in self._get_candidate_split_thresholds(subtree=subtree, inputs=inputs,
                                                                        split_feature=split_feature):
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

                    # Check if split improves objective value and update best split if so
                    score = self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])
                    if score > max_score:
                        max_score, best_split_feature, best_split_threshold = score, split_feature, split_threshold

                    # Check if split has deteriorated sufficiently to end threshold search for feature
                    if score > max_score_for_feature:
                        max_score_for_feature = score
                    elif max_score_for_feature - score > self.split_tol:
                        break

        # Remove children if best root is leaf
        if best_split_feature is None:
            subtree.left, subtree.right = None, None

        # Reset root split to best split and best objective value
        subtree.update_splits(
            inputs=inputs,
            targets=targets,
            split_feature=best_split_feature,
            split_threshold=best_split_threshold,
            update_leaf_values_only=True
        )
        subtree.update_depth()
        return max_score

    def _optimize_subtree_root_node(self, subtree: Tree, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Optimize a subtree by modifying or deleting the root node split.
        :param subtree: a Tree object
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a float representing the objective value (i.e. score) of the subtree with the optimized root node
        """

        # Copy children
        left, right = map(deepcopy, subtree.get_children())

        # Optimize root split if split does not exceed max depth and calculate best objective value
        if subtree.root_depth < self.max_depth and not subtree.is_pure():
            max_score = self._optimize_subtree_root_split(subtree=subtree, inputs=inputs, targets=targets)
        else:
            max_score = self.criterion(predicted=subtree.predict(), targets=targets[subtree.data.key])

        # Check if root deletion improves objective value and replace with child if so
        for child in (left, right):
            if child:
                child.data = subtree.data
                child.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=True)
                if child.get_min_leaf_size() >= self.min_leaf_size:
                    score = self.criterion(predicted=child.predict(), targets=targets[child.data.key])
                    if score > max_score:
                        max_score = score
                        subtree.left, subtree.right = child.left, child.right
                        subtree.update_splits(
                            inputs=inputs,
                            targets=targets,
                            split_feature=child.split_feature,
                            split_threshold=child.split_threshold,
                            update_leaf_values_only=True
                        )
        subtree.update_depth()
        return max_score

    def _initialize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Initialize a feasible tree before beginning local search.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: None
        """

        # If no tree provided, initialize a feasible solution one using a greedy splitting heuristic
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

        #  Else check provided tree is feasible.
        else:
            assert (self.tree.get_max_depth() <= self.max_depth) and (
                    self.tree.get_min_leaf_size() >= self.min_leaf_size), "Error - initial tree is infeasible."

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def optimize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[Tree, float]:
        """
        Use a local search heuristic to iteratively improve a feasible tree until a local optimum is reached.
        :param inputs: a numpy array of shape (n_samples, n_features) specifying the input values of the data
        :param targets: a numpy array of shape (n_samples,) specifying the target values of the data
        :return: a tuple consisting of a Tree object that is locally optimal for the given data, and a float that
        represents the corresponding score
        """

        # Set seed
        np.random.seed(self.random_state)

        # Initialize tree and objective value
        self._initialize_tree(inputs=inputs, targets=targets)
        prev_score, score = -np.inf, self.criterion(predicted=self.tree.predict(inputs=inputs), targets=targets)
        self._log(f"Iteration: {0} \t Objective value: {'{0:.3f}'.format(score)}")

        # Iterate until improvement is less than cut-off tolerance or max iterations reached
        for i in range(self.max_iterations):

            # Iterate through subtrees is random order and optimize each subtree root node
            subtrees = self.tree.get_subtrees()
            np.random.shuffle(subtrees)
            for subtree in subtrees:
                self._optimize_subtree_root_node(subtree=subtree, inputs=inputs, targets=targets)

            # Recalculate objective value
            self.tree.update_splits(inputs=inputs)
            prev_score, score = score, self.criterion(predicted=self.tree.predict(inputs=inputs), targets=targets)
            self._log(f"Iteration: {i + 1} \t Objective value: {'{0:.3f}'.format(score)}")

            if np.abs(score - prev_score) < self.search_tol:
                self._log(f"No improvement found - terminating search")
                break

        # Return locally optimal tree
        self.tree.update_splits(inputs=inputs, targets=targets)
        return self.tree, score
