import logging
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np

from algorithms.greedy_search import GreedySearch
from metrics.classification_metrics import gini
from metrics.regression_metrics import mean_squared_error
from core.tree import Tree

logger = logging.getLogger(__name__)


class LocalSearch:

    def __init__(
            self,
            criterion: Callable,
            is_classifier: bool,
            min_leaf_size: int = 1,
            max_depth: int = 10,
            search_tol: float = 1e-5,
            split_tol: Optional[float] = None,
            max_candidates_per_split: int = 100,
            max_iterations: int = 10,
            random_state: int = 0,
            verbose: bool = False,
            tree: Optional[Tree] = None,
    ):
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

        # Get all possible candidate split thresholds based on data
        feature_values = inputs[subtree.data.key, split_feature]
        feature_values = np.unique(feature_values[self.min_leaf_size:-self.min_leaf_size])
        candidate_split_thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        # If more candidates than the max amount
        n_candidates = candidate_split_thresholds.shape[0]
        if n_candidates > self.max_candidates_per_split:
            candidate_indices = np.round(
                np.arange(self.max_candidates_per_split) * n_candidates / self.max_candidates_per_split
            ).astype(int)
            candidate_split_thresholds = candidate_split_thresholds[candidate_indices]

        # If a split search termination toleration is specified, randomly select whether the search is performed from
        # left to right or right to left
        if (np.random.rand() > 0.5) and self.split_tol:
            candidate_split_thresholds = candidate_split_thresholds[::-1]

        return candidate_split_thresholds

    def _optimize_subtree_root_split(
            self,
            subtree: Tree,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> float:

        # Initialize objective value
        max_score = self.criterion(
            predicted=subtree.predict(inputs=inputs[subtree.data.key]),
            targets=targets[subtree.data.key],
            weight=sample_weights[subtree.data.key]
        )

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
            for split_threshold in self._get_candidate_split_thresholds(
                    subtree=subtree,
                    inputs=inputs,
                    split_feature=split_feature
            ):
                # Update root split
                subtree.update_splits(
                    inputs=inputs,
                    targets=targets,
                    sample_weights=sample_weights,
                    split_feature=split_feature,
                    split_threshold=split_threshold,
                    update_leaf_values_only=True
                )

                # Check if split is feasible
                if subtree.get_min_leaf_size() >= self.min_leaf_size:

                    # Check if split improves objective value and update best split if so
                    score = self.criterion(
                        predicted=subtree.predict(inputs=inputs[subtree.data.key]),
                        targets=targets[subtree.data.key],
                        sample_weights=sample_weights[subtree.data.key]
                    )
                    if score > max_score:
                        max_score, best_split_feature, best_split_threshold = score, split_feature, split_threshold

                    # Check if split has deteriorated sufficiently to end threshold search for feature
                    if score > max_score_for_feature:
                        max_score_for_feature = score
                    elif self.split_tol:
                        if max_score_for_feature - score > self.split_tol:
                            break

        # Remove children if best root is leaf
        if best_split_feature is None:
            subtree.left, subtree.right = None, None

        # Reset root split to best split and best objective value
        subtree.update_splits(
            inputs=inputs,
            targets=targets,
            sample_weights=sample_weights,
            split_feature=best_split_feature,
            split_threshold=best_split_threshold
        )
        subtree.update_depth()
        return max_score

    def _optimize_subtree_root_node(
            self,
            subtree: Tree,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> float:

        # Copy children
        left, right = map(deepcopy, subtree.get_children())

        # Optimize root split if split does not exceed max depth and calculate best objective value
        if subtree.root_depth < self.max_depth and not subtree.is_pure():
            max_score = self._optimize_subtree_root_split(
                subtree=subtree,
                inputs=inputs,
                targets=targets,
                sample_weights=sample_weights
            )
        else:
            max_score = self.criterion(
                predicted=subtree.predict(),
                targets=targets[subtree.data.key],
                sample_weights=sample_weights[subtree.data.key]
            )

        # Check if root deletion improves objective value and replace with child if so
        for child in (left, right):
            if child:
                child.data = subtree.data
                child.update_splits(
                    inputs=inputs,
                    targets=targets,
                    sample_weights=sample_weights
                )
                if child.get_min_leaf_size() >= self.min_leaf_size:
                    score = self.criterion(
                        predicted=child.predict(inputs=inputs[child.data.key]),
                        targets=targets[child.data.key],
                        sample_weights=sample_weights[child.data.key]
                    )
                    if score > max_score:
                        max_score = score
                        subtree.left, subtree.right = child.left, child.right
                        subtree.update_splits(
                            inputs=inputs,
                            targets=targets,
                            split_feature=child.split_feature,
                            split_threshold=child.split_threshold
                        )
        subtree.update_depth()
        return max_score

    def _initialize_tree(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> None:

        # If no tree provided, initialize a feasible solution one using a greedy splitting heuristic
        if self.tree is None:
            greedy_start = GreedySearch(
                criterion=(gini if self.is_classifier else mean_squared_error),
                is_classifier=self.is_classifier,
                min_parent_size=self.min_leaf_size,
                max_depth=self.max_depth,
                max_features=1 / np.sqrt(inputs.shape[1]),
                random_state=self.random_state
            )
            self.tree = greedy_start.fit(
                inputs=inputs,
                targets=targets,
                sample_weights=sample_weights
            )

        #  Else check provided tree is feasible.
        else:
            assert (
                (self.tree.get_max_depth() <= self.max_depth) and (self.tree.get_min_leaf_size() >= self.min_leaf_size)
            ), "Error - initial tree is infeasible."

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def optimize_tree(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[Tree, float]:

        # Set seed
        np.random.seed(self.random_state)

        # Initialize tree and objective value
        self._initialize_tree(
            inputs=inputs,
            targets=targets,
            weights=sample_weights
        )
        prev_score, score = -np.inf, self.criterion(
            predicted=self.tree.predict(inputs=inputs),
            targets=targets,
            weights=sample_weights
        )
        self._log(f"Iteration: {0} \t Objective value: {'{0:.3f}'.format(score)}")

        # Iterate until improvement is less than cut-off tolerance or max iterations reached
        for i in range(self.max_iterations):

            # Iterate through subtrees is random order and optimize each subtree root node
            subtrees = self.tree.get_subtrees()
            np.random.shuffle(subtrees)
            for subtree in subtrees:
                self._optimize_subtree_root_node(
                    subtree=subtree,
                    inputs=inputs,
                    targets=targets,
                    sample_weights=sample_weights
                )

            # Recalculate objective value
            prev_score, score = score, self.criterion(
                predicted=self.tree.predict(inputs=inputs),
                targets=targets,
                weights=sample_weights
            )

            self._log(f"Iteration: {i + 1} \t Objective value: {'{0:.3f}'.format(score)}")

            if np.abs(score - prev_score) < self.search_tol:
                self._log(f"No improvement found - terminating search")
                break

        # Return locally optimal tree and corresponding score
        return self.tree, score
