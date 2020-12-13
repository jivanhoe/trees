from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from core.tree import Tree
from core.soft_tree import SoftTree
from core.node_data import NodeData
from core.soft_node_data import SoftNodeData
from algorithms.greedy_splitter import GreedySplitter
from algorithms.soft_splitter import SoftSplitter


class GreedySearch:

    def __init__(
            self,
            is_classifier: bool,
            is_soft: bool,
            criterion: Callable,
            min_parent_size: int = 1,
            max_depth: int = 10,
            max_features: Optional[float] = None,
            max_samples: Optional[float] = None,
            gating_param: float = 1.0,
            cutoff_weight: float = 0,
            solver: str = "lbfgs",
            max_iter: int = 10,
            tol: float = 1e-3,
            eps: float = 1e-3,
            random_state: int = 0,
            verbose: bool = False
    ):

        self.is_classifier = is_classifier
        self.is_soft = is_soft
        self.criterion = criterion
        self.min_parent_size = min_parent_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.gating_param = gating_param
        self.cutoff_weight = cutoff_weight
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose
        self.splitter_ = None
        self.tree_ = None

    def _initialize_splitter(self) -> None:
        if self.is_soft:
            self.splitter_ = SoftSplitter(
                is_classifier=self.is_classifier,
                criterion=self.criterion,
                gating_param=self.gating_param,
                max_iter=self.max_iter,
                solver=self.solver,
                tol=self.tol,
                eps=self.eps,
                cutoff_weight=self.cutoff_weight,
                verbose=self.verbose,
            )
        else:
            self.splitter_ = GreedySplitter(
                is_classifer=self.is_classifier,
                criterion=self.criterion,
                max_features=self.max_features
            )

    def _initialize_tree(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        n_samples = inputs.shape[0]
        root_value = np.unique(targets, return_counts=True)[1] / n_samples if self.is_classifier else targets.mean()
        if self.is_soft:
            self.tree_ = SoftTree(
                data=SoftNodeData(
                    is_classifier=self.is_classifier,
                    key=np.zeros(n_samples, dtype=float),
                    value=root_value,
                ),
                gating_param=self.gating_param
            )
        else:
            self.tree_ = Tree(
                data=NodeData(
                    is_classifier=self.is_classifier,
                    key=np.ones(n_samples, dtype=bool),
                    value=root_value,
                )
            )

    def _grow(
            self,
            subtree: Tree,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray,
    ) -> None:

        # Expand the leaf with the best greedy split
        split_feature, split_threshold = self.splitter_.optimize_split(
            inputs=inputs if self.is_soft else inputs[subtree.data.key],
            targets=targets if self.is_soft else targets[subtree.data.key],
            sample_weights=sample_weights * np.exp(subtree.data.key) if self.is_soft else
            sample_weights[subtree.data.key]
        )

        #
        if split_feature and split_threshold:
            subtree.make_children(
                split_feature=split_feature,
                split_threshold=split_threshold,
                inputs=inputs,
                targets=targets,
                sample_weights=sample_weights
            )

            # Repeat recursively if further splits are feasible
            if subtree.root_depth < self.max_depth - 1:
                for child in subtree.get_children():
                    if (child.data.n_train_samples >= self.min_parent_size) and not child.is_pure():
                        self._grow(
                            subtree=child,
                            inputs=inputs,
                            targets=targets,
                            sample_weights=sample_weights
                    )

    def fit(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> Tree:

        # Set seed
        np.random.seed(self.random_state)

        # If max examples is specified, select a stratified sample of training data
        if self.max_samples:
            inputs, _, targets, _ = train_test_split(inputs, targets, shuffle=True, stratify=targets,
                                                     test_size=1-self.max_samples)

        # Assign default sample weights if none provided
        if sample_weights is None:
            sample_weights = np.ones(inputs.shape[0])

        # Initialize splitter
        self._initialize_splitter()

        # Initialize tree data
        self._initialize_tree(inputs=inputs, targets=targets)

        # Grow tree and return
        self._grow(self.tree_, inputs=inputs, targets=targets, sample_weights=sample_weights)
        return self.tree_
