from typing import Optional, NoReturn
from copy import deepcopy

import numpy as np

from core.tree import Tree
from core.soft_node_data import SoftNodeData


def logistic(
        x: np.ndarray,
        gating_param: float = 1.0,
        threshold: float = 0.0
) -> np.ndarray:
    return 1 / (1 + np.exp(-gating_param * (x - threshold)))


class SoftTree(Tree):

    def __init__(
            self,
            data: SoftNodeData,
            split_feature: Optional[int] = None,
            split_threshold: Optional[float] = None,
            left: Optional[Tree] = None,
            right: Optional[Tree] = None,
            root_depth: Optional[int] = 0,
            gating_param: float = 1.0,
            eps: float = 1e-3
    ):
        super().__init__(
            data=data,
            split_feature=split_feature,
            split_threshold=split_threshold,
            left=left,
            right=right,
            root_depth=root_depth
        )
        self.gating_param = gating_param
        self.eps = eps

    def make_children(
            self,
            split_feature: int,
            split_threshold: float,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> None:
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left = SoftTree(data=deepcopy(self.data))
        self.right = SoftTree(data=deepcopy(self.data))
        self.update_splits(inputs=inputs, targets=targets, sample_weights=sample_weights)
        self.update_depth()

    def _apply_gate(self, inputs: np.ndarray) -> None:
        prob = logistic(
            x=inputs[:, self.split_feature],
            gating_param=self.gating_param,
            threshold=self.split_threshold
        )
        self.left.data.key = np.log(np.clip(prob, a_min=self.eps, a_max=1 - self.eps)) + self.data.key
        self.right.data.key = np.log(np.clip(1 - prob, a_min=self.eps, a_max=1 - self.eps)) + self.data.key

    def predict(self, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        n_samples = inputs.shape[0]
        if inputs is not None:
            self.data.key = np.zeros(inputs.shape[0], dtype=float)
            self.update_splits(inputs=inputs)
        predicted = np.zeros((n_samples, self.data.value.shape[0])) if self.data.is_classifier else np.zeros(n_samples)
        for leaf in self.get_leaves():
            predicted += np.exp(leaf.data.key)[:, None] * leaf.data.value
        return predicted
