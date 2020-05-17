from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np


class Tree:

    def __init__(
            self,
            split_feature: Optional[int] = None,
            split_threshold: Optional[float] = None,
            left: Optional[Tree] = None,
            right: Optional[Tree] = None,
            data: Optional[np.ndarray] = None,
            root_depth: Optional[int] = 0
    ):
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right
        self.data = data
        self.root_depth = root_depth

    def is_leaf(self):
        return (
                (self.split_feature is None)
                or (self.split_threshold is None)
                or (self.left is None)
                or (self.right is None)
        )

    def get_node_count(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + self.left.get_node_count() + self.right.get_node_count()

    def get_max_depth(self) -> int:
        if self.is_leaf():
            return 0
        return max(self.left.get_max_depth(), self.right.get_max_depth()) + 1

    def get_children(self) -> Tuple[Tree, Tree]:
        return self.left, self.right

    def get_subtrees(self, subtrees: Optional[List] = None) -> List[Tree]:
        if subtrees is None:
            subtrees = []
        subtrees.append(self)
        if not self.is_leaf():
            self.left.nodes(subtrees=subtrees)
            self.right.nodes(subtrees=subtrees)
        return subtrees

    def get_leaf_data(self, leaf_data: Optional[List] = None) -> List[np.ndarray]:
        if leaf_data is None:
            leaf_data = []
        if self.is_leaf():
            leaf_data.append(self.data)
        else:
            self.left.leaf_data(data=leaf_data)
            self.right.leaf_data(data=leaf_data)
        return leaf_data

    def update_splits(
            self,
            inputs: np.ndarray,
            split_feature: Optional[int] = None,
            split_threshold: Optional[float] = None
    ) -> None:
        if split_feature:
            self.split_feature = split_feature
            self.split_threshold = split_threshold
        if not self.is_leaf():
            mask = inputs[:, self.split_feature] < self.split_threshold
            self.left.data = mask * self.data
            self.right.data = ~mask * self.data
            self.left.update_splits(inputs=inputs)
            self.right.update_splits(inputs=inputs)

    def update_depth(self) -> None:
        for child in self.get_children():
            child.root_depth = self.root_depth + 1
            child.update_depth()




