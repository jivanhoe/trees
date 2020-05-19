from __future__ import annotations
from typing import Optional, Tuple, List, Union

import numpy as np


class NodeData:

    def __init__(
            self,
            key: np.ndarray,
            value: Union[float, np.ndarray]
    ):
        self.key = key
        self.value = value

    def update_values(self, targets: np.ndarray):
        if type(self.value) is float:
            self.value = targets[self.key].mean()
        else:
            node_targets = targets[self.key]
            num_samples = node_targets.shape[0]
            for k in range(self.value.shape[0]):
                self.value[k] = np.sum(node_targets == k) / num_samples if num_samples > 0 else 0


class Tree:

    def __init__(
            self,
            data: NodeData,
            split_feature: Optional[int] = None,
            split_threshold: Optional[float] = None,
            left: Optional[Tree] = None,
            right: Optional[Tree] = None,
            root_depth: Optional[int] = 0,
    ):
        self.data = data
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right
        self.root_depth = root_depth

    def is_leaf(self) -> bool:
        return (
                (self.split_feature is None)
                or (self.split_threshold is None)
                or (self.left is None)
                or (self.right is None)
        )

    def is_pure(self) -> bool:
        if self.data.value is float:
            return False
        return np.max(self.data.value) == 1

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
            self.left.get_subtrees(subtrees=subtrees)
            self.right.get_subtrees(subtrees=subtrees)
        return subtrees

    def get_leaf_data(self, leaf_data: Optional[List] = None) -> List[NodeData]:
        if leaf_data is None:
            leaf_data = []
        if self.is_leaf():
            leaf_data.append(self.data)
        else:
            self.left.get_leaf_data(leaf_data=leaf_data)
            self.right.get_leaf_data(leaf_data=leaf_data)
        return leaf_data

    def update_splits(
            self,
            inputs: np.ndarray,
            targets: Optional[np.ndarray] = None,
            split_feature: Optional[int] = None,
            split_threshold: Optional[float] = None,
            update_leaf_values_only: bool = False
    ) -> None:

        # Update split feature of root if provided
        if split_feature:
            self.split_feature = split_feature

        # Update root split threshold of root if provided
        if split_threshold:
            self.split_threshold = split_threshold

        # Recursively partition the data
        if not self.is_leaf():

            # Update keys of children's node data
            mask = inputs[:, self.split_feature] < self.split_threshold
            self.left.data.key = mask * self.data.key
            self.right.data.key = ~mask * self.data.key

            # Update values of children's node data if desired
            if targets is not None:
                for child in self.get_children():
                    if child.is_leaf() or not update_leaf_values_only:
                        child.data.update_values(targets=targets)

            self.left.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=update_leaf_values_only)
            self.right.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=update_leaf_values_only)

    def update_depth(self) -> None:
        for child in self.get_children():
            if child:
                child.root_depth = self.root_depth + 1
                child.update_depth()




