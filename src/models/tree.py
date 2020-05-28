from __future__ import annotations
from typing import Optional, Tuple, List, Union

import numpy as np


class NodeData:

    def __init__(
            self,
            key: np.ndarray,
            value: np.ndarray,
            is_classifier: bool
    ):
        self.key = key
        self.value = value
        self.is_classifier = is_classifier
        self.n_train_samples = self.key.sum()

    def update_value(self, targets: np.ndarray) -> None:
        if self.is_classifier:
            self.n_train_samples = self.key.sum()
            node_targets = targets[self.key]
            for k in range(self.value.shape[0]):
                self.value[k] = np.sum(node_targets == k) / self.n_train_samples if self.n_train_samples > 0 else 0
        else:
            self.value = targets[self.key]


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
        if self.data.is_classifier:
            return np.max(self.data.value) == 1
        return np.std(self.data.value) == 0

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

    def get_subtrees(self, subtrees: Optional[List[Tree]] = None) -> List[Tree]:
        if subtrees is None:
            subtrees = []
        subtrees.append(self)
        if not self.is_leaf():
            for child in self.get_children():
                child.get_subtrees(subtrees=subtrees)
        return subtrees

    def get_leaves(self, leaves: Optional[List[Tree]] = None) -> List[Tree]:
        if leaves is None:
            leaves = []
        if self.is_leaf():
            leaves.append(self)
        else:
            for child in self.get_children():
                child.get_leaves(leaves=leaves)
        return leaves

    def get_leaf_parents(self, leaf_parents: Optional[List[Tree]] = None) -> List[Tree]:
        if leaf_parents is None:
            leaf_parents = []
        if not self.is_leaf():
            if self.left.is_leaf() or self.right.is_leaf():
                leaf_parents.append(self)
            for child in self.get_children():
                child.get_leaf_parents(leaf_parents=leaf_parents)
        return leaf_parents

    def delete_children(self):
        self.left, self.right, self.split_feature, self.split_threshold = None, None, None, None

    def get_min_leaf_size(self) -> int:
        return min(int(leaf.data.n_train_samples) for leaf in self.get_leaves())

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
                        child.data.update_value(targets=targets)
            for child in self.get_children():
                child.update_splits(inputs=inputs, targets=targets, update_leaf_values_only=update_leaf_values_only)

    def update_depth(self) -> None:
        for child in self.get_children():
            if child:
                child.root_depth = self.root_depth + 1
                child.update_depth()

    def predict(self, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        if inputs is not None:
            self.data.key = np.ones(inputs.shape[0], dtype=bool)
            self.update_splits(inputs=inputs)
        if self.data.is_classifier:
            n_classes = self.data.value.shape[0]
            predicted = np.zeros((self.data.key.shape[0], n_classes))
            for leaf in self.get_leaves():
                for k in range(n_classes):
                    predicted[leaf.data.key, k] = leaf.data.value[k]
        else:
            predicted = np.zeros(inputs.shape)
            for leaf in self.get_leaves():
                predicted[leaf.data.key] = leaf.data.value.mean()
        return predicted[self.data.key]




