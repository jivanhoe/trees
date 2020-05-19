from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from models.tree import Tree


def predict_probs(tree: Tree, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = tree.data.value.shape[0]
    probs = np.zeros((targets.shape[0], num_classes))
    for leaf_data in tree.get_leaf_data():
        for k in range(num_classes):
            probs[leaf_data.key, k] = leaf_data.value[k]
    return targets[tree.data.key], probs[tree.data.key]


def accuracy(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    return accuracy_score(y_true=targets, y_pred=probs.argmax(1))


def roc_auc(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    probs = probs[:, np.any(probs > 0, axis=0)]
    if probs.shape[1] > 2:
        return roc_auc_score(y_true=targets, y_score=probs, multi_class="ovr", average="macro")
    if probs.shape[1] == 2:
        return roc_auc_score(y_true=targets, y_score=probs[:, 1])
    return 1.0


def average_precision(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    if probs.shape[1] > 2:
        raise NotImplementedError
    if probs.shape[1] == 2:
        return average_precision_score(y_true=targets, y_score=probs[:, 1])
    return 1.0


def weighted_gini_purity(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    return (probs * probs).sum(1).mean()

