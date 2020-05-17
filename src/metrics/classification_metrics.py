from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from models.tree import Tree


def predict_probs(tree: Tree, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    targets = targets[tree.data]
    classes = np.unique(targets)
    probs = np.zeros((targets.shape[0], classes.shape[0]))
    for leaf_data in tree.get_leaf_data():
        leaf_data = leaf_data[tree.data]
        for k, class_id in enumerate(classes):
            probs[leaf_data, k] = (leaf_data * (targets == class_id)).sum() / leaf_data.sum()
    return targets, probs


def accuracy(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    return accuracy_score(y_true=targets, y_pred=probs.argmax(1))


def roc_auc(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
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


def weighted_gini_impurity(tree: Tree, targets: np.ndarray) -> float:
    targets, probs = predict_probs(tree=tree, targets=targets)
    return (1 - (probs * probs).sum(1)).mean()

