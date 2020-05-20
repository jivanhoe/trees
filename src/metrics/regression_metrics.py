import numpy as np
from sklearn.metrics import mean_squared_error as mse_score, r2_score

from models.tree import Tree


def get_regression_values(tree: Tree, targets: np.ndarray):
    predictions = np.zeros(targets.shape)
    for leaf_data in tree.get_leaf_data():
        predictions[leaf_data.key] = leaf_data.value.mean()
    return targets[tree.data.key], predictions[tree.data.key]


def mean_squared_error(tree: Tree, targets: np.ndarray) -> float:
    targets, predictions = get_regression_values(tree=tree, targets=targets)
    return mse_score(y_true=targets, y_pred=predictions)


def r_squared(tree: Tree, targets: np.ndarray) -> float:
    targets, predictions = get_regression_values(tree=tree, targets=targets)
    return r2_score(y_true=targets, y_pred=predictions)

