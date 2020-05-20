import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def accuracy(predicted: np.ndarray, targets: np.ndarray) -> float:
    return accuracy_score(y_true=targets, y_pred=predicted.argmax(1))


def roc_auc(predicted: np.ndarray, targets: np.ndarray) -> float:
    predicted = predicted[:, np.unique(targets)]
    if predicted.shape[1] > 2:
        return roc_auc_score(y_true=targets, y_score=predicted, multi_class="ovr", average="macro")
    if predicted.shape[1] == 2:
        return roc_auc_score(y_true=targets, y_score=predicted[:, 1])
    return 1.0


def average_precision(predicted: np.ndarray, targets: np.ndarray) -> float:
    if predicted.shape[1] > 2:
        raise NotImplementedError
    if predicted.shape[1] == 2:
        return average_precision_score(y_true=targets, y_score=predicted[:, 1])
    return 1.0


def weighted_gini_purity(predicted: np.ndarray, targets: np.ndarray) -> float:
    return (predicted * predicted).sum(1).mean()

