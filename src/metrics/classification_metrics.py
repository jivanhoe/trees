from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, log_loss
from torch.nn import NLLLoss


def accuracy(predicted: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    return accuracy_score(y_true=targets, y_pred=predicted.argmax(1), sample_weight=sample_weights)


def cross_entropy(predicted: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    return -log_loss(y_true=targets, y_pred=predicted, sample_weight=sample_weights)


def gini(predicted: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray]) -> float:
    if sample_weights is None:
        sample_weights = np.ones(targets.shape[0])
    return ((predicted * predicted).sum(1) * sample_weights).mean() / sample_weights.sum()


def roc_auc(predicted: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    predicted = predicted[:, np.unique(targets)]
    if predicted.shape[1] > 2:
        return roc_auc_score(y_true=targets, y_score=predicted, sample_weight=sample_weights,
                             multi_class="ovr", average="macro")
    if predicted.shape[1] == 2:
        return roc_auc_score(y_true=targets, y_score=predicted[:, 1], sample_weight=sample_weights)
    return 1.0


def pr_auc(predicted: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    predicted = predicted[:, np.unique(targets)]
    if predicted.shape[1] > 2:
        raise NotImplementedError
    if predicted.shape[1] == 2:
        return average_precision_score(y_true=targets, y_score=predicted[:, 1], sample_weight=sample_weights)
    return 1.0


def torch_cross_entropy(predicted: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:

    return NLLLoss()(torch.log(predicted + 1e-10), targets)


def torch_gini(predicted: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    if sample_weights is None:
        sample_weights = torch.ones(targets.shape[0])
    return - (sample_weights[:, None] * predicted * predicted).sum()


accuracy.__name__ = "accuracy"
cross_entropy.__name__ = "cross_entropy"
gini.__name__ = "gini"
roc_auc.__name__ = "roc_auc"
pr_auc.__name__ = "pr_auc"
torch_cross_entropy.__name__ = "cross_entropy"
torch_gini.__name__ = "gini"


CLASSIFICATION_CRITERION_LOOKUP = {
    "accuracy": accuracy,
    "cross_entropy": cross_entropy,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "gini": gini
}

TORCH_CLASSIFICATION_CRITERION_LOOKUP = {
    "cross_entropy": torch_cross_entropy,
    "gini": torch_gini,
}

