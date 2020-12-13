from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error as mse_score
from torch.nn import MSELoss
import torch


def mean_squared_error(predicted: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    return mse_score(y_true=targets, y_pred=predicted, sample_weight=weights)


def torch_mean_squared_error(
        predicted: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return MSELoss()(predicted, targets, sample_weights)


REGRESSION_METRICS_LOOKUP = {
    "mse": mean_squared_error
}

TORCH_REGRESSION_LOOKUP = {
    "mse": torch_mean_squared_error
}

