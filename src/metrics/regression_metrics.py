import numpy as np
from sklearn.metrics import mean_squared_error as mse_score, r2_score


def mean_squared_error(predicted: np.ndarray, targets: np.ndarray) -> float:
    return mse_score(y_true=targets, y_pred=predicted)


def r_squared(predicted: np.ndarray, targets: np.ndarray) -> float:
    return r2_score(y_true=targets, y_pred=predicted)

