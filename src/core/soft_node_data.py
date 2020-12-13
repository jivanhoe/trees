from typing import Optional

import numpy as np

from core.node_data import NodeData


class SoftNodeData(NodeData):

    def __init__(
            self,
            key: np.ndarray,
            value: np.ndarray,
            is_classifier: bool
    ):
        super().__init__(
            key=key,
            value=value,
            is_classifier=is_classifier
        )

    def update_value(
            self,
            targets: np.ndarray,
            sample_weights: Optional[np.ndarray] = None
    ) -> None:
        self.n_train_samples = int(np.round(np.exp(self.key).sum()))
        weights = np.exp(self.key) * (sample_weights if sample_weights is not None else 1)
        if self.is_classifier:
            if self.n_train_samples > 0:
                for k in range(self.value.shape[0]):
                    self.value[k] = (weights * (targets == k)).sum() / weights.sum()
            else:
                self.value[:] = np.nan
        else:
            self.value = (weights * targets).mean() / weights.sum()