from typing import Optional

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

    def update_value(self, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:
        node_sample_weights = sample_weights[self.key] if sample_weights is not None else np.ones(self.n_train_samples)
        node_targets = targets[self.key]
        self.n_train_samples = int(self.key.sum())
        if self.is_classifier:
            if self.n_train_samples > 0:
                for k in range(self.value.shape[0]):
                    self.value[k] = node_sample_weights[node_targets == k].sum() / node_sample_weights.sum()
            else:
                self.value[:] = np.nan
        else:
            self.value = (node_targets * node_sample_weights).sum() / node_sample_weights.sum()