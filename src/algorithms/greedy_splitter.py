from typing import Callable, Optional, Tuple

import numpy as np


class GreedySplitter:

    def __init__(
            self,
            is_classifer: bool,
            criterion: Callable,
            max_features: float,
            max_candidate_thresholds: Optional[int] = None
    ):
        self.is_classifer = is_classifer
        self.criterion = criterion
        self.max_features = max_features
        self.max_candidate_thresholds = max_candidate_thresholds
        self.split_feature = None
        self.split_threshold = None

    def predict(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> np.ndarray:

        # Create split mask
        mask = inputs[:, self.split_feature] < self.split_threshold

        # Compute predictions
        if self.is_classifer:
            n_classes = np.unique(targets).shape[0]
            n_samples = inputs.shape[0]
            left = np.zeros((n_samples, n_classes))
            right = np.zeros((n_samples, n_classes))
            for k in range(n_classes):
                left[:, k] = (sample_weights * (targets == k))[mask].sum() / sample_weights[mask].sum()
                right[:, k] = (sample_weights * (targets == k))[~mask].sum() / sample_weights[~mask].sum()
            return np.where(mask[:, None], left, right)
        else:
            left = (sample_weights * targets)[mask].sum() / sample_weights[mask].sum()
            right = (sample_weights * targets)[~mask].sum() / sample_weights[~mask].sum()
            return mask * left * + ~mask * right

    def _get_candidate_thresholds_for_feature(self, inputs: np.ndarray) -> np.ndarray:

        # Get candidate thresholds for split
        feature_values = np.unique(inputs[:, self.split_feature])
        candidate_thresholds = (feature_values[1:] + feature_values[:-1]) / 2

        # If there are more candidates than permitted, filter the candidates array
        if self.max_candidate_thresholds:
            n_candidates = candidate_thresholds.shape[0]
            if n_candidates > self.max_candidate_thresholds:
                step = n_candidates / self.max_candidate_thresholds
                selected = np.round(np.arange(self.max_candidate_thresholds) * step).astype(int)
                candidate_thresholds = candidate_thresholds[selected]

        return candidate_thresholds

    def _optimize_split_for_feature(
            self,
            split_feature: int,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> Tuple[float, float]:

        # Initialize variables
        self.split_feature = split_feature
        best_split_threshold_for_feature = None
        max_score_for_feature = -np.inf

        # Perform grid search for best split threshold
        for split_threshold in self._get_candidate_thresholds_for_feature(inputs=inputs):
            self.split_threshold = split_threshold
            predicted = self.predict(inputs=inputs, targets=targets, sample_weights=sample_weights)
            score = self.criterion(predicted=predicted, targets=targets, sample_weights=sample_weights)
            if score > max_score_for_feature:
                max_score_for_feature = score
                best_split_threshold_for_feature = split_threshold
        return best_split_threshold_for_feature, max_score_for_feature

    def optimize_split(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> Tuple[int, float]:

        # Select features to consider for split
        features = np.arange(inputs.shape[1])
        if self.max_features:
            np.random.shuffle(features)
            features = features[:int(np.ceil(self.max_features * features.shape[0]))]

        # Initialize variables
        best_split_feature = None
        best_split_threshold = None
        max_score = -np.inf

        # Perform grid search for best split feature and threshold pairing
        for split_feature in features:
            split_threshold, score = self._optimize_split_for_feature(
                split_feature=split_feature,
                inputs=inputs,
                targets=targets,
                sample_weights=sample_weights
            )
            if score > max_score:
                best_split_feature = split_feature
                best_split_threshold = split_threshold
                max_score = score

        # Set best split feature and threshold and return
        self.split_feature = best_split_feature
        self.split_threshold = best_split_threshold
        return self.split_feature, self.split_threshold

