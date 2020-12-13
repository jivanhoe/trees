import logging
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SoftSplitter(nn.Module):

    def __init__(
            self,
            is_classifier: bool,
            gating_param: float,
            criterion: Callable,
            solver: str = "lbfgs",
            max_iter: int = 2,
            tol: float = 1e-3,
            eps: float = 1e-3,
            cutoff_weight: float = 0,
            verbose: bool = False,
    ):
        super().__init__()
        self.is_classifier = is_classifier
        self.gating_param = gating_param
        self.criterion = criterion
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.cutoff_weight = cutoff_weight
        self.verbose = verbose
        self.split_feature = None
        self.split_threshold = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.trajectories_ = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:

        # Compute weights
        split_weights = torch.clamp(
            1 / (1 + torch.exp(-self.gating_param * (inputs[:, self.split_feature] - self.split_threshold))),
            min=self.eps,
            max=1 - self.eps,
        )
        left_weights = sample_weights * split_weights
        right_weights = sample_weights * (1 - split_weights)

        # Compute predictions
        if self.is_classifier:
            n_classes = torch.unique(targets).shape[0]
            left = torch.zeros(n_classes)
            right = torch.zeros(n_classes)
            for k in range(n_classes):
                left[k] = (left_weights * (targets == k)).sum() / left_weights.sum()
                right[k] = (right_weights * (targets == k)).sum() / right_weights.sum()
            return split_weights[:, None] * left + (1 - split_weights)[:, None] * right
        else:
            left = (left_weights * targets).sum() / left_weights.sum()
            right = (right_weights * targets).sum() / left_weights.sum()
            return split_weights * left + (1 - split_weights) * right

    def _optimize_split_for_feature(
            self,
            split_feature: int,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            sample_weights: torch.Tensor
    ) -> Tuple[nn.Parameter, float]:

        # Initialize variables
        self.split_feature = split_feature
        self.split_threshold = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        if self.solver == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters())
        elif self.solver == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        else:
            raise NotImplementedError(f"Unable to recognise '{self.solver}' solver - select 'lbfgs' or 'adam'")
        prev_loss = np.inf
        losses = []
        params = []

        # Perform gradient descent
        for i in range(self.max_iter):

            # Make gradient step
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                predicted = self(inputs=inputs, targets=targets, sample_weights=sample_weights)
                loss = self.criterion(predicted=predicted, targets=targets, sample_weights=sample_weights)
                if loss.requires_grad:
                    loss.backward()
                losses.append(torch.mean(loss).item())
                return loss
            optimizer.step(closure)
            params.append(self.split_threshold.item())

            # Check for convergence
            if abs(prev_loss - losses[-1]) < self.tol:
                break
            else:
                prev_loss = losses[-1]

        # Store trajectory
        self.trajectories_[split_feature] = {"loss": losses, "param": params}

        # Return optimized threshold value  and corresponding loss
        return self.split_threshold,  losses[-1]

    def optimize_split(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            sample_weights: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:

        # Filter data
        mask = sample_weights > self.cutoff_weight
        inputs = inputs[mask]
        targets = targets[mask]
        sample_weights = sample_weights[mask]

        if np.unique(targets).shape[0] > 1:

            # Convert numpy arrays to tensors
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            sample_weights = torch.from_numpy(sample_weights)

            min_loss = np.inf
            best_split_feature = None
            best_split_threshold = None
            for split_feature in range(inputs.shape[1]):

                # Try gradient descent procedure
                try:

                    # Optimize threshold for selected
                    threshold, loss = self._optimize_split_for_feature(
                        split_feature=split_feature,
                        inputs=inputs,
                        targets=targets,
                        sample_weights=sample_weights
                    )

                    # Update best feature and threshold if improvement
                    if loss < min_loss:
                        min_loss = loss
                        best_split_feature = split_feature
                        best_split_threshold = threshold

                # Catch if gradient descent diverges
                except ValueError:
                    pass

            self.split_feature = best_split_feature
            self.split_threshold = best_split_threshold
            return self.split_feature, self.split_threshold.item()

        return None, None








