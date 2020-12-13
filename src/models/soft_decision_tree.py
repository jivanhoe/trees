from models.decision_tree import DecisionTree
from algorithms.greedy_search import GreedySearch
from metrics.classification_metrics import CLASSIFICATION_CRITERION_LOOKUP, TORCH_CLASSIFICATION_CRITERION_LOOKUP
from metrics.regression_metrics import REGRESSION_METRICS_LOOKUP,TORCH_REGRESSION_LOOKUP
from typing import Optional, Dict, List
import numpy as np
from copy import deepcopy
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SoftDecisionTree(DecisionTree):

    def __init__(
            self,
            is_classifier: bool,
            max_depth: int = 3,
            min_parent_size: int = 10,
            cutoff_weight: float = 0,
            gating_param_grid: np.ndarray = np.logspace(-1, 1.5, 10),
            training_criterion: str = "gini",
            validation_criterion: str = "roc_auc",
            n_folds: int = 5,
            n_early_stopping_iters: int = 2,
            class_weights: Optional[Dict[int, float]] = None,
            use_balanced_class_weights: bool = False,
            verbose: bool = False
    ):
        super().__init__(
            is_classifier=is_classifier,
            max_depth=max_depth,
            min_parent_size=min_parent_size,
            training_criterion=training_criterion,
            validation_criterion=validation_criterion,
            n_folds=n_folds,
            class_weights=class_weights,
            use_balanced_class_weights=use_balanced_class_weights,
            refit=False,
            verbose=verbose
        )
        self.min_sample_weight = cutoff_weight
        self.n_early_stopping_iters = n_early_stopping_iters
        self.gating_param_grid = gating_param_grid
        self.torch_criterion_callback_ = None
        self.gating_param_cv_scores_ = None
        self.gating_param_ = None

    def _assign_pytorch_criterion_callback(self) -> None:
        if self.is_classifier:
            self.torch_criterion_callback_ = TORCH_CLASSIFICATION_CRITERION_LOOKUP[self.training_criterion]
        else:
            self.torch_criterion_callback_ = TORCH_REGRESSION_LOOKUP[self.training_criterion]

    def _initialize_greedy_search(self) -> GreedySearch:
        return GreedySearch(
            is_classifier=self.is_classifier,
            criterion=self.torch_criterion_callback_,
            max_depth=self.max_depth,
            cutoff_weight=self.min_sample_weight,
            gating_param=self.gating_param_,
            is_soft=True
        )

    def fit(self, inputs: np.ndarray, targets: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:

        self._assign_pytorch_criterion_callback()

        # Initialize grid search
        max_score = -np.inf
        best_gating_param = None
        best_complexity_param = None
        n_iters_since_improvement = 0
        self.gating_param_cv_scores_ = np.zeros((self.gating_param_grid.shape[0], self.n_folds))

        # Perform grid search
        for i, gating_param in tqdm(enumerate(self.gating_param_grid), total=self.gating_param_grid.shape[0]):

            if self.verbose:
                tqdm.write(f"Performing CV with gating param: {'{0:.2e}'.format(gating_param)}")

            # Fit model with incumbent gating parameter
            self.gating_param_ = gating_param
            super().fit(inputs=inputs, targets=targets, sample_weights=sample_weights)

            # Update best parameters if there is an improvement
            if self.score_ > max_score:
                max_score = deepcopy(self.score_)
                best_gating_param = gating_param
                best_complexity_param = deepcopy(self.complexity_param_)
                n_iters_since_improvement = 0
            else:
                n_iters_since_improvement += 1
            if self.verbose:
                tqdm.write(
                    f"Score: {'{0:.3f}'.format(self.score_)} \t "
                    f"Max score: {'{0:.3f}'.format(max_score)} \t "
                )

            # Store results
            self.gating_param_cv_scores_[i, :] = self.complexity_param_cv_scores_[
                self.complexity_param_cv_scores_.mean(axis=1).argmax(), :
            ]

            # Check early stopping criteria
            if n_iters_since_improvement >= self.n_early_stopping_iters:
                self.gating_param_cv_scores_ = self.gating_param_cv_scores_[:i, :]
                if self.verbose:
                    tqdm.write(
                        f"No improvement in {self.n_early_stopping_iters} iterations"
                        f" - terminating grid search"
                    )
                break

        # Refit tree with best parameters using all data
        self.gating_param_ = best_gating_param
        self.complexity_param_ = best_complexity_param
        self._refit_tree(
            inputs=inputs,
            targets=targets,
            sample_weights=self._assign_sample_weights(targets=targets)
        )
