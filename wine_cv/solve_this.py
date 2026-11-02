"""Candidate tasks for Exercise 05. Batch loops and the model are supplied."""
from collections.abc import Sequence

import torch

from . import support
from .exercise import WineCVExercise
from .support import FoldResult, TrainingConfig, TrialResult


def standardize(
    train_x: torch.Tensor, val_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit feature mean/population std on train_x ONLY; transform both tensors.

    Inputs are finite CPU float32 (N, 13) tensors. Replace zero std by 1.
    Preserve inputs and return new tensors, including for constant columns.
    Do not implement the row-wise normalization from exercise 03.
    """
    raise NotImplementedError("Implement standardize")


def fit_fold(
    train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor,
    config: TrainingConfig, seed: int, device: torch.device,
) -> FoldResult:
    """Train one ALREADY STANDARDIZED fold; see the full contract in --describe.

    Seed torch, build a fresh support.create_model() on device and a fresh AdamW
    using config.lr/weight_decay. Use support.make_loader and supplied
    support.train_epoch/evaluate. Evaluate after every epoch.

    Keep an independent best state on strictly lower validation loss. Stop after
    config.patience consecutive non-improvements or config.max_epochs.
    Restore the best weights (not just the last weights), leave the model in
    eval mode, and return model, one-based best_epoch, and best validation loss.
    No scheduler or mixed precision is needed for this tiny model.
    """
    raise NotImplementedError("Implement fit_fold")


def select_best(trials: Sequence[TrialResult]) -> TrialResult:
    """Return the trial with smallest unweighted mean fold validation loss.

    Keep the first trial on exact ties. Raise ValueError for an empty sequence.
    Do not rank by training loss, accuracy, a single fold, or test performance.
    """
    raise NotImplementedError("Implement select_best")


class UserSolution(WineCVExercise):
    standardize = staticmethod(standardize)
    fit_fold = staticmethod(fit_fold)
    select_best = staticmethod(select_best)

    def cross_validate(
        self, x: torch.Tensor, y: torch.Tensor, config: TrainingConfig,
        n_splits: int = 5, seed: int = 42, device: torch.device = torch.device("cpu"),
    ) -> TrialResult:
        """Use support.stratified_folds(y, n_splits, seed); never touch test data.

        For each fold, call self.standardize and self.fit_fold with
        seed + fold_index (zero based). Use the same split seed for every trial.
        Evaluate each restored model and scatter CPU float32 logits into
        original row order. Return one loss and best_epoch per fold, plus
        detached (N, 3) OOF logits. Every row must be predicted exactly once by
        a model that did not train on it.
        """
        raise NotImplementedError("Implement cross_validate")
