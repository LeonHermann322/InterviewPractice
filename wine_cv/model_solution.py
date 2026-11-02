"""Reference orchestration; supplied helpers own the ordinary batch loops."""
from collections.abc import Sequence
from copy import deepcopy

import torch

from . import support
from .exercise import WineCVExercise
from .support import FoldResult, TrainingConfig, TrialResult


class ModelSolution(WineCVExercise):
    @staticmethod
    def standardize(
        train_x: torch.Tensor, val_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = train_x.mean(dim=0)
        scale = train_x.std(dim=0, correction=0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return (train_x - mean) / scale, (val_x - mean) / scale

    @staticmethod
    def fit_fold(
        train_x: torch.Tensor, train_y: torch.Tensor,
        val_x: torch.Tensor, val_y: torch.Tensor,
        config: TrainingConfig, seed: int, device: torch.device,
    ) -> FoldResult:
        torch.manual_seed(seed)
        model = support.create_model().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        loader = support.make_loader(train_x, train_y, config, seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state = deepcopy(model.state_dict())
        stale_epochs = 0
        for epoch in range(1, config.max_epochs + 1):
            support.train_epoch(model, loader, optimizer, device)
            loss, _ = support.evaluate(model, val_x, val_y, device)
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return FoldResult(model, best_epoch, best_loss)

    def cross_validate(
        self, x: torch.Tensor, y: torch.Tensor, config: TrainingConfig,
        n_splits: int = 5, seed: int = 42, device: torch.device = torch.device("cpu"),
    ) -> TrialResult:
        losses, epochs = [], []
        oof_logits = torch.empty(len(y), 3, dtype=torch.float32)
        for fold_index, (train, val) in enumerate(support.stratified_folds(y, n_splits, seed)):
            train_x, val_x = self.standardize(x[train], x[val])
            result = self.fit_fold(
                train_x, y[train], val_x, y[val], config, seed + fold_index, device,
            )
            loss, logits = support.evaluate(result.model, val_x, y[val], device)
            losses.append(loss)
            epochs.append(result.best_epoch)
            oof_logits[val] = logits
        return TrialResult(config, losses, epochs, oof_logits)

    @staticmethod
    def select_best(trials: Sequence[TrialResult]) -> TrialResult:
        if not trials:
            raise ValueError("At least one trial is required")
        return min(trials, key=lambda trial: trial.mean_loss)
