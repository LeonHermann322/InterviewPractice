"""Supplied data, model, and batch loops: these are not candidate tasks."""
import math
from dataclasses import dataclass
from statistics import mean, pstdev

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 12

    def __post_init__(self):
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        for name in ("batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class FoldResult:
    model: nn.Module
    best_epoch: int
    val_loss: float


@dataclass
class TrialResult:
    config: TrainingConfig
    fold_losses: list[float]
    best_epochs: list[int]
    oof_logits: torch.Tensor

    @property
    def mean_loss(self) -> float:
        return mean(self.fold_losses)

    @property
    def std_loss(self) -> float:
        """Population standard deviation across folds, not a confidence interval."""
        return pstdev(self.fold_losses)


def load_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load sklearn's bundled UCI Wine copy; no network access is needed."""
    from sklearn.datasets import load_wine

    x, y = load_wine(return_X_y=True)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def stratified_folds(
    y: torch.Tensor, n_splits: int, seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return CPU train/validation indices, shared across every trial."""
    from sklearn.model_selection import StratifiedKFold

    if y.ndim != 1 or y.dtype != torch.long or y.device.type != "cpu":
        raise ValueError("y must be a one-dimensional CPU int64 tensor")
    _, counts = y.unique(return_counts=True)
    if n_splits < 2 or counts.numel() < 2 or int(counts.min()) < n_splits:
        raise ValueError("Each class must have at least n_splits samples; n_splits >= 2")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (torch.tensor(train), torch.tensor(val))
        for train, val in splitter.split(torch.zeros(len(y), 1).numpy(), y.numpy())
    ]


def create_model() -> nn.Module:
    """Fixed 547-parameter model: 13 -> 32 -> 3 raw multiclass logits."""
    return nn.Sequential(nn.Linear(13, 32), nn.ReLU(), nn.Linear(32, 3))


def make_loader(
    x: torch.Tensor, y: torch.Tensor, config: TrainingConfig, seed: int,
) -> DataLoader:
    """CPU data, shuffled reproducibly; zero workers also works on Windows."""
    return DataLoader(
        TensorDataset(x, y), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Wine fits in one evaluation batch; return loss and detached CPU logits."""
    model.eval()
    logits = model(x.to(device))
    loss = nn.functional.cross_entropy(logits, y.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite evaluation loss")
    return loss.item(), logits.cpu()
