"""Reference answers; keep closed during timed practice."""
import torch
from torch import nn

from .exercise import GeneDebuggingExercise
from .support import LOSS_CASES


def regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, case: str,
) -> torch.Tensor:
    if case not in LOSS_CASES:
        raise ValueError(f"Unknown loss case: {case}")
    return (prediction - target).square().mean()


def train_model(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor,
    case: str, epochs: int = 80, lr: float = 0.1,
) -> list[float]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    target = target.reshape(-1, 1)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = regression_loss(prediction, target, case)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class GCRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=-1))


class MotifClassifier(nn.Module):
    def __init__(self, filters: int = 8, kernel_size: int = 3):
        super().__init__()
        self.scanner = nn.Conv1d(4, filters, kernel_size)
        self.head = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.scanner(x)).amax(dim=-1)
        return self.head(features)


class ResidualBlock(nn.Module):
    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.expand = nn.Linear(features, hidden_dim)
        self.project = nn.Linear(hidden_dim, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(torch.relu(self.expand(x)))


class ModelSolution(GeneDebuggingExercise):
    regression_loss = staticmethod(regression_loss)
    train_model = staticmethod(train_model)
    create_gc_regressor = staticmethod(GCRegressor)
    create_motif_classifier = staticmethod(MotifClassifier)
    create_residual_block = staticmethod(ResidualBlock)
