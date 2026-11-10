"""Candidate workspace. Part 1 runs, but contains intentional bugs.

For a timed mock, uncomment one faulty loss formula and select one part-2
module. Repair the active loss to mean squared error and debug train_model.
Do not open model_solution.py until the mock is finished.
"""
import torch
from torch import nn

from .exercise import GeneDebuggingExercise


def regression_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Inputs have identical shapes; return a scalar regression objective."""
    residual = target - prediction
    return residual.mean()
    # return -residual.square().mean()
    # return residual.pow(3).mean()


def train_model(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor,
    epochs: int = 80, lr: float = 0.1,
) -> list[float]:
    """Full-batch SGD; target is (batch,) or (batch, 1), model output (batch, 1).

    Return one pre-update loss per epoch. This is intentionally buggy.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        prediction = model(x)
        loss = regression_loss(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class GCRegressor(nn.Module):
    """Mean over sequence length, then Linear(4, 1); return (batch, 1), no sigmoid."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError("Register the GC regression head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float (batch, 4, length), with length >= 1."""
        raise NotImplementedError("Implement GCRegressor.forward")


class MotifClassifier(nn.Module):
    """Conv1d(4, filters, kernel_size), ReLU, global max, Linear(filters, 1).

    Use valid convolution and biases in both layers; return raw (batch, 1)
    logits. Every sequence is at least kernel_size bases long.
    """

    def __init__(self, filters: int = 8, kernel_size: int = 3):
        super().__init__()
        raise NotImplementedError("Register the motif scanner and output head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement MotifClassifier.forward")


class ResidualBlock(nn.Module):
    """Linear(features, hidden_dim), ReLU, Linear(hidden_dim, features), add input x.

    Use biases in both layers and no activation after the addition.
    Support (batch, features) and (batch, length, features).
    """

    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        raise NotImplementedError("Register the two residual branch layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement ResidualBlock.forward")


class UserSolution(GeneDebuggingExercise):
    regression_loss = staticmethod(regression_loss)
    train_model = staticmethod(train_model)
    create_gc_regressor = staticmethod(GCRegressor)
    create_motif_classifier = staticmethod(MotifClassifier)
    create_residual_block = staticmethod(ResidualBlock)
