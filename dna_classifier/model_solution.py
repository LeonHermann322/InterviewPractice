"""Reference solution for the PyTorch interview practice exercise."""
from typing import Sequence
import torch
from torch import nn

from .exercise import DNAExercise


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(seq: str) -> torch.Tensor:
    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    if any(base not in BASE_TO_IDX for base in seq):
        raise ValueError("Sequence may contain only A, C, G, T")
    encoded = torch.zeros((len(seq), 4), dtype=torch.float32)
    for i, base in enumerate(seq):
        encoded[i, BASE_TO_IDX[base]] = 1.0
    return encoded.flatten()


def encode_sequences(sequences: Sequence[str]) -> torch.Tensor:
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError("All sequences must have equal length")
    return torch.stack([encode_sequence(seq) for seq in sequences])


class DNAClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1,
) -> list[float]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    y = y.to(dtype=torch.float32).reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of examples")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        if logits.shape != y.shape:
            raise ValueError(f"Expected logits shape {tuple(y.shape)}, got {tuple(logits.shape)}")
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return losses


def predict_labels(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    if was_training:
        model.train()
    return preds


class ModelSolution(DNAExercise):
    """Reference implementation of the exercise's abstract stubs."""

    encode_sequence = staticmethod(encode_sequence)
    encode_sequences = staticmethod(encode_sequences)
    create_model = staticmethod(DNAClassifier)
    train_model = staticmethod(train_model)
    predict_labels = staticmethod(predict_labels)
