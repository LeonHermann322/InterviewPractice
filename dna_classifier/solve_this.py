"""Candidate implementation file.

Implement the TODOs below. UserSolution binds them to DNAExercise's abstract
methods. Keep names and signatures unchanged; no reference code is imported.
"""
from collections.abc import Sequence

import torch
from torch import nn

from .exercise import DNAExercise

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(seq: str) -> torch.Tensor:
    """One-hot encode a DNA sequence.

    Args:
        seq: String containing only A, C, G, T.

    Returns:
        Float tensor of shape (len(seq) * 4,), with one 1 per nucleotide.
    """
    raise NotImplementedError("Implement encode_sequence")


def encode_sequences(sequences: Sequence[str]) -> torch.Tensor:
    """Encode equal-length DNA sequences into a batch.

    Returns a float tensor of shape (batch_size, sequence_length * 4).
    """
    raise NotImplementedError("Implement encode_sequences")


class DNAClassifier(nn.Module):
    """MLP mapping one-hot encoded DNA sequences to a single logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        # TODO: define layers. Expected architecture:
        # Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
        raise NotImplementedError("Implement DNAClassifier.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch_size, 1)."""
        raise NotImplementedError("Implement DNAClassifier.forward")


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1,
) -> list[float]:
    """Train binary classifier using BCEWithLogitsLoss and SGD.

    y may be shape (N,) or (N, 1); normalize it to match model output.
    Return one scalar loss value per epoch.
    """
    raise NotImplementedError("Implement train_model")


def predict_labels(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Return binary predictions (0/1) with shape (N, 1), without gradients."""
    raise NotImplementedError("Implement predict_labels")


class UserSolution(DNAExercise):
    """Override the exercise's abstract stubs with the implementations above."""

    encode_sequence = staticmethod(encode_sequence)
    encode_sequences = staticmethod(encode_sequences)
    create_model = staticmethod(DNAClassifier)
    train_model = staticmethod(train_model)
    predict_labels = staticmethod(predict_labels)
