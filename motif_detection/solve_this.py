"""Exercise 02: CNN motif detection on variable-length DNA sequences.

Implement the TODOs below. UserSolution binds them to MotifExercise's abstract
methods. Keep names and signatures unchanged; no reference code is imported.
Suggested time box: 35-45 minutes.
"""
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .exercise import MotifExercise


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement (A<->T, C<->G, N->N). Case-insensitive input,
    uppercase output. Raise ValueError for any other character."""
    raise NotImplementedError


def one_hot_channels_first(seq: str) -> torch.Tensor:
    """One-hot encode for Conv1d: float32 tensor of shape (4, len(seq)).

    - Case-insensitive.
    - 'N' (unknown base) is encoded as 0.25 in all four channels.
    - Any other character raises ValueError.
    """
    raise NotImplementedError


def pad_batch(sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode variable-length sequences into a zero-padded batch.

    Returns:
        X:    float32 (batch, 4, max_len), padded positions are all zeros.
        mask: bool    (batch, max_len), True at real positions, False at padding.
    Raise ValueError for an empty batch.
    """
    raise NotImplementedError


class SequenceDataset(Dataset):
    """Dataset yielding (sequence_string, label_as_float).

    Raise ValueError if sequences and labels differ in length.
    """

    def __init__(self, sequences: Sequence[str], labels: Sequence[int]):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[str, float]:
        raise NotImplementedError


def collate_sequences(batch):
    """collate_fn for DataLoader: list of (seq, label) -> (X, mask, y).

    y must be float32 with shape (batch, 1).
    """
    raise NotImplementedError


class MotifCNN(nn.Module):
    """Conv1d motif scanner.

    Architecture:
        Conv1d(4, n_filters, kernel_size, 'same' padding) -> ReLU
        -> global max pool over length, IGNORING padded positions (use mask)
        -> Dropout(dropout) -> Linear(n_filters, 1)

    Attribute names `conv`, `dropout`, `fc` are checked by tests.
    Raise ValueError for an even kernel_size.
    """

    def __init__(self, n_filters: int = 16, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, 4, L), mask: (batch, L) bool or None. Returns logits (batch, 1)."""
        raise NotImplementedError


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Return n_negative / n_positive as a float tensor of shape (1,),
    suitable for BCEWithLogitsLoss(pos_weight=...). Raise ValueError if no positives."""
    raise NotImplementedError


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve, implemented from scratch (no sklearn).

    AUROC = P(score_pos > score_neg), counting ties as 0.5.
    Raise ValueError if only one class is present.
    """
    raise NotImplementedError


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn) -> float:
    """Run one training epoch. Return the sample-weighted mean loss."""
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    """Evaluate without gradients and with dropout disabled.

    Return (sample-weighted mean loss, auroc over the whole loader).
    """
    raise NotImplementedError


class UserSolution(MotifExercise):
    """Override the exercise's abstract stubs with the implementations above."""

    reverse_complement = staticmethod(reverse_complement)
    one_hot_channels_first = staticmethod(one_hot_channels_first)
    pad_batch = staticmethod(pad_batch)
    create_dataset = staticmethod(SequenceDataset)
    collate_sequences = staticmethod(collate_sequences)
    create_model = staticmethod(MotifCNN)
    compute_pos_weight = staticmethod(compute_pos_weight)
    auroc = staticmethod(auroc)
    train_epoch = staticmethod(train_epoch)
    evaluate = staticmethod(evaluate)
