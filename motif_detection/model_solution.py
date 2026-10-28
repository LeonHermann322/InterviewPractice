"""Reference solution for exercise 02: CNN motif detection on variable-length DNA."""
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .exercise import MotifExercise


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def reverse_complement(seq: str) -> str:
    seq = seq.upper()
    try:
        return "".join(COMPLEMENT[b] for b in reversed(seq))
    except KeyError as exc:
        raise ValueError(f"Invalid base: {exc.args[0]}") from None


def one_hot_channels_first(seq: str) -> torch.Tensor:
    seq = seq.upper()
    out = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, base in enumerate(seq):
        if base == "N":
            out[:, i] = 0.25
        elif base in BASE_TO_IDX:
            out[BASE_TO_IDX[base], i] = 1.0
        else:
            raise ValueError(f"Invalid base: {base}")
    return out


def pad_batch(sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(sequences) == 0:
        raise ValueError("Empty batch")
    max_len = max(len(s) for s in sequences)
    X = torch.zeros(len(sequences), 4, max_len, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, s in enumerate(sequences):
        X[i, :, : len(s)] = one_hot_channels_first(s)
        mask[i, : len(s)] = True
    return X, mask


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str], labels: Sequence[int]):
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        self.sequences = list(sequences)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.sequences[idx], float(self.labels[idx])


def collate_sequences(batch):
    seqs, labels = zip(*batch)
    X, mask = pad_batch(seqs)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return X, mask, y


class MotifCNN(nn.Module):
    def __init__(self, n_filters: int = 16, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(1), float("-inf"))
        pooled = h.max(dim=2).values
        return self.fc(self.dropout(pooled))


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.flatten()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0:
        raise ValueError("No positive examples")
    return (n_neg.float() / n_pos.float()).reshape(1)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.flatten().float()
    labels = labels.flatten()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC requires both classes")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    return (wins / (len(pos) * len(neg))).item()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for X, mask, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X, mask), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * X.shape[0]
        n += X.shape[0]
    return total / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()
    total, n = 0.0, 0
    all_logits, all_labels = [], []
    for X, mask, y in loader:
        logits = model(X, mask)
        total += loss_fn(logits, y).item() * X.shape[0]
        n += X.shape[0]
        all_logits.append(logits)
        all_labels.append(y)
    return total / n, auroc(torch.cat(all_logits), torch.cat(all_labels))


class ModelSolution(MotifExercise):
    """Reference implementation of the exercise's abstract stubs."""

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
