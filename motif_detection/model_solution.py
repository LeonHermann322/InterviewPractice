"""Reference solution for exercise 02: CNN motif detection on variable-length DNA."""
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .exercise import MotifExercise

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def reverse_complement(seq: str) -> str:
    """Return a DNA string's reverse complement.

    Uppercase first so lowercase input uses the same lookup table. Then walk
    the bases in reverse order and replace each with its complement. Convert
    an unknown-base lookup failure into a clear ``ValueError``.
    """
    seq = seq.upper()
    try:
        return "".join(COMPLEMENT[b] for b in reversed(seq))
    except KeyError as exc:
        raise ValueError(f"Invalid base: {exc.args[0]}") from None


def one_hot_channels_first(seq: str) -> torch.Tensor:
    """Encode a DNA string as a ``(4, length)`` tensor for ``Conv1d``.

    Uppercase the input and initialize the output to zeros. For each position,
    set its matching A/C/G/T channel to 1; for N, assign 0.25 to all channels
    to represent an unknown base without favoring one nucleotide. Reject any
    other character.
    """
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
    """Encode variable-length strings into a padded tensor and validity mask.

    Reject an empty batch, find its longest sequence, and allocate zero-filled
    tensors sized for that length. Encode each sequence into the left portion
    of its batch row and mark those real positions True; untouched padding
    remains zero and False. Return ``X`` shaped ``(batch, 4, max_length)`` and
    ``mask`` shaped ``(batch, max_length)``.
    """
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
    """Store sequence-label pairs for use by a PyTorch ``DataLoader``."""

    def __init__(self, sequences: Sequence[str], labels: Sequence[int]):
        """Check pair counts, then copy both inputs into lists for indexed access."""
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        self.sequences = list(sequences)
        self.labels = list(labels)

    def __len__(self) -> int:
        """Return the number of examples, as required by the Dataset protocol."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        """Fetch one pair and cast its label to float for the loss function."""
        return self.sequences[idx], float(self.labels[idx])


def collate_sequences(batch):
    """Combine dataset examples into the three tensors used by the CNN.

    Separate the sequence and label values, encode and pad the sequences with
    ``pad_batch``, then make labels float32 and reshape them to ``(batch, 1)``.
    Return the encoded inputs, their padding mask, and the labels.
    """
    seqs, labels = zip(*batch)
    X, mask = pad_batch(seqs)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return X, mask, y


class MotifCNN(nn.Module):
    """Detect sequence patterns by scanning and pooling convolutional features.

    Each filter can respond to a local motif. Global max pooling keeps the
    strongest response from each filter, and the final linear layer combines
    those responses into a sequence-level logit.
    """

    def __init__(self, n_filters: int = 16, kernel_size: int = 7, dropout: float = 0.2):
        """Create the convolution, regularization, and classification layers.

        Require an odd kernel size so symmetric padding preserves sequence
        length. The convolution reads four nucleotide channels, and its filter
        outputs feed dropout and a one-logit linear head.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Turn a padded sequence batch into one raw classification logit each.

        Convolve across positions and apply ReLU. If a mask is provided, replace
        padded-position activations with negative infinity so max pooling cannot
        select padding. Pool over sequence length, apply dropout, then map the
        pooled filter responses to logits shaped ``(batch, 1)``.
        """
        h = torch.relu(self.conv(x))
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(1), float("-inf"))
        pooled = h.max(dim=2).values
        return self.fc(self.dropout(pooled))


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Return ``negative_count / positive_count`` for weighted binary loss.

    Flatten labels, count each class, reject data with no positives (which
    would make the ratio undefined), and return the ratio as a one-element
    float tensor suitable for ``BCEWithLogitsLoss``.
    """
    labels = labels.flatten()
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0:
        raise ValueError("No positive examples")
    return (n_neg.float() / n_pos.float()).reshape(1)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUROC as the fraction of correctly ordered positive-negative pairs.

    Flatten scores and labels, split scores by class, and require both classes.
    Compare every positive score with every negative score: a greater positive
    earns one win, an equal score earns half, and a lower score earns none.
    Divide by the total number of pairs and return a Python float.
    """
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
    """Train over all batches once and return the sample-weighted mean loss.

    Enable training mode. For each batch, clear gradients, run the masked
    forward pass, compute loss, backpropagate, and step the optimizer. Accumulate
    each batch's mean loss multiplied by its sample count, then divide by the
    total number of samples so a smaller final batch is weighted correctly.
    """
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
    """Evaluate the full loader and return sample-weighted loss and AUROC.

    The decorator disables gradient tracking, and evaluation mode disables
    training-only layer behavior such as dropout. For each batch, compute
    logits and loss, accumulate sample-weighted loss, and retain logits and
    labels. Concatenate all batches before calculating AUROC so it reflects
    the whole dataset rather than an average of batch-level AUROCs.
    """
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
    """Provide the reference helpers and CNN as the contract's implementation.

    Static bindings expose the helper functions through the exercise instance;
    ``create_dataset`` and ``create_model`` bind the corresponding classes.
    """

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
