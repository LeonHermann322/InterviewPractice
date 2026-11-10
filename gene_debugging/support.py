"""Supplied data and encoding; no candidate or reference implementations."""
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


SEQUENCES = ("AAAA", "AAAT", "AACC", "ACGT", "CCCG", "GCGC", "TTTT", "GGTT")


def encode_sequences(sequences: Sequence[str]) -> torch.Tensor:
    """Return float32 (batch, 4, length) one-hot DNA in A/C/G/T order."""
    if not sequences or not sequences[0]:
        raise ValueError("Provide nonempty sequences")
    if any(len(seq) != len(sequences[0]) for seq in sequences):
        raise ValueError("Sequences must have equal lengths")
    alphabet = {base: index for index, base in enumerate("ACGT")}
    try:
        indices = torch.tensor([[alphabet[base] for base in seq] for seq in sequences])
    except KeyError as error:
        raise ValueError(f"Unknown DNA base: {error.args[0]}") from error
    return F.one_hot(indices, num_classes=4).permute(0, 2, 1).float()


def regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return encoded strings and GC fractions with target shape (batch,)."""
    x = encode_sequences(SEQUENCES)
    return x, x[:, 1:3].sum(dim=1).mean(dim=1)


class SuppliedRegressor(nn.Module):
    """A working model for part 1, independent of the part 2 tasks."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=-1))
