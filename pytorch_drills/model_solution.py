"""Reference solution for Exercise 03: PyTorch tensor and vectorization drills."""
from collections.abc import Sequence

import torch

from .exercise import PyTorchDrillsExercise


def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x + bias


def masked_mean(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores[mask].mean()


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(
        dim=1, unbiased=False, keepdim=True,
    )


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    return ((pred - target).square() * mask).sum() / mask.sum()


def pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, None, :] - b[None, :, :]).square().sum(dim=-1)


def context_mask(n_context: int, n_query: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.ones(n_context + n_query, n_context, dtype=torch.bool),
            torch.zeros(n_context + n_query, n_query, dtype=torch.bool),
        ),
        dim=1,
    )


def causal_mask(n: int) -> torch.Tensor:
    positions = torch.arange(n)
    return positions.unsqueeze(1) >= positions.unsqueeze(0)


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    counts = x.new_zeros(x.shape[0])
    src, dst = edge_index
    out.index_add_(0, dst, x[src])
    counts.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    incoming = counts > 0
    out[incoming] /= counts[incoming].unsqueeze(1)
    return out


def pad_graphs(
    graphs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_n = max(graph.shape[0] for graph in graphs)
    batch = graphs[0].new_zeros((len(graphs), max_n, graphs[0].shape[1]))
    mask = torch.zeros((len(graphs), max_n), dtype=torch.bool, device=graphs[0].device)
    for i, graph in enumerate(graphs):
        batch[i, :graph.shape[0]] = graph
        mask[i, :graph.shape[0]] = True
    return batch, mask


class ModelSolution(PyTorchDrillsExercise):
    """Reference implementation of the exercise's abstract stubs."""

    add_bias = staticmethod(add_bias)
    masked_mean = staticmethod(masked_mean)
    row_zscore = staticmethod(row_zscore)
    masked_mse = staticmethod(masked_mse)
    pairwise_sq = staticmethod(pairwise_sq)
    context_mask = staticmethod(context_mask)
    causal_mask = staticmethod(causal_mask)
    mean_aggregate = staticmethod(mean_aggregate)
    pad_graphs = staticmethod(pad_graphs)
