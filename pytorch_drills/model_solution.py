"""Reference solution for Exercise 03: PyTorch tensor and vectorization drills."""
from collections.abc import Sequence

import torch

from .exercise import PyTorchDrillsExercise


def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add the feature bias to every row using PyTorch broadcasting.

    The trailing feature dimension of ``x`` matches ``bias``, so no explicit
    repeat or row-wise loop is needed; the returned tensor has ``x``'s shape.
    """
    return x + bias


def masked_mean(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean of only the entries selected by a boolean mask.

    Boolean indexing gathers the True positions into a one-dimensional tensor;
    taking its mean produces the scalar result.
    """
    return scores[mask].mean()


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    """Normalize every row independently to zero mean and unit population spread.

    Compute each row's mean and population standard deviation while keeping
    singleton dimensions for broadcasting. Subtract the mean from each value,
    then divide by that row's standard deviation.
    """
    return (x - x.mean(dim=1, keepdim=True)) / x.std(
        dim=1, unbiased=False, keepdim=True,
    )


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean squared error over positions marked True.

    Subtract targets from predictions and square the differences. The boolean
    mask selects which errors contribute; sum them and divide by the number of
    selected elements to produce a scalar mean.
    """
    return ((pred - target).square() * mask).sum() / mask.sum()


def pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance for every vector pair.

    Add singleton dimensions so broadcasting compares each row of ``a`` with
    each row of ``b``. Square the feature-wise differences and sum across the
    feature dimension, returning an ``(N, M)`` distance matrix.
    """
    return (a[:, None, :] - b[None, :, :]).square().sum(dim=-1)


def context_mask(n_context: int, n_query: int) -> torch.Tensor:
    """Create a mask that lets every row attend to context, but not query columns.

    Construct an all-True context block and an all-False query block, each with
    ``n_context + n_query`` rows. Concatenate them by columns to form the
    square boolean mask.
    """
    return torch.cat(
        (
            torch.ones(n_context + n_query, n_context, dtype=torch.bool),
            torch.zeros(n_context + n_query, n_query, dtype=torch.bool),
        ),
        dim=1,
    )


def causal_mask(n: int) -> torch.Tensor:
    """Build a lower-triangular mask that prevents attending to future positions.

    Make the position indices for a sequence, compare each row index with each
    column index, and mark a pair True when the column is not after the row.
    The diagonal is included, so each position can attend to itself.
    """
    positions = torch.arange(n)
    return positions.unsqueeze(1) >= positions.unsqueeze(0)


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Replace each destination's output with the mean of incoming source features.

    Start with zero outputs and counts. Split ``edge_index`` into source and
    destination rows, then use indexed accumulation to add each source feature
    and increment its destination's count. Divide outputs only for destinations
    with incoming edges; nodes with none remain zero.
    """
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
    """Pad graphs to a shared node count and return a mask for their real nodes.

    Find the largest graph, allocate a zero-filled ``(batch, max_nodes,
    features)`` tensor and a False ``(batch, max_nodes)`` mask. For each graph,
    copy its node rows into the start of its batch slot and mark those positions
    True. Remaining padding stays zero and False.
    """
    max_n = max(graph.shape[0] for graph in graphs)
    batch = graphs[0].new_zeros((len(graphs), max_n, graphs[0].shape[1]))
    mask = torch.zeros((len(graphs), max_n), dtype=torch.bool, device=graphs[0].device)
    for i, graph in enumerate(graphs):
        batch[i, :graph.shape[0]] = graph
        mask[i, :graph.shape[0]] = True
    return batch, mask


class ModelSolution(PyTorchDrillsExercise):
    """Expose the reference drill functions through the exercise contract.

    Static bindings make each function available to the shared test suite
    without duplicating implementation code in this class.
    """

    add_bias = staticmethod(add_bias)
    masked_mean = staticmethod(masked_mean)
    row_zscore = staticmethod(row_zscore)
    masked_mse = staticmethod(masked_mse)
    pairwise_sq = staticmethod(pairwise_sq)
    context_mask = staticmethod(context_mask)
    causal_mask = staticmethod(causal_mask)
    mean_aggregate = staticmethod(mean_aggregate)
    pad_graphs = staticmethod(pad_graphs)
