"""Walkthrough reference solution for Exercise 04 tensor dimension operations."""
import torch

from .exercise import TensorManipulationExercise


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute each x-to-y squared distance without mixing batches.

    The inputs are ``(B, N, D)`` and ``(B, M, D)``. Insert an item axis into
    each tensor so broadcasting creates ``(B, N, M, D)`` differences. Square
    each feature difference, then reduce over D to return ``(B, N, M)``.
    """
    differences = x.unsqueeze(2) - y.unsqueeze(1)
    return differences.square().sum(dim=-1)


def pairwise_cosine_similarity(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """Compare every vector pair in each batch using cosine similarity.

    First compute the pairwise dot products with batched matrix multiplication.
    Separately compute each vector's L2 norm along its feature axis. Clamp norms
    to eps so a zero vector cannot cause division by zero, then add singleton
    item axes so the two norm tensors broadcast across every pair. A pair with
    a zero vector consequently has dot product zero and similarity zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    x_norm = torch.linalg.vector_norm(x, dim=-1).clamp_min(eps)
    y_norm = torch.linalg.vector_norm(y, dim=-1).clamp_min(eps)
    dot_products = torch.matmul(x, y.transpose(-1, -2))
    return dot_products / (x_norm.unsqueeze(-1) * y_norm.unsqueeze(-2))


def weighted_pool(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Use batched matrix multiplication to combine values with query weights.

    ``values`` has shape ``(B, S, D)`` and ``weights`` has shape ``(B, Q, S)``.
    The shared S axis is the weighted source dimension; multiplying weights by
    values sums over it and returns ``(B, Q, D)``. The caller owns any weight
    normalization, so the weights are used without modification.
    """
    return torch.matmul(weights, values)


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1,
) -> torch.Tensor:
    """Normalize logits along dim while excluding masked entries.

    Expand the mask to the logits shape using broadcasting. Replace excluded
    logits by negative infinity so they receive no probability in softmax.
    If an entire slice is excluded, softmax would otherwise yield NaNs; replace
    those NaNs with zeros to satisfy the all-masked-row contract.
    """
    expanded_mask = torch.broadcast_to(mask, logits.shape)
    masked_logits = logits.masked_fill(~expanded_mask, float("-inf"))
    probabilities = torch.softmax(masked_logits, dim=dim)
    return torch.nan_to_num(probabilities, nan=0.0)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Combine the head and per-head feature axes while keeping batch and time.

    Read B, T, H, and D from the input shape, then reshape the final two axes
    into a single H*D dimension. ``reshape`` also supports non-contiguous inputs
    when a view cannot be formed directly.
    """
    batch_size, time_steps, n_heads, head_dim = x.shape
    return x.reshape(batch_size, time_steps, n_heads * head_dim)


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Split combined features and put the head axis before the time axis.

    Check that the combined feature width is divisible by n_heads. Infer each
    head's width, reshape ``(B, T, H*D)`` to ``(B, T, H, D)``, then permute axes
    to the common attention layout ``(B, H, T, D)``.
    """
    batch_size, time_steps, combined_dim = x.shape
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if combined_dim % n_heads != 0:
        raise ValueError("feature dimension must be divisible by n_heads")
    head_dim = combined_dim // n_heads
    split = x.reshape(batch_size, time_steps, n_heads, head_dim)
    return split.permute(0, 2, 1, 3)


def gather_last_valid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Select the final real timestep from every padded sequence.

    Convert each sequence length to its last zero-based position by subtracting
    one. Build one batch index per row and use paired advanced indices to select
    a single ``(D,)`` feature vector from each ``(T, D)`` sequence.
    """
    batch_indices = torch.arange(x.shape[0], device=x.device)
    time_indices = lengths - 1
    return x[batch_indices, time_indices]


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average valid timesteps independently for every sequence and feature.

    Add a trailing singleton axis to the ``(B, T)`` mask so it broadcasts over
    the D features. Sum masked values and valid-position counts along time.
    Clamp counts to at least one before division: nonempty rows get their true
    mean, while empty rows stay zero instead of producing NaNs.
    """
    expanded_mask = mask.unsqueeze(-1)
    sums = (x * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp_min(1)
    return sums / counts


class ModelSolution(TensorManipulationExercise):
    """Bind the walkthrough implementations to the shared exercise test suite."""

    pairwise_squared_distances = staticmethod(pairwise_squared_distances)
    pairwise_cosine_similarity = staticmethod(pairwise_cosine_similarity)
    weighted_pool = staticmethod(weighted_pool)
    masked_softmax = staticmethod(masked_softmax)
    merge_heads = staticmethod(merge_heads)
    split_heads = staticmethod(split_heads)
    gather_last_valid = staticmethod(gather_last_valid)
    masked_mean_pool = staticmethod(masked_mean_pool)
