"""Walkthrough reference solution for Exercise 04 tensor dimension operations."""
import torch

from .exercise import TensorManipulationExercise


def sliding_windows(
    x: torch.Tensor, window_size: int, step: int = 1,
) -> torch.Tensor:
    """Extract time windows using a strided tensor view rather than a Python loop.

    1. Check that a window fits and the step is positive.
    2. Unfold the time axis. PyTorch appends the within-window axis at the end,
       producing (B, W, D, window_size), not the requested layout.
    3. Move that final axis before features to obtain (B, W, window_size, D).
       The result can share storage with x; overlapping windows must not be
       modified in-place because they may refer to the same input entries.
    """
    if not 1 <= window_size <= x.shape[1] or step <= 0:
        raise ValueError("window_size must fit the time axis and step must be positive")
    return x.unfold(1, window_size, step).movedim(-1, -2)


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


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Separate spatial patch axes from local pixel axes, then flatten tokens.

    1. Validate the patch size and exact divisibility of both image dimensions.
    2. Split H and W into patch-grid and within-patch dimensions. The reshape
       gives (B, C, grid_rows, local_rows, grid_cols, local_cols).
    3. Permute to (B, grid_rows, grid_cols, C, local_rows, local_cols).
       This makes the patch-grid axes adjacent and puts each token's features
       in channel-first pixel order.
    4. Flatten the grid into P and the remaining three axes into token features.
       Reshape handles non-contiguous tensors, copying only when needed.
    """
    batch, channels, height, width = x.shape
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError("patch_size must be positive and divide both spatial dimensions")
    rows, cols = height // patch_size, width // patch_size
    patches = x.reshape(batch, channels, rows, patch_size, cols, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch, rows * cols, channels * patch_size * patch_size)


def topk_features(x: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """Rank items, then gather their full feature vectors without copying indices.

    1. Validate k and obtain the descending top-k item indices, shaped (B, k).
    2. Add a singleton feature axis and expand to (B, k, D). Expansion reuses
       the same item index for every feature without allocating repeated data.
    3. Gather along x's item axis. Each output vector belongs to the ranked item
       in that same batch row. Autograd sends gradients to selected x entries;
       the discrete ranking itself is not differentiable with respect to scores.
    """
    if not 1 <= k <= x.shape[1]:
        raise ValueError("k must be between 1 and the number of items")
    indices = scores.topk(k, dim=1, sorted=True).indices
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[2]))


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Interleave channel groups by changing axis order, not pixel values.

    1. Validate the group count and divide C into groups and channels per group.
    2. Reshape to (B, groups, channels_per_group, H, W).
    3. Swap the two channel-related axes: the first channel of every group now
       comes before the second channel of every group, and so on.
    4. Merge the swapped axes back into C. Spatial axes remain untouched.
       Reshape can materialize the reordered layout after the transpose.
    """
    batch, channels, height, width = x.shape
    if groups <= 0 or channels % groups:
        raise ValueError("groups must be positive and divide the channel count")
    grouped = x.reshape(batch, groups, channels // groups, height, width)
    return grouped.transpose(1, 2).reshape(batch, channels, height, width)


def batched_diagonal(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Extract matrix diagonals without flattening or iterating through batch axes.

    Specify the last two axes as the matrix axes so all earlier dimensions
    survive unchanged. PyTorch's diagonal operation appends one axis containing
    the selected entries, with the requested offset and rectangular-matrix
    bounds handled automatically. It returns a view, even for strided inputs.
    """
    return x.diagonal(offset=offset, dim1=-2, dim2=-1)


class ModelSolution(TensorManipulationExercise):
    """Bind the walkthrough implementations to the shared exercise test suite."""

    sliding_windows = staticmethod(sliding_windows)
    pairwise_cosine_similarity = staticmethod(pairwise_cosine_similarity)
    weighted_pool = staticmethod(weighted_pool)
    masked_softmax = staticmethod(masked_softmax)
    merge_heads = staticmethod(merge_heads)
    split_heads = staticmethod(split_heads)
    gather_last_valid = staticmethod(gather_last_valid)
    patchify = staticmethod(patchify)
    topk_features = staticmethod(topk_features)
    channel_shuffle = staticmethod(channel_shuffle)
    batched_diagonal = staticmethod(batched_diagonal)
