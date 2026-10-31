"""Exercise 04 contract and tests. Reference code is loaded only on request."""
import unittest
from abc import abstractmethod

import torch

from exercise import Exercise


class TensorManipulationExercise(Exercise):
    title = "Advanced tensor dimension operations"
    problem = """Implement eleven vectorized PyTorch functions in
tensor_manipulation\\solve_this.py. Focus on broadcasting, reshaping,
permuting, reductions, and batched matrix operations; avoid Python loops.

1. Extract sliding windows: x (B, T, D) -> (B, W, window_size, D),
   where W = 1 + (T - window_size) // step. No padding.
2. Pairwise cosine similarities: x (B, N, D), y (B, M, D) -> (B, N, M).
   Zero vectors should
   produce similarity 0; eps is a positive lower bound for norm denominators.
3. Weighted batch pooling: values (B, S, D), weights (B, Q, S) -> (B, Q, D).
   Weights are used as supplied and are not normalized by the function.
4. Masked softmax over a selected dimension. The boolean mask must be
   broadcastable to logits; masked entries are zero, and fully masked rows
   produce all zeros rather than NaNs.
5. Merge the head and per-head feature axes: (B, T, H, D) -> (B, T, H*D).
6. Split a combined feature axis into heads and move heads before time:
   (B, T, H*D) -> (B, H, T, D). The feature width must divide by n_heads.
7. Gather the last valid timestep from each sequence: x (B, T, D), lengths
   (B,) -> (B, D), where each length is between 1 and T.
8. Patchify images: x (B, C, H, W) -> (B, P, C*patch_size*patch_size).
   Use non-overlapping square patches in row-major spatial order; flatten each
   patch in channel, local-row, local-column order. H and W must divide evenly.
9. Select features for the k highest scores per batch item:
   x (B, N, D), scores (B, N) -> (B, k, D), in descending score order.
10. Shuffle image channels by grouping, swapping group and within-group axes,
    and merging back: (B, C, H, W) -> same shape.
11. Extract diagonals from matrices with any number of leading batch axes.
    x (..., R, C) -> (..., diagonal_length), including off-diagonal offsets.

Preserve input dtype/device and batch membership; do not mutate inputs.
Support non-contiguous tensors. Positive window/step/patch/group parameters
and valid top-k counts are required; invalid values raise ValueError.
Suggested time box: 45-60 minutes."""

    @staticmethod
    @abstractmethod
    def sliding_windows(
        x: torch.Tensor, window_size: int, step: int = 1,
    ) -> torch.Tensor:
        """Extract unpadded (B, windows, window_size, D) sequence windows."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pairwise_cosine_similarity(
        x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return batched cosine similarities with shape (B, N, M)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def weighted_pool(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Combine (B, S, D) values with (B, Q, S) weights into (B, Q, D)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def masked_softmax(
        logits: torch.Tensor, mask: torch.Tensor, dim: int = -1,
    ) -> torch.Tensor:
        """Softmax along dim with broadcastable masking and zero output for empty rows."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def merge_heads(x: torch.Tensor) -> torch.Tensor:
        """Merge the last two axes of (B, T, H, D) into (B, T, H*D)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Reshape (B, T, H*D) into (B, H, T, D)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gather_last_valid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Gather each sequence's final valid timestep into a (B, D) tensor."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Turn images into flattened, row-major, non-overlapping square patches."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def topk_features(x: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Gather feature vectors for each batch item's k highest-scoring entries."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        """Interleave groups of image channels without changing spatial positions."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def batched_diagonal(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Extract a diagonal over the last two axes, preserving all leading axes."""
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "TensorManipulationExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Exercise04Tests(unittest.TestCase):
            def test_sliding_windows_values_order_and_step(self):
                x = torch.arange(20).reshape(2, 5, 2)
                for step in (1, 2):
                    with self.subTest(step=step):
                        expected = torch.stack(
                            [x[:, start:start + 3] for start in range(0, 3, step)], dim=1,
                        )
                        torch.testing.assert_close(solution.sliding_windows(x, 3, step), expected)

            def test_sliding_windows_noncontiguous_and_full_length(self):
                x = torch.arange(30).reshape(2, 3, 5).transpose(1, 2)
                torch.testing.assert_close(solution.sliding_windows(x, 5), x.unsqueeze(1))

            def test_sliding_windows_invalid_sizes(self):
                for size, step in ((0, 1), (6, 1), (2, 0), (2, -1)):
                    with self.subTest(size=size, step=step), self.assertRaises(ValueError):
                        solution.sliding_windows(torch.zeros(2, 5, 3), size, step)

            def test_cosine_similarity_preserves_batch_dimension(self):
                x = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
                y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 2)
                expected_cosine = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
                cosine = solution.pairwise_cosine_similarity(x, y)
                torch.testing.assert_close(cosine, expected_cosine)

            def test_cosine_similarity_zero_vectors(self):
                x = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
                y = torch.tensor([[[1.0, 0.0]]])
                expected = torch.tensor([[[0.0], [2.0 ** -0.5]]])
                torch.testing.assert_close(solution.pairwise_cosine_similarity(x, y), expected)

            def test_cosine_similarity_requires_positive_eps(self):
                with self.assertRaises(ValueError):
                    solution.pairwise_cosine_similarity(
                        torch.ones(1, 1, 2), torch.ones(1, 1, 2), eps=0,
                    )

            def test_weighted_pool(self):
                values = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
                weights = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]])
                expected = torch.tensor([[[1.0, 10.0], [2.5, 25.0]]])
                torch.testing.assert_close(solution.weighted_pool(values, weights), expected)

            def test_masked_softmax_broadcasts_mask_and_zeros_masked_values(self):
                logits = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]])
                mask = torch.tensor([[[True, False, True]]])
                actual = solution.masked_softmax(logits, mask)
                expected = torch.tensor([[
                    [0.1192029, 0.0, 0.8807971],
                    [0.7310586, 0.0, 0.2689414],
                ]])
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1, 2))

            def test_masked_softmax_all_masked_row(self):
                logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
                mask = torch.tensor([[True, False], [False, False]])
                expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
                torch.testing.assert_close(solution.masked_softmax(logits, mask), expected)

            def test_merge_and_split_heads_round_trip(self):
                x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
                merged = solution.merge_heads(x)
                self.assertEqual(tuple(merged.shape), (2, 3, 4))
                split = solution.split_heads(merged, n_heads=2)
                torch.testing.assert_close(split, x.permute(0, 2, 1, 3))

            def test_merge_heads_handles_noncontiguous_input(self):
                x = torch.randn(2, 3, 4, 5).transpose(1, 2)
                actual = solution.merge_heads(x)
                expected = x.reshape(2, 4, 15)
                torch.testing.assert_close(actual, expected)

            def test_split_heads_rejects_nondivisible_feature_width(self):
                with self.assertRaises(ValueError):
                    solution.split_heads(torch.randn(2, 3, 10), n_heads=3)

            def test_split_heads_requires_positive_head_count(self):
                with self.assertRaises(ValueError):
                    solution.split_heads(torch.randn(2, 3, 10), n_heads=0)

            def test_gather_last_valid_timestep(self):
                x = torch.tensor([
                    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                    [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
                ])
                lengths = torch.tensor([1, 3])
                expected = torch.tensor([[1.0, 10.0], [6.0, 60.0]])
                torch.testing.assert_close(solution.gather_last_valid(x, lengths), expected)

            def test_patchify_patch_and_feature_order(self):
                x = torch.arange(2 * 2 * 4 * 6).reshape(2, 2, 4, 6)
                expected = torch.stack([
                    x[:, :, row:row + 2, col:col + 2].reshape(2, -1)
                    for row in (0, 2) for col in (0, 2, 4)
                ], dim=1)
                torch.testing.assert_close(solution.patchify(x, 2), expected)

            def test_patchify_noncontiguous_and_unit_patches(self):
                x = torch.arange(24).reshape(1, 2, 3, 4).transpose(-1, -2)
                expected = x.permute(0, 2, 3, 1).reshape(1, 12, 2)
                torch.testing.assert_close(solution.patchify(x, 1), expected)

            def test_patchify_invalid_size(self):
                for size in (0, -1, 3):
                    with self.subTest(size=size), self.assertRaises(ValueError):
                        solution.patchify(torch.zeros(1, 2, 4, 6), size)

            def test_topk_features_batch_order_and_repeated_values(self):
                x = torch.arange(24).reshape(2, 4, 3)
                scores = torch.tensor([[2., 4., 1., 3.], [4., 1., 3., 2.]])
                expected = torch.stack((x[0, [1, 3]], x[1, [0, 2]]))
                torch.testing.assert_close(solution.topk_features(x, scores, 2), expected)

            def test_topk_features_noncontiguous_and_gradients(self):
                base = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
                base.requires_grad_()
                x = base.transpose(1, 2)
                scores = torch.tensor([[2., 4., 1., 3.], [4., 1., 3., 2.]])
                actual = solution.topk_features(x, scores, 1)
                torch.testing.assert_close(actual, torch.stack((x[0, 1], x[1, 0])).unsqueeze(1))
                actual.sum().backward()
                expected_grad = torch.zeros_like(base)
                expected_grad[0, :, 1] = 1
                expected_grad[1, :, 0] = 1
                torch.testing.assert_close(base.grad, expected_grad)

            def test_topk_features_invalid_k(self):
                for k in (0, -1, 5):
                    with self.subTest(k=k), self.assertRaises(ValueError):
                        solution.topk_features(torch.zeros(2, 4, 3), torch.zeros(2, 4), k)

            def test_channel_shuffle_order(self):
                x = torch.arange(48).reshape(2, 6, 2, 2)
                torch.testing.assert_close(solution.channel_shuffle(x, 2), x[:, [0, 3, 1, 4, 2, 5]])
                torch.testing.assert_close(solution.channel_shuffle(x, 3), x[:, [0, 2, 4, 1, 3, 5]])

            def test_channel_shuffle_identity_and_noncontiguous(self):
                x = torch.arange(48).reshape(2, 6, 2, 2).transpose(-1, -2)
                for groups in (1, 6):
                    with self.subTest(groups=groups):
                        torch.testing.assert_close(solution.channel_shuffle(x, groups), x)

            def test_channel_shuffle_invalid_groups(self):
                for groups in (0, -1, 4):
                    with self.subTest(groups=groups), self.assertRaises(ValueError):
                        solution.channel_shuffle(torch.zeros(2, 6, 2, 2), groups)

            def test_batched_diagonal_offsets_and_leading_axes(self):
                x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
                for offset in (-2, 0, 2, 6):
                    with self.subTest(offset=offset):
                        rows = list(range(max(0, -offset), min(4, 5 - offset)))
                        expected = x[..., rows, [row + offset for row in rows]]
                        torch.testing.assert_close(solution.batched_diagonal(x, offset), expected)

            def test_batched_diagonal_noncontiguous_and_no_batch(self):
                x = torch.arange(12).reshape(3, 4).transpose(0, 1)
                torch.testing.assert_close(solution.batched_diagonal(x), torch.tensor([0, 5, 10]))

        Exercise04Tests.__name__ = f"Exercise04Tests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Exercise04Tests)
