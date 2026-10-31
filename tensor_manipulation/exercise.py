"""Exercise 04 contract and tests. Reference code is loaded only on request."""
import unittest
from abc import abstractmethod

import torch

from exercise import Exercise


class TensorManipulationExercise(Exercise):
    title = "Advanced tensor dimension operations"
    problem = """Implement eight vectorized PyTorch functions in
tensor_manipulation\\solve_this.py. Focus on broadcasting, reshaping,
permuting, reductions, and batched matrix operations; avoid Python loops.

1. Pairwise squared distances: x (B, N, D), y (B, M, D) -> (B, N, M).
2. Pairwise cosine similarities for the same shapes. Zero vectors should
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
8. Compute a masked mean over time: x (B, T, D), mask (B, T) -> (B, D).
   An item with no valid timesteps must produce a zero vector.

Preserve batch membership and return the specified shapes. Suggested time box:
30-40 minutes."""

    @staticmethod
    @abstractmethod
    def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return batched squared distances with shape (B, N, M)."""
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
    def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return a time-masked mean of (B, T, D), using zeros for empty masks."""
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "TensorManipulationExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Exercise04Tests(unittest.TestCase):
            def test_pairwise_squared_distances(self):
                x = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
                y = torch.tensor([[[1.0, 0.0], [2.0, 2.0], [-1.0, 0.0]]])
                expected = torch.tensor([[[1.0, 8.0, 1.0], [1.0, 2.0, 5.0]]])
                torch.testing.assert_close(solution.pairwise_squared_distances(x, y), expected)

            def test_pairwise_operations_preserve_batch_dimension(self):
                x = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
                y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 2)
                expected_distances = torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]])
                expected_cosine = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
                distances = solution.pairwise_squared_distances(x, y)
                cosine = solution.pairwise_cosine_similarity(x, y)
                torch.testing.assert_close(distances, expected_distances)
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

            def test_masked_mean_pool_and_empty_sequence(self):
                x = torch.tensor([
                    [[1.0, 10.0], [3.0, 30.0], [100.0, 1000.0]],
                    [[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]],
                ])
                mask = torch.tensor([[True, True, False], [False, False, False]])
                expected = torch.tensor([[2.0, 20.0], [0.0, 0.0]])
                torch.testing.assert_close(solution.masked_mean_pool(x, mask), expected)

        Exercise04Tests.__name__ = f"Exercise04Tests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Exercise04Tests)
