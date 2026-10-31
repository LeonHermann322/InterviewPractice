"""Exercise 03 contract and tests. Reference code is loaded only on request."""
import unittest
from abc import abstractmethod
from collections.abc import Sequence

import torch

from exercise import Exercise


class PyTorchDrillsExercise(Exercise):
    title = "PyTorch tensor and vectorization drills"
    problem = """Implement nine PyTorch functions in pytorch_drills\\solve_this.py.

1. Add a feature bias to every row using broadcasting.
2. Return the mean of scores selected by a boolean mask.
3. Compute each row's z-score using population standard deviation.
4. Compute masked mean-squared error over True elements.
5. Return pairwise squared distances for two sets of feature vectors.
6. Build a boolean mask where every row can attend to context columns only.
7. Build a boolean causal mask where row i can attend through column i.
8. Mean-aggregate source node features at each destination using edge_index.
   Nodes without incoming edges must remain zero.
9. Zero-pad a list of variable-length graph tensors and return a real-node mask.

Do not use Python loops for drills 1-8. For drill 9, a loop over graphs is
allowed, but loops over nodes or features are not. Match the specified shapes
and preserve tensor values and dtypes.
Suggested time box: 20 minutes."""

    @staticmethod
    @abstractmethod
    def add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Add a feature vector to every row of x."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def masked_mean(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the mean of scores where mask is True."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def row_zscore(x: torch.Tensor) -> torch.Tensor:
        """Normalize each row using its population mean and standard deviation."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def masked_mse(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean squared error over True mask elements."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the squared Euclidean distances with shape (len(a), len(b))."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def context_mask(n_context: int, n_query: int) -> torch.Tensor:
        """Return bool (total, total) mask allowing all rows to see context only."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def causal_mask(n: int) -> torch.Tensor:
        """Return bool (n, n) lower-triangular mask, including the diagonal."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean-aggregate source features at destinations; isolated nodes stay zero."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pad_graphs(
        graphs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-padded (batch, max_nodes, features) data and a bool node mask."""
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "PyTorchDrillsExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Exercise03Tests(unittest.TestCase):
            def test_add_bias_broadcasts_across_rows(self):
                x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                bias = torch.tensor([10.0, 20.0])
                expected = torch.tensor([[11.0, 22.0], [13.0, 24.0]])
                torch.testing.assert_close(solution.add_bias(x, bias), expected)

            def test_masked_mean(self):
                scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
                mask = torch.tensor([True, False, True, False])
                torch.testing.assert_close(solution.masked_mean(scores, mask), torch.tensor(2.0))

            def test_row_zscore_uses_population_standard_deviation(self):
                x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
                expected = torch.tensor(
                    [[-1.2247449, 0.0, 1.2247449], [float("nan")] * 3]
                )
                # Constant rows produce NaNs for the same mathematical reason
                # as the reference operation: their standard deviation is zero.
                actual = solution.row_zscore(x)
                torch.testing.assert_close(actual[0], expected[0])
                self.assertTrue(torch.isnan(actual[1]).all())

            def test_masked_mse_ignores_false_elements(self):
                pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                target = torch.zeros_like(pred)
                mask = torch.tensor([[True, False], [True, False]])
                torch.testing.assert_close(solution.masked_mse(pred, target, mask), torch.tensor(5.0))

            def test_pairwise_squared_distances(self):
                a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
                b = torch.tensor([[1.0, 0.0], [2.0, 2.0], [-1.0, 0.0]])
                expected = torch.tensor([[1.0, 8.0, 1.0], [1.0, 2.0, 5.0]])
                torch.testing.assert_close(solution.pairwise_sq(a, b), expected)

            def test_context_mask(self):
                expected = torch.tensor(
                    [[1, 1, 0, 0, 0]] * 5, dtype=torch.bool,
                )
                torch.testing.assert_close(solution.context_mask(2, 3), expected)

            def test_causal_mask(self):
                expected = torch.tensor(
                    [[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool,
                )
                torch.testing.assert_close(solution.causal_mask(3), expected)

            def test_mean_aggregate_and_isolated_nodes(self):
                x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
                edge_index = torch.tensor([[0, 1, 2, 0], [1, 1, 3, 3]])
                expected = torch.tensor([[0.0, 0.0], [2.0, 3.0], [0.0, 0.0], [3.0, 4.0]])
                torch.testing.assert_close(solution.mean_aggregate(x, edge_index), expected)

            def test_pad_graphs_values_and_mask(self):
                graphs = [
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.tensor([[5.0, 6.0]]),
                ]
                padded, mask = solution.pad_graphs(graphs)
                expected_padded = torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [0.0, 0.0]]]
                )
                expected_mask = torch.tensor([[True, True], [True, False]])
                torch.testing.assert_close(padded, expected_padded)
                torch.testing.assert_close(mask, expected_mask)

        Exercise03Tests.__name__ = f"Exercise03Tests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Exercise03Tests)
