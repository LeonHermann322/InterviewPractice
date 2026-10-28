"""Exercise 01 contract and tests. Reference code is loaded only on request."""
import math
import unittest
from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from torch import nn

from exercise import Exercise


class DNAExercise(Exercise):
    title = "DNA MLP classifier"
    problem = """Implement a binary DNA classifier in PyTorch.

Encode uppercase A/C/G/T sequences as flattened float32 one-hot tensors.
Batch sequences of equal length and reject unknown bases or unequal lengths.
Build Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1),
returning raw logits of shape (N, 1). Train with BCEWithLogitsLoss and SGD,
accepting labels of shape (N,) or (N, 1), and return one scalar loss per epoch.
Predict binary labels of shape (N, 1) without building a gradient graph.
Layer attribute names are up to you; nn.Sequential is also supported.
Unknown bases must raise ValueError or KeyError; unequal sequence lengths
must raise ValueError.

Implement the functions and DNAClassifier in dna_classifier\\solve_this.py.
Suggested time box: 20 minutes."""
    sequences: ClassVar[tuple[str, ...]] = (
        "AAAAAC", "CCCCGT", "TTTTGA", "GCGCCA", "AAAATA", "CGCGCG",
    )
    labels: ClassVar[tuple[int, ...]] = (1, 0, 1, 0, 1, 0)

    @staticmethod
    @abstractmethod
    def encode_sequence(seq: str) -> torch.Tensor:
        """Return float32 one-hot DNA of shape (len(seq) * 4,)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def encode_sequences(sequences: Sequence[str]) -> torch.Tensor:
        """Return a float32 batch of equal-length encoded sequences."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_model(input_dim: int, hidden_dim: int = 16) -> nn.Module:
        """Construct the MLP described in the problem."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train_model(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.1,
    ) -> list[float]:
        """Train with BCEWithLogitsLoss and SGD; return per-epoch losses."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict_labels(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        """Return binary labels of shape (N, 1), without gradients."""
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "DNAExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class ExerciseTests(unittest.TestCase):
            def test_encode_sequence_shape_and_values(self):
                got = solution.encode_sequence("ACGT")
                expected = torch.tensor([
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ], dtype=torch.float32)
                self.assertEqual(tuple(got.shape), (16,))
                self.assertTrue(torch.equal(got, expected))
                self.assertEqual(got.dtype, torch.float32)

            def test_encode_sequence_rejects_unknown_base(self):
                with self.assertRaises((ValueError, KeyError)):
                    solution.encode_sequence("ACNX")

            def test_encode_sequences_batch(self):
                got = solution.encode_sequences(["AC", "GT"])
                self.assertEqual(tuple(got.shape), (2, 8))
                self.assertTrue(torch.equal(got[0], solution.encode_sequence("AC")))
                self.assertTrue(torch.equal(got[1], solution.encode_sequence("GT")))

            def test_encode_sequences_rejects_unequal_lengths(self):
                with self.assertRaises(ValueError):
                    solution.encode_sequences(["AC", "ACG"])

            def test_model_output_shape_and_finiteness(self):
                model = solution.create_model(input_dim=24, hidden_dim=16)
                for batch_size in (1, 8):
                    output = model(torch.randn(batch_size, 24))
                    self.assertEqual(tuple(output.shape), (batch_size, 1))
                    self.assertTrue(torch.isfinite(output).all().item())

            def test_model_returns_logits_not_probabilities_by_contract(self):
                model = solution.create_model(input_dim=24, hidden_dim=16)
                model.eval()
                linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
                self.assertTrue(linear_layers, "The specified MLP must contain Linear layers")
                # Constant affine outputs expose probability squashing without
                # depending on layer names, nesting, or module registration order.
                for logit in (-2.0, 2.0):
                    with self.subTest(logit=logit), torch.no_grad():
                        for layer in linear_layers:
                            layer.weight.zero_()
                            if layer.bias is not None:
                                layer.bias.fill_(logit)
                        output = model(torch.zeros(3, 24))
                        torch.testing.assert_close(output, torch.full((3, 1), logit))

            def test_forward_pass_supports_gradients(self):
                model = solution.create_model(input_dim=24, hidden_dim=16)
                x = torch.randn(4, 24)
                y = torch.tensor([[1.], [0.], [1.], [0.]])
                loss = nn.BCEWithLogitsLoss()(model(x), y)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(grads)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

            def test_training_returns_finite_losses_and_updates_model(self):
                torch.manual_seed(7)
                X = solution.encode_sequences(solution.sequences)
                y = torch.tensor(solution.labels, dtype=torch.float32)
                model = solution.create_model(input_dim=X.shape[1], hidden_dim=16)
                before = [p.detach().clone() for p in model.parameters()]
                losses = solution.train_model(model, X, y, epochs=20, lr=0.1)

                self.assertEqual(len(losses), 20)
                self.assertTrue(all(math.isfinite(v) for v in losses))
                after = list(model.parameters())
                self.assertTrue(any(not torch.equal(a, b.detach()) for a, b in zip(before, after)))

            def test_training_reduces_loss_on_toy_data(self):
                torch.manual_seed(11)
                X = solution.encode_sequences(solution.sequences)
                y = torch.tensor(solution.labels, dtype=torch.float32)
                model = solution.create_model(input_dim=X.shape[1], hidden_dim=16)
                losses = solution.train_model(model, X, y, epochs=150, lr=0.1)
                self.assertLess(losses[-1], losses[0])

            def test_predict_labels_shape_and_values(self):
                X = solution.encode_sequences(solution.sequences)
                model = solution.create_model(input_dim=X.shape[1], hidden_dim=16)
                predictions = solution.predict_labels(model, X)
                self.assertEqual(tuple(predictions.shape), (len(solution.sequences), 1))
                self.assertTrue(set(predictions.unique().tolist()).issubset({0, 1}))

            def test_predict_labels_does_not_build_grad_graph(self):
                X = solution.encode_sequences(solution.sequences)
                model = solution.create_model(input_dim=X.shape[1], hidden_dim=16)
                predictions = solution.predict_labels(model, X)
                self.assertFalse(predictions.requires_grad)

        ExerciseTests.__name__ = f"ExerciseTests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(ExerciseTests)
