"""Exercise 06 contract and tests; reference answers are imported lazily."""
import math
import unittest
from abc import abstractmethod

import torch
from torch import nn

from exercise import Exercise
from .support import LOSS_CASES, SuppliedRegressor, encode_sequences, regression_data


class GeneDebuggingExercise(Exercise):
    title = "Gene-sequence debugging and nn.Module mini-interviews"
    problem = """Work in gene_debugging\\solve_this.py. Documentation is allowed;
do not use AI assistance or open model_solution.py during a timed attempt.

Pick ONE loss case and ONE module per mock: 20 minutes debugging, then
15 minutes implementing. Move on at the time limit, even if part 1 is unfinished.

Part 1: DNA strings are encoded as float32 (batch, 4, length), A/C/G/T order.
The supplied linear model predicts GC fraction as (batch, 1), with raw output.
Run its training loop, inspect loss, predictions, shapes and gradients, explain
why a decreasing objective need not mean better predictions, and repair it.
Three alternative broken objectives use r = target - prediction:
  signed: mean(r); negative_square: -mean(r**2); cubic: mean(r**3).
The signed objective is affine (convex), but unbounded below. The other two
are non-convex and also unbounded below. Non-convexity alone is not the bug.
Repair each to mean squared error without abs/clamping/detaching predictions.
train_model must use fresh full-batch SGD, enter training mode, align target
(batch,) or (batch, 1) to predictions, clear gradients each epoch, and return
one pre-update Python float loss per epoch. No updates for zero epochs.

Part 2 options (layer names are unrestricted, parameters must be registered):
 A. GCRegressor: mean over length -> Linear(4, 1), no sigmoid.
 B. MotifClassifier: Conv1d(4, filters, kernel_size), valid padding -> ReLU ->
    max over length -> Linear(filters, 1), raw logits. Use biases.
 C. ResidualBlock: x + Linear(hidden_dim, features)(
    ReLU(Linear(features, hidden_dim)(x))), biases, no final activation.
    Accept 2D and 3D inputs; transform only the final feature dimension.

All modules preserve batch dimension for batch size 1 and support autograd.
DNA inputs are nonempty equal-length uppercase A/C/G/T strings; supplied
encoding rejects invalid inputs. Motif input length >= kernel_size.
Use README.md inside the folder for isolated case tests and Colab cells."""

    @staticmethod
    @abstractmethod
    def regression_loss(
        prediction: torch.Tensor, target: torch.Tensor, case: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train_model(
        model: nn.Module, x: torch.Tensor, target: torch.Tensor,
        case: str, epochs: int = 80, lr: float = 0.1,
    ) -> list[float]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_gc_regressor() -> nn.Module:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_motif_classifier(filters: int = 8, kernel_size: int = 3) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_residual_block(features: int, hidden_dim: int) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "GeneDebuggingExercise":
        from .model_solution import ModelSolution
        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Tests(unittest.TestCase):
            def setUp(self):
                self.random_state = torch.random.get_rng_state()
                torch.manual_seed(6)

            def tearDown(self):
                torch.random.set_rng_state(self.random_state)

            def check_loss(self, case):
                for values in ([1.0, -2.0, 4.0], [100.0, -100.0, 0.0]):
                    prediction = torch.tensor(values, requires_grad=True)
                    target = torch.tensor([0.0, 1.0, 2.0])
                    loss = solution.regression_loss(prediction, target, case)
                    self.assertEqual(loss.ndim, 0)
                    torch.testing.assert_close(loss, (prediction - target).square().mean())
                    loss.backward()
                    torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - target) / 3)
                target = torch.tensor([[0.2], [0.8]])
                torch.testing.assert_close(
                    solution.regression_loss(target, target, case), torch.tensor(0.0),
                )

            def test_loss_signed(self):
                self.check_loss("signed")

            def test_loss_negative_square(self):
                self.check_loss("negative_square")

            def test_loss_cubic(self):
                self.check_loss("cubic")

            def test_loss_unknown_case(self):
                with self.assertRaises(ValueError):
                    solution.regression_loss(torch.zeros(1), torch.zeros(1), "unknown")

            def test_encoding_strings_and_validation(self):
                expected = torch.eye(4).unsqueeze(0)
                torch.testing.assert_close(encode_sequences(["ACGT"]), expected)
                for sequences in ([], [""], ["AX"], ["AC", "A"]):
                    with self.subTest(sequences=sequences), self.assertRaises(ValueError):
                        encode_sequences(sequences)

            def test_training_matches_two_sgd_steps(self):
                x, target = regression_data()
                for case in LOSS_CASES:
                    for column in (False, True):
                        with self.subTest(case=case, column=column):
                            model = SuppliedRegressor()
                            expected = SuppliedRegressor()
                            expected.load_state_dict(model.state_dict())
                            expected_losses = []
                            optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
                            for _ in range(2):
                                optimizer.zero_grad()
                                loss = (expected(x) - target[:, None]).square().mean()
                                expected_losses.append(loss.item())
                                loss.backward()
                                optimizer.step()
                            model.eval()
                            # Existing gradients must not contaminate the first step.
                            for parameter in model.parameters():
                                parameter.grad = torch.ones_like(parameter)
                            actual = solution.train_model(
                                model, x, target[:, None] if column else target,
                                case, epochs=2, lr=0.1,
                            )
                            self.assertTrue(model.training)
                            self.assertEqual(len(actual), 2)
                            self.assertTrue(all(isinstance(value, float) for value in actual))
                            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected_losses))
                            for got, want in zip(model.parameters(), expected.parameters()):
                                torch.testing.assert_close(got, want)

            def test_training_learns_gc_and_zero_epochs_does_not_update(self):
                x, target = regression_data()
                for case in LOSS_CASES:
                    with self.subTest(case=case):
                        model = SuppliedRegressor()
                        before = [p.detach().clone() for p in model.parameters()]
                        self.assertEqual(solution.train_model(model, x, target, case, epochs=0), [])
                        for got, want in zip(model.parameters(), before):
                            torch.testing.assert_close(got, want)
                        losses = solution.train_model(model, x, target, case, epochs=200)
                        self.assertEqual(len(losses), 200)
                        self.assertTrue(all(math.isfinite(value) for value in losses))
                        self.assertLess(losses[-1], losses[0])
                        with torch.no_grad():
                            mse = (model(x).flatten() - target).square().mean().item()
                        self.assertLess(mse, 0.005)

            def check_gradients(self, model, x):
                model(x).sum().backward()
                parameters = list(model.parameters())
                self.assertTrue(parameters)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

            def test_module_gc_values_shapes_and_gradients(self):
                model = solution.create_gc_regressor()
                self.assertIsInstance(model, nn.Module)
                layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
                self.assertEqual(len(layers), 1)
                head = layers[0]
                self.assertEqual((head.in_features, head.out_features), (4, 1))
                with torch.no_grad():
                    head.weight.copy_(torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
                    head.bias.fill_(-0.25)
                for sequences in (["ACGT"], ["CCCC", "AAAA"], ["G"]):
                    x = encode_sequences(sequences)
                    torch.testing.assert_close(model(x), x[:, 1:3].sum(1).mean(1, keepdim=True) - 0.25)
                self.check_gradients(model, encode_sequences(["ACGT"]))

            def test_module_motif_matches_spec_and_gradients(self):
                model = solution.create_motif_classifier(filters=2, kernel_size=3)
                self.assertIsInstance(model, nn.Module)
                convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv1d)]
                heads = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
                self.assertEqual(len(convolutions), 1)
                self.assertEqual(len(heads), 1)
                scanner, head = convolutions[0], heads[0]
                self.assertEqual((scanner.in_channels, scanner.out_channels, scanner.kernel_size), (4, 2, (3,)))
                self.assertEqual((head.in_features, head.out_features), (2, 1))
                self.assertIsNotNone(scanner.bias)
                self.assertIsNotNone(head.bias)
                with torch.no_grad():
                    scanner.weight.copy_(torch.arange(24).reshape(2, 4, 3) / 10 - 1)
                    scanner.bias.fill_(-0.3)
                    head.weight.copy_(torch.tensor([[0.7, -0.5]]))
                    head.bias.fill_(-2.0)
                for sequences in (["ACG"], ["AAAAACGT", "TTTTTTTT"]):
                    x = encode_sequences(sequences)
                    conv = torch.nn.functional.conv1d(x, scanner.weight, scanner.bias)
                    expected = torch.nn.functional.linear(torch.relu(conv).amax(-1), head.weight, head.bias)
                    self.assertEqual(tuple(model(x).shape), (len(sequences), 1))
                    torch.testing.assert_close(model(x), expected)
                self.check_gradients(model, x)

            def test_module_residual_formula_shapes_and_gradients(self):
                model = solution.create_residual_block(features=3, hidden_dim=5)
                self.assertIsInstance(model, nn.Module)
                layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
                self.assertEqual(len(layers), 2)
                expand = next(layer for layer in layers if layer.in_features == 3)
                project = next(layer for layer in layers if layer.in_features == 5)
                self.assertEqual((expand.out_features, project.out_features), (5, 3))
                with torch.no_grad():
                    expand.weight.fill_(0.2)
                    expand.bias.fill_(0.4)
                    project.weight.fill_(-0.3)
                    project.bias.fill_(-0.5)
                for shape in ((1, 3), (4, 3), (2, 4, 3)):
                    x = torch.linspace(-2, 2, math.prod(shape)).reshape(shape)
                    expected = x + torch.nn.functional.linear(
                        torch.relu(torch.nn.functional.linear(x, expand.weight, expand.bias)),
                        project.weight, project.bias,
                    )
                    torch.testing.assert_close(model(x), expected)
                self.check_gradients(model, torch.ones(2, 3))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                x = torch.tensor([[-2.0, 0.0, 1.0]])
                torch.testing.assert_close(model(x), x)

        Tests.__name__ = f"GeneDebuggingTests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
