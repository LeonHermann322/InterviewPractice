"""Run tests against either the candidate implementation or reference solution.

Examples:
    python test.py --solution solve_this
    python test.py --solution model_solution
    python test.py --compare

The --compare option runs the same test suite against both modules separately.
It checks correctness, not that independently initialized models have identical
weights or predictions.
"""
import argparse
import importlib
import math
import sys
import unittest

import torch
from torch import nn


SEQUENCES = ["AAAAAC", "CCCCGT", "TTTTGA", "GCGCCA", "AAAATA", "CGCGCG"]
LABELS = [1, 0, 1, 0, 1, 0]


def build_suite(module_name: str):
    module = importlib.import_module(module_name)

    class ExerciseTests(unittest.TestCase):
        def test_encode_sequence_shape_and_values(self):
            got = module.encode_sequence("ACGT")
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
                module.encode_sequence("ACNX")

        def test_encode_sequences_batch(self):
            got = module.encode_sequences(["AC", "GT"])
            self.assertEqual(tuple(got.shape), (2, 8))
            self.assertTrue(torch.equal(got[0], module.encode_sequence("AC")))
            self.assertTrue(torch.equal(got[1], module.encode_sequence("GT")))

        def test_encode_sequences_rejects_unequal_lengths(self):
            with self.assertRaises(ValueError):
                module.encode_sequences(["AC", "ACG"])

        def test_model_output_shape_and_finiteness(self):
            model = module.DNAClassifier(input_dim=24, hidden_dim=16)
            for batch_size in (1, 8):
                output = model(torch.randn(batch_size, 24))
                self.assertEqual(tuple(output.shape), (batch_size, 1))
                self.assertTrue(torch.isfinite(output).all().item())

        def test_model_returns_logits_not_probabilities_by_contract(self):
            # Structural test: the final layer should be a linear layer and
            # forward should not constrain outputs to [0, 1].
            model = module.DNAClassifier(input_dim=24, hidden_dim=16)
            self.assertIsInstance(model.fc2, nn.Linear)
            self.assertEqual(model.fc2.out_features, 1)

        def test_forward_pass_supports_gradients(self):
            model = module.DNAClassifier(input_dim=24, hidden_dim=16)
            x = torch.randn(4, 24)
            y = torch.tensor([[1.], [0.], [1.], [0.]])
            loss = nn.BCEWithLogitsLoss()(model(x), y)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            self.assertTrue(grads)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

        def test_training_returns_finite_losses_and_updates_model(self):
            torch.manual_seed(7)
            X = module.encode_sequences(SEQUENCES)
            y = torch.tensor(LABELS, dtype=torch.float32)
            model = module.DNAClassifier(input_dim=X.shape[1], hidden_dim=16)
            before = [p.detach().clone() for p in model.parameters()]
            losses = module.train_model(model, X, y, epochs=20, lr=0.1)

            self.assertEqual(len(losses), 20)
            self.assertTrue(all(math.isfinite(v) for v in losses))
            after = list(model.parameters())
            self.assertTrue(any(not torch.equal(a, b.detach()) for a, b in zip(before, after)))

        def test_training_reduces_loss_on_toy_data(self):
            torch.manual_seed(11)
            X = module.encode_sequences(SEQUENCES)
            y = torch.tensor(LABELS, dtype=torch.float32)
            model = module.DNAClassifier(input_dim=X.shape[1], hidden_dim=16)
            losses = module.train_model(model, X, y, epochs=150, lr=0.1)
            self.assertLess(losses[-1], losses[0])

        def test_predict_labels_shape_and_values(self):
            X = module.encode_sequences(SEQUENCES)
            model = module.DNAClassifier(input_dim=X.shape[1], hidden_dim=16)
            preds = module.predict_labels(model, X)
            self.assertEqual(tuple(preds.shape), (len(SEQUENCES), 1))
            self.assertTrue(set(preds.unique().tolist()).issubset({0, 1}))

        def test_predict_labels_does_not_build_grad_graph(self):
            X = module.encode_sequences(SEQUENCES)
            model = module.DNAClassifier(input_dim=X.shape[1], hidden_dim=16)
            preds = module.predict_labels(model, X)
            self.assertFalse(preds.requires_grad)

    ExerciseTests.__name__ = f"ExerciseTests_{module_name}"
    return unittest.defaultTestLoader.loadTestsFromTestCase(ExerciseTests)


def run_for(module_name: str) -> bool:
    print(f"\n{'=' * 18} Testing {module_name} {'=' * 18}")
    suite = build_suite(module_name)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="PyTorch interview exercise tests")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--solution", choices=["solve_this", "model_solution"],
                       default="solve_this", help="Module to test")
    group.add_argument("--compare", action="store_true",
                       help="Run the test suite against candidate and reference")
    args = parser.parse_args()

    if args.compare:
        candidate_ok = run_for("solve_this")
        reference_ok = run_for("model_solution")
        print("\nSummary:")
        print(f"  solve_this.py:      {'PASS' if candidate_ok else 'FAIL'}")
        print(f"  model_solution.py:  {'PASS' if reference_ok else 'FAIL'}")
        sys.exit(0 if candidate_ok and reference_ok else 1)
    else:
        ok = run_for(args.solution)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
