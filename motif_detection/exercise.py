"""Exercise 02 contract and tests. Reference code is loaded only on request."""
import random
import unittest
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from exercise import Exercise


class MotifExercise(Exercise):
    title = "CNN motif detection"
    problem = """Detect a planted DNA motif in variable-length sequences.

Implement case-insensitive reverse complement and channels-first one-hot
encoding of shape (4, L). Encode N as 0.25 in every channel and reject other
unknown characters. Zero-pad batches and return a boolean mask of real bases.
Create a Dataset yielding (sequence, float_label) and a collate function
returning (X, mask, y), with float32 labels of shape (N, 1).

Build Conv1d(4, n_filters, odd kernel_size, same padding) -> ReLU ->
masked global max pooling -> Dropout -> Linear(n_filters, 1).
Ignore padded positions when pooling and return raw logits.
Compute positive-class weight as n_negative / n_positive, and AUROC from
scratch, awarding half credit to ties and rejecting single-class labels.
Train for one epoch and evaluate with sample-weighted mean losses.
Evaluation must disable dropout and gradients and compute AUROC over all batches.
The final test requires held-out AUROC > 0.9 on a planted TATAAT motif.

Implement the functions, SequenceDataset, and MotifCNN in motif_detection\\solve_this.py.
Suggested time box: 35-45 minutes."""
    motif: ClassVar[str] = "TATAAT"

    @staticmethod
    @abstractmethod
    def reverse_complement(seq: str) -> str:
        """Return uppercase reverse complement; allow A/C/G/T/N."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def one_hot_channels_first(seq: str) -> torch.Tensor:
        """Return float32 DNA of shape (4, L), encoding N as 0.25."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pad_batch(sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-padded (X, mask); reject an empty batch."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_dataset(sequences: Sequence[str], labels: Sequence[int]) -> Dataset[tuple[str, float]]:
        """Create a dataset; reject mismatched sequence/label counts."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def collate_sequences(
        batch: Sequence[tuple[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (X, mask, y) for a DataLoader batch."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_model(n_filters: int = 16, kernel_size: int = 7, dropout: float = 0.2) -> nn.Module:
        """Create a motif CNN; reject even kernel sizes."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
        """Return float n_negative / n_positive of shape (1,); reject no positives."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Return AUROC with half credit for ties; require both classes."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        """Train one epoch and return the sample-weighted mean loss."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[float, float]:
        """Return (mean loss, full-dataset AUROC) in eval mode without gradients."""
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "MotifExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    @classmethod
    def make_motif_dataset(
        cls, n: int, seed: int, pos_frac: float = 0.3,
    ) -> tuple[list[str], list[int]]:
        """Random DNA of length 30-40; positives contain the exercise motif."""
        rng = random.Random(seed)
        seqs, labels = [], []
        for _ in range(n):
            length = rng.randint(30, 40)
            s = [rng.choice("ACGT") for _ in range(length)]
            label = 1 if rng.random() < pos_frac else 0
            if label:
                start = rng.randint(0, length - len(cls.motif))
                s[start:start + len(cls.motif)] = cls.motif
            else:
                # Exclude accidental motif occurrences from negative examples.
                while cls.motif in "".join(s):
                    s = [rng.choice("ACGT") for _ in range(length)]
            seqs.append("".join(s))
            labels.append(label)
        return seqs, labels

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Exercise02Tests(unittest.TestCase):
            def test_reverse_complement(self):
                self.assertEqual(solution.reverse_complement("AACGTN"), "NACGTT")
                self.assertEqual(solution.reverse_complement("acgt"), "ACGT")

            def test_reverse_complement_is_involution(self):
                s = "GATTACANNCG"
                self.assertEqual(solution.reverse_complement(solution.reverse_complement(s)), s)

            def test_reverse_complement_rejects_invalid(self):
                with self.assertRaises(ValueError):
                    solution.reverse_complement("ACGX")

            def test_one_hot_channels_first(self):
                got = solution.one_hot_channels_first("ACgT")
                self.assertEqual(tuple(got.shape), (4, 4))
                self.assertEqual(got.dtype, torch.float32)
                self.assertTrue(torch.equal(got, torch.eye(4)))

            def test_one_hot_handles_n(self):
                got = solution.one_hot_channels_first("AN")
                self.assertTrue(torch.allclose(got[:, 1], torch.full((4,), 0.25)))
                self.assertAlmostEqual(got.sum().item(), 2.0)

            def test_one_hot_rejects_invalid(self):
                with self.assertRaises(ValueError):
                    solution.one_hot_channels_first("ACZ")

            def test_pad_batch_shapes_and_mask(self):
                X, mask = solution.pad_batch(["ACG", "A", "ACGTA"])
                self.assertEqual(tuple(X.shape), (3, 4, 5))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertEqual(mask.sum(dim=1).tolist(), [3, 1, 5])
                self.assertTrue(torch.equal(X[1, :, 1:], torch.zeros(4, 4)))
                self.assertTrue(torch.equal(X[0, :, :3], solution.one_hot_channels_first("ACG")))

            def test_pad_batch_rejects_empty(self):
                with self.assertRaises(ValueError):
                    solution.pad_batch([])

            def test_dataset(self):
                ds = solution.create_dataset(["AC", "GTA"], [0, 1])
                self.assertEqual(len(ds), 2)
                seq, label = ds[1]
                self.assertEqual(seq, "GTA")
                self.assertIsInstance(label, float)
                with self.assertRaises(ValueError):
                    solution.create_dataset(["AC"], [0, 1])

            def test_dataloader_with_collate(self):
                ds = solution.create_dataset(["AC", "GTA", "T"], [0, 1, 1])
                loader = DataLoader(ds, batch_size=3, collate_fn=solution.collate_sequences)
                X, mask, y = next(iter(loader))
                self.assertEqual(tuple(X.shape), (3, 4, 3))
                self.assertEqual(tuple(mask.shape), (3, 3))
                self.assertEqual(tuple(y.shape), (3, 1))
                self.assertEqual(y.dtype, torch.float32)

            def test_model_structure(self):
                model = solution.create_model(n_filters=8, kernel_size=5)
                self.assertIsInstance(model.conv, nn.Conv1d)
                self.assertEqual(model.conv.in_channels, 4)
                self.assertEqual(model.conv.out_channels, 8)
                self.assertIsInstance(model.fc, nn.Linear)
                self.assertEqual(model.fc.out_features, 1)
                with self.assertRaises(ValueError):
                    solution.create_model(kernel_size=4)

            def test_model_output_shape_variable_length(self):
                model = solution.create_model().eval()
                for length in (7, 20, 53):
                    out = model(torch.randn(3, 4, length))
                    self.assertEqual(tuple(out.shape), (3, 1))

            def test_masking_makes_output_independent_of_padding(self):
                # Pooling must ignore the extra positions introduced by batching.
                torch.manual_seed(0)
                model = solution.create_model(n_filters=8, kernel_size=5).eval()
                short = "ACGTACGTAC"
                X_alone, mask_alone = solution.pad_batch([short])
                X_batch, mask_batch = solution.pad_batch([short, "G" * 60])
                with torch.no_grad():
                    alone = model(X_alone, mask_alone)[0]
                    batched = model(X_batch, mask_batch)[0]
                self.assertTrue(torch.allclose(alone, batched, atol=1e-6))

            def test_mask_actually_used(self):
                torch.manual_seed(0)
                model = solution.create_model(n_filters=8, kernel_size=1).eval()
                x1 = torch.zeros(1, 4, 10)
                x1[0, 0, 0] = 1
                x2 = x1.clone()
                x2[0, :, 1:] = 5.0
                mask = torch.zeros(1, 10, dtype=torch.bool)
                mask[0, 0] = True
                with torch.no_grad():
                    self.assertTrue(torch.allclose(model(x1, mask), model(x2, mask)))

            def test_pos_weight(self):
                w = solution.compute_pos_weight(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0]))
                self.assertEqual(tuple(w.shape), (1,))
                self.assertAlmostEqual(w.item(), 3.0)
                with self.assertRaises(ValueError):
                    solution.compute_pos_weight(torch.zeros(5))

            def test_auroc_known_values(self):
                labels = torch.tensor([0, 0, 1, 1])
                self.assertAlmostEqual(solution.auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
                self.assertAlmostEqual(solution.auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
                self.assertAlmostEqual(solution.auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), labels), 0.75)

            def test_auroc_ties_and_shapes(self):
                self.assertAlmostEqual(
                    solution.auroc(torch.ones(4, 1), torch.tensor([[0], [1], [0], [1]])), 0.5,
                )

            def test_auroc_rejects_single_class(self):
                with self.assertRaises(ValueError):
                    solution.auroc(torch.rand(4), torch.ones(4))

            def test_evaluate_is_deterministic_and_gradient_free(self):
                torch.manual_seed(0)
                seqs, labels = solution.make_motif_dataset(40, seed=1)
                loader = DataLoader(solution.create_dataset(seqs, labels), batch_size=16,
                                    collate_fn=solution.collate_sequences)
                model = solution.create_model(dropout=0.5)
                model.train()  # evaluate must switch to eval mode itself
                loss_fn = nn.BCEWithLogitsLoss()
                r1 = solution.evaluate(model, loader, loss_fn)
                r2 = solution.evaluate(model, loader, loss_fn)
                self.assertEqual(r1, r2)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

            def test_train_epoch_updates_parameters(self):
                torch.manual_seed(0)
                seqs, labels = solution.make_motif_dataset(40, seed=2)
                loader = DataLoader(solution.create_dataset(seqs, labels), batch_size=16,
                                    shuffle=True, collate_fn=solution.collate_sequences)
                model = solution.create_model()
                before = [p.detach().clone() for p in model.parameters()]
                loss = solution.train_epoch(
                    model, loader, torch.optim.Adam(model.parameters(), 1e-2), nn.BCEWithLogitsLoss(),
                )
                self.assertIsInstance(loss, float)
                self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, model.parameters())))

            def test_learns_planted_motif_on_held_out_data(self):
                torch.manual_seed(42)
                tr_seqs, tr_labels = solution.make_motif_dataset(300, seed=10)
                va_seqs, va_labels = solution.make_motif_dataset(100, seed=20)
                train_loader = DataLoader(solution.create_dataset(tr_seqs, tr_labels), batch_size=32,
                                          shuffle=True, collate_fn=solution.collate_sequences)
                val_loader = DataLoader(solution.create_dataset(va_seqs, va_labels), batch_size=64,
                                        collate_fn=solution.collate_sequences)
                model = solution.create_model(n_filters=16, kernel_size=7)
                pos_weight = solution.compute_pos_weight(torch.tensor(tr_labels))
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                opt = torch.optim.Adam(model.parameters(), lr=1e-2)
                for _ in range(25):
                    solution.train_epoch(model, train_loader, opt, loss_fn)
                _, val_auc = solution.evaluate(model, val_loader, loss_fn)
                self.assertGreater(val_auc, 0.9)

        Exercise02Tests.__name__ = f"Exercise02Tests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Exercise02Tests)
