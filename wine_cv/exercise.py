"""Exercise 05 contract and offline tests; no reference implementation imports."""
import unittest
from abc import abstractmethod
from collections.abc import Sequence
from unittest.mock import patch

import torch
from torch import nn

from exercise import Exercise

from . import support
from .support import FoldResult, TrainingConfig, TrialResult


class WineCVExercise(Exercise):
    title = "Real-data Wine classification: leakage-free k-fold CV"
    problem = """Implement four functions in wine_cv\\solve_this.py.

Real task: identify one of three wine cultivars from 13 chemical measurements.
Use sklearn.datasets.load_wine: 178 real UCI samples, bundled with sklearn.
The fixed model, data loading, stratified splitting, batch loops and final
refit runner are SUPPLIED, not tasks. Model: Linear(13,32) -> ReLU ->
Linear(32,3), 547 parameters, raw logits with CrossEntropyLoss/int64 labels.
This needs far less than 16 GB VRAM; CPU is recommended for this tiny dataset.

1. standardize: fit feature-wise mean and population std on the training fold
   ONLY; transform training and validation with those statistics. Replace zero
   std by 1. Do not mutate inputs. This is not per-row normalization.
2. fit_fold: seed torch; create a NEW model on device, NEW AdamW and shuffled
   loader on EVERY call. Use config.lr/weight_decay and supplied
   support.make_loader/train_epoch/evaluate. After each epoch, track strictly
   improving validation cross entropy. Copy checkpoint tensors independently;
   state_dict() alone aliases live weights. Stop after patience consecutive
   non-improvements or max_epochs. Reset patience on improvement. Restore best
   weights, leave eval mode, return FoldResult(model, one-based epoch, loss).
3. cross_validate: call support.stratified_folds(y,n_splits,seed). For every
   fold call self.standardize and self.fit_fold (seed + zero-based fold index).
   Evaluate the RESTORED model; return TrialResult with one loss and best epoch
   per fold plus detached CPU float32 (N,3) out-of-fold logits in original row
   order. Every sample is validation exactly once. Reuse identical split seeds
   across trials; never carry weights, optimizer state or scaling across folds.
4. select_best: minimize the unweighted arithmetic mean of fold validation
   losses. Keep the first trial on exact ties. Reject empty trials with ValueError.

Run unit tests: python test.py --exercise 05 --solution own
Real experiment: python -m wine_cv.run --solution own
The runner locks away a stratified 20% test set BEFORE CV. On the remaining
142 samples, compare 5-fold CV for lr in {0.001,0.01} and weight_decay in
{0,0.01}: 20 fits. Report mean/std validation CE and OOF accuracy.
Choose by mean CV loss, then train a fresh model/scaler on all development
data for int(median(best fold epochs)), at least 1 epoch. Evaluate the 36 test
samples ONCE. Do not tune from that result. CV scores used for selection are
optimistic; the untouched test is the final estimate, not the tuning objective.

Discussion: Why stratify? Why is global standardization leakage? Why must
AdamW be recreated? Why can test-driven early stopping leak? Why are tiny
fold/test scores noisy? How would you use nested CV without a held-out test?
Suggested time box: 45-60 minutes."""

    @staticmethod
    @abstractmethod
    def standardize(
        train_x: torch.Tensor, val_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fit_fold(
        train_x: torch.Tensor, train_y: torch.Tensor,
        val_x: torch.Tensor, val_y: torch.Tensor,
        config: TrainingConfig, seed: int, device: torch.device,
    ) -> FoldResult:
        raise NotImplementedError

    @abstractmethod
    def cross_validate(
        self, x: torch.Tensor, y: torch.Tensor, config: TrainingConfig,
        n_splits: int = 5, seed: int = 42, device: torch.device = torch.device("cpu"),
    ) -> TrialResult:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def select_best(trials: Sequence[TrialResult]) -> TrialResult:
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "WineCVExercise":
        from .model_solution import ModelSolution

        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self
        device = torch.device("cpu")

        class Exercise05Tests(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.previous_threads = torch.get_num_threads()
                torch.set_num_threads(1)

            @classmethod
            def tearDownClass(cls):
                torch.set_num_threads(cls.previous_threads)

            def test_standardization_is_featurewise_training_only_and_nonmutating(self):
                train = torch.tensor([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]).repeat(1, 7)[:, :13]
                val = torch.full((2, 13), 101.0)
                original_train, original_val = train.clone(), val.clone()
                actual_train, actual_val = solution.standardize(train, val)
                scale = ((original_train - original_train.mean(0)) ** 2).mean(0).sqrt()
                scale[scale == 0] = 1
                torch.testing.assert_close(actual_train, (original_train - original_train.mean(0)) / scale)
                torch.testing.assert_close(actual_val, (original_val - original_train.mean(0)) / scale)
                torch.testing.assert_close(train, original_train)
                torch.testing.assert_close(val, original_val)
                self.assertTrue(torch.isfinite(actual_val).all())

            def test_single_training_row_and_constant_columns(self):
                train, val = torch.ones(1, 13), torch.full((2, 13), 3.0)
                a, b = solution.standardize(train, val)
                torch.testing.assert_close(a, torch.zeros_like(train))
                torch.testing.assert_close(b, torch.full_like(val, 2.0))

            def check_checkpoint(self, losses: list[float], best_epoch: int, epochs_run: int):
                models, optimizers, seeds = [], [], []

                def create_model():
                    seeds.append(torch.initial_seed())
                    model = nn.Linear(13, 3)
                    nn.init.zeros_(model.weight)
                    nn.init.zeros_(model.bias)
                    models.append(model)
                    return model

                def train(model, loader, optimizer, actual_device):
                    self.assertIsInstance(optimizer, torch.optim.AdamW)
                    self.assertEqual(optimizer.defaults["lr"], 0.003)
                    self.assertEqual(optimizer.defaults["weight_decay"], 0.02)
                    self.assertEqual(actual_device, device)
                    self.assertEqual(loader.batch_size, 4)
                    self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)
                    optimizers.append(optimizer)
                    with torch.no_grad():
                        model.weight.add_(1)
                        model.bias.add_(1)
                    return 1.0

                config = TrainingConfig(
                    lr=0.003, weight_decay=0.02, batch_size=4,
                    max_epochs=len(losses), patience=2,
                )
                x, y = torch.zeros(9, 13), torch.arange(9) % 3
                with patch.object(support, "create_model", side_effect=create_model), \
                        patch.object(support, "train_epoch", side_effect=train) as train_mock, \
                        patch.object(support, "evaluate") as evaluate_mock:
                    for seed in (17, 18):
                        evaluate_mock.side_effect = [(loss, torch.zeros(9, 3)) for loss in losses]
                        result = solution.fit_fold(x, y, x, y, config, seed, device)
                        self.assertEqual(result.best_epoch, best_epoch)
                        self.assertAlmostEqual(result.val_loss, losses[best_epoch - 1])
                        torch.testing.assert_close(
                            result.model.state_dict()["weight"],
                            torch.full((3, 13), float(best_epoch)),
                        )
                        self.assertFalse(result.model.training)
                    self.assertEqual(train_mock.call_count, epochs_run * 2)
                    self.assertEqual(evaluate_mock.call_count, epochs_run * 2)
                self.assertEqual(seeds, [17, 18])
                self.assertIsNot(models[0], models[1])
                self.assertTrue(all(opt is optimizers[0] for opt in optimizers[:epochs_run]))
                self.assertTrue(all(opt is optimizers[-1] for opt in optimizers[epochs_run:]))
                self.assertIsNot(optimizers[0], optimizers[-1])

            def test_fit_fold_restores_independent_checkpoint_and_stops_on_ties(self):
                self.check_checkpoint([0.9, 0.5, 0.5, 0.7, 0.1], best_epoch=2, epochs_run=4)

            def test_fit_fold_resets_patience_after_improvement(self):
                self.check_checkpoint([0.9, 1.0, 0.8, 0.85, 0.85, 0.1], best_epoch=3, epochs_run=5)

            def test_fit_fold_respects_max_epochs(self):
                self.check_checkpoint([0.9, 0.8, 0.7], best_epoch=3, epochs_run=3)

            def test_cross_validation_wires_folds_scaling_seeds_and_oof_order(self):
                x = torch.arange(27 * 13, dtype=torch.float32).reshape(27, 13)
                y = torch.arange(27) % 3
                folds = support.stratified_folds(y, 3, 19)
                config = TrainingConfig(max_epochs=1)
                expected_logits = torch.empty(27, 3)
                fit_calls = []

                class ConstantModel(nn.Module):
                    def __init__(self, value: int):
                        super().__init__()
                        self.value = value

                    def forward(self, inputs):
                        return torch.tensor([float(self.value), 0.0, -1.0]).expand(len(inputs), 3)

                def fit(train_x, train_y, val_x, val_y, actual_config, seed, actual_device):
                    index = len(fit_calls)
                    train, val = folds[index]
                    mean = x[train].mean(0)
                    scale = x[train].std(0, correction=0)
                    torch.testing.assert_close(train_x, (x[train] - mean) / scale)
                    torch.testing.assert_close(val_x, (x[val] - mean) / scale)
                    torch.testing.assert_close(train_y, y[train])
                    torch.testing.assert_close(val_y, y[val])
                    self.assertEqual(actual_config, config)
                    self.assertEqual(seed, 19 + index)
                    self.assertEqual(actual_device, device)
                    model = ConstantModel(index)
                    expected_logits[val] = model(val_x)
                    loss = nn.functional.cross_entropy(model(val_x), val_y).item()
                    fit_calls.append(loss)
                    return FoldResult(model, index + 1, loss)

                with patch.object(solution, "fit_fold", side_effect=fit):
                    trial = solution.cross_validate(x, y, config, 3, 19, device)
                self.assertEqual(len(fit_calls), 3)
                self.assertEqual(trial.config, config)
                self.assertEqual(trial.best_epochs, [1, 2, 3])
                torch.testing.assert_close(torch.tensor(trial.fold_losses), torch.tensor(fit_calls))
                torch.testing.assert_close(trial.oof_logits, expected_logits)
                self.assertFalse(trial.oof_logits.requires_grad)
                self.assertEqual(trial.oof_logits.device.type, "cpu")

            def test_stratified_folds_have_full_disjoint_reproducible_coverage(self):
                y = torch.arange(45) % 3
                folds = support.stratified_folds(y, 5, 42)
                seen = torch.zeros(45, dtype=torch.long)
                for (train, val), (again_train, again_val) in zip(
                    folds, support.stratified_folds(y, 5, 42),
                ):
                    self.assertFalse(torch.isin(train, val).any())
                    self.assertEqual(len(train) + len(val), len(y))
                    self.assertEqual(y[val].bincount().tolist(), [3, 3, 3])
                    seen[val] += 1
                    torch.testing.assert_close(train, again_train)
                    torch.testing.assert_close(val, again_val)
                torch.testing.assert_close(seen, torch.ones_like(seen))
                with self.assertRaises(ValueError):
                    support.stratified_folds(y, 16, 42)

            def test_selection_uses_mean_loss_with_stable_ties(self):
                config = TrainingConfig()
                logits = torch.zeros(9, 3)
                first = TrialResult(config, [0.1, 1.0, 1.0], [1] * 3, logits)
                best = TrialResult(config, [0.5, 0.5, 0.5], [2] * 3, logits)
                tied = TrialResult(config, [0.5, 0.5, 0.5], [3] * 3, logits)
                self.assertIs(solution.select_best([first, best, tied]), best)
                with self.assertRaises(ValueError):
                    solution.select_best([])

            def test_configuration_rejects_invalid_values(self):
                for kwargs in (
                    {"lr": 0.0}, {"lr": float("nan")}, {"weight_decay": -1},
                    {"batch_size": 0}, {"max_epochs": 0}, {"patience": 0},
                    {"max_epochs": 1.5},
                ):
                    with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                        TrainingConfig(**kwargs)

            def test_real_wine_cv_learns_and_is_reproducible(self):
                from sklearn.model_selection import train_test_split

                x, y = support.load_data()
                self.assertEqual(x.shape, (178, 13))
                self.assertEqual(y.bincount().tolist(), [59, 71, 48])
                dev, _ = train_test_split(
                    torch.arange(len(y)).numpy(), test_size=0.2,
                    stratify=y.numpy(), random_state=42,
                )
                x, y = x[dev], y[dev]
                config = TrainingConfig(max_epochs=60, patience=8)
                first = solution.cross_validate(x, y, config, 3, 42, device)
                second = solution.cross_validate(x, y, config, 3, 42, device)
                self.assertGreater((first.oof_logits.argmax(1) == y).float().mean().item(), 0.85)
                self.assertLess(first.mean_loss, 0.7)
                self.assertTrue(torch.isfinite(first.oof_logits).all())
                torch.testing.assert_close(first.oof_logits, second.oof_logits)
                self.assertEqual(first.best_epochs, second.best_epochs)
                self.assertEqual(len(first.fold_losses), 3)
                self.assertTrue(all(1 <= epoch <= 60 for epoch in first.best_epochs))

        Exercise05Tests.__name__ = f"Exercise05Tests_{type(solution).__module__}"
        return unittest.defaultTestLoader.loadTestsFromTestCase(Exercise05Tests)
