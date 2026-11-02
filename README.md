# PyTorch Live Coding Interview Practice

## Files

Shared files stay at the root; each exercise has its own Python package:

```text
InterviewPractice
|-- exercise.py
|-- test.py
|-- requirements.txt
|-- README.md
|-- dna_classifier
|   |-- __init__.py
|   |-- exercise.py
|   |-- solve_this.py
|   `-- model_solution.py
|-- motif_detection
|   |-- __init__.py
|   |-- exercise.py
|   |-- solve_this.py
|   `-- model_solution.py
|-- pytorch_drills
|   |-- __init__.py
|   |-- exercise.py
|   |-- solve_this.py
|   `-- model_solution.py
|-- tensor_manipulation
|   |-- __init__.py
|   |-- exercise.py
|   |-- solve_this.py
|   `-- model_solution.py
`-- wine_cv
    |-- __init__.py
    |-- exercise.py
    |-- solve_this.py
    |-- model_solution.py
    |-- support.py
    `-- run.py
```

- Root `exercise.py`: abstract base class shared by every exercise.
- Root `test.py`: registry, shared command-line runner, and framework regression tests.
- Each package's `exercise.py`: problem description, abstract solution methods,
  tests, and lazy reference-solution factory.
- Each package's `solve_this.py`: your implementation, without reference code.
- Each package's `model_solution.py`: separate reference implementation.
- Package `__init__.py` files do not import solutions.

## Setup

Use Python 3.10+ in a virtual environment if possible:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Run the tests

Run commands from the repository root, not from an individual exercise folder.

List the available exercises:

```bash
python test.py --list
```

Read a problem without loading either solution:

```bash
python test.py --exercise 02 --describe
```

Select an exercise and test your own implementation:

```bash
python test.py --exercise 02 --solution own
```

Test the reference implementation, or compare both implementations:

```bash
python test.py --exercise 02 --solution model
python test.py --exercise 02 --solution both
python test.py --exercise 02 --compare
```

Run all registered exercises against the reference solutions:

```bash
python test.py --exercise all --solution model
```

With no options, the runner tests your implementation of exercise `01`.
The original `--solution solve_this`, `--solution model_solution`, and `--compare`
commands still work; the module-name options are aliases for `own` and `model`.
Each selected exercise runs even if an earlier suite fails. Import errors,
incomplete abstract subclasses, and other solution-loading errors are reported
explicitly and marked as failures, without preventing the reference solution
or subsequent exercises from running. The runner exits with code 0 only if every
selected suite passes, 1 for test/loading failures, and 2 for invalid command-line
arguments. Comparing runs the same tests independently, not a weight-by-weight
comparison. Empty test suites are rejected.

The scaffold is expected to fail until you implement it. The reference solution should pass.
The tests cover encoding, input validation, model output shape, gradient flow, training,
and prediction behavior. Passing these tests is not a proof of biological usefulness or
generalization. Exercises 01-02 use deliberately tiny synthetic datasets;
exercise 05 uses a small real dataset and a separate held-out test set.

## Debugging in VS Code

`.vscode/launch.json` has an "own solution" and a "model solution" configuration for
each exercise. Pick one in the Run and Debug view, set breakpoints in that exercise's
`solve_this.py` or `exercise.py`, and press F5. When adding an exercise, copy a pair of
configurations and change the `--exercise` ID.

## Exercise 02: CNN motif detection (harder)

Work in `motif_detection\solve_this.py`. Its contract and tests live in
`motif_detection\exercise.py`; the reference implementation stays in
`motif_detection\model_solution.py`.

Covers reverse complement, channels-first one-hot encoding with `N` handling, padding and
masking of variable-length sequences, a custom `Dataset` and `collate_fn`, a `Conv1d` motif
scanner with masked global max pooling, class imbalance (`pos_weight`), AUROC from scratch,
and correct `train()`/`eval()`/`no_grad` handling. The final test checks that the model learns
a planted `TATAAT` motif on held-out data (validation AUROC > 0.9).

```bash
python test.py --exercise 02 --solution own
python test.py --exercise 02 --compare
```

Suggested time box: 35–45 minutes.

## Exercise 03: PyTorch tensor and vectorization drills

Work in `pytorch_drills\solve_this.py`. The nine tasks cover broadcasting,
boolean masking, row-wise normalization, pairwise distances, attention masks,
graph aggregation, and padding variable-size graph tensors. Implement the
functions without Python loops where specified; the reference solution is
available separately in `pytorch_drills\model_solution.py`.

```bash
python test.py --exercise 03 --solution own
python test.py --exercise 03 --compare
```

Suggested time box: 20 minutes.

## Exercise 04: Advanced tensor dimension operations

Work in `tensor_manipulation\solve_this.py`. These eleven vectorized tasks add
new operations beyond exercise 03: sliding windows, batched cosine similarity,
weighted pooling, masked softmax, attention-head reshape/permutation, gathering
sequence endpoints, image patchification, top-k feature selection, channel
shuffling, and batched diagonals. There are no repeated squared-distance,
masked-mean, attention-mask construction, or graph-padding tasks.
The reference solution in
`tensor_manipulation\model_solution.py` explains the tensor-shape steps.

```bash
python test.py --exercise 04 --solution own
python test.py --exercise 04 --compare
```

Suggested time box: 45–60 minutes.

## Exercise 05: Real-data training and k-fold cross-validation

Work in `wine_cv\solve_this.py`. Predict a wine's cultivar from chemical
measurements using the [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
([sklearn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)):
178 samples, 13 numeric features, three classes, no missing values. The dataset
is bundled with scikit-learn, so no download, account, or external files are
needed at runtime. Install the updated `requirements.txt` first.

The supplied model is `Linear(13, 32) -> ReLU -> Linear(32, 3)`:
**547 parameters**, float32 logits, integer labels, and cross-entropy loss.
Weights alone are about 2.2 KB; weights, gradients and AdamW moment tensors
total about 8.8 KB, excluding activations/framework/CUDA overhead. This is
comfortably below your RTX 5080's 16 GB budget, but too small to benefit from
that GPU. CPU is the default and recommended starting point.

Unlike earlier exercises, you do **not** implement encoding, tensor drills,
a model architecture, a Dataset, a batch loop, or a metric from scratch.
Those are supplied in `wine_cv\support.py`. The four candidate tasks focus on:

1. Fold-local feature standardization, with no validation leakage.
2. Fresh model/AdamW setup per fold, reproducible shuffling, early stopping,
   independent checkpoints, and restoring the best epoch.
3. Stratified k-fold orchestration and correctly ordered out-of-fold predictions.
4. Selecting hyperparameters by mean validation cross entropy, not test accuracy.

```powershell
python test.py --exercise 05 --describe
python test.py --exercise 05 --solution own
python test.py --exercise 05 --solution model

# Full real-data experiment, once you have implemented the tasks:
python -m wine_cv.run --solution own
# Separate reference implementation:
python -m wine_cv.run --solution model
# Optional, with a CUDA-enabled PyTorch build compatible with your GPU:
python -m wine_cv.run --solution own --device cuda
```

The experiment reserves a stratified **20% test set (36 samples)** before
tuning. On the other **142 samples**, it runs **5-fold CV** over
`lr = {0.001, 0.01}` and `weight_decay = {0, 0.01}`: **20 independent fits**,
each capped at 100 epochs with patience 12. All trials reuse the same folds
and per-fold seeds. The default console report shows per-trial fold tables,
mean and population standard deviation of fold losses, OOF accuracy, a
comparison table marking the selected configuration, and a separate final
test summary with percentage accuracy and the number of correct predictions.
Losses are displayed to six decimals; selection still uses full precision.
For machine-readable output, use `--output json` to retain the original
JSON Lines format (four trial records followed by one final test record):

```powershell
python -m wine_cv.run --solution model --output json
```

After selection, a fresh model and scaler are fitted on all development data.
The refit uses the integer median of the winning trial's best epochs, not
test-driven early stopping. The test set is evaluated **once**, at the end.
CV results used for early stopping and hyperparameter selection are optimistic;
they are not an unbiased final score. Do not iterate on the test result.
Fold standard deviation is descriptive, not a confidence interval. With only
36 test examples, one changed prediction moves accuracy by about 2.8 points.
This is a training-protocol exercise, not evidence that an MLP beats simpler
classifiers or is suitable for production.

For a quick pipeline smoke test, add `--folds 3 --epochs 5 --patience 2`.
The default seed is 42; `--seed` changes the held-out split and fold seeds.
The unit tests are offline/CPU and check leakage, patience resets, checkpoint
aliasing/restoration, fresh optimizers, OOF ordering, deterministic folds, and
learning on the real development data. They never score the final test set.

Suggested time box: 45-60 minutes. Discussion prompts are in `--describe`.

## Adding an exercise

An exercise definition subclasses `Exercise` and owns:

- `title` and `problem`: strings used by `--list` and `--describe`.
- One or more `@abstractmethod` solution stubs, with signatures specific to
  the problem. A single-function exercise can simply define `solve`.
- `build_suite(self)`: a fresh `unittest.TestSuite` that tests the current instance.
- `model_solution()`: a class method that lazily imports and returns the
  reference subclass. Do not import reference implementations at module scope.
- Any exercise-specific variables or helpers, such as a motif or synthetic dataset.

The definition is abstract and cannot be instantiated until its solution stubs
are overridden. Both `UserSolution` and `ModelSolution` inherit the same tests;
the runner chooses which instance to test.

For example, create a `double_integer` folder with an `__init__.py` that does
not import solutions, then add `double_integer\exercise.py`:

```python
from abc import abstractmethod
import unittest

from exercise import Exercise


class DoubleExercise(Exercise):
    title = "Double an integer"
    problem = "Implement solve(value) to return twice the input integer."

    @abstractmethod
    def solve(self, value: int) -> int:
        raise NotImplementedError

    @classmethod
    def model_solution(cls) -> "DoubleExercise":
        from .model_solution import ModelSolution
        return ModelSolution()

    def build_suite(self) -> unittest.TestSuite:
        solution = self

        class Tests(unittest.TestCase):
            def test_positive(self):
                self.assertEqual(solution.solve(3), 6)

            def test_zero_and_negative(self):
                self.assertEqual(solution.solve(0), 0)
                self.assertEqual(solution.solve(-2), -4)

        return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
```

In the separate `double_integer\solve_this.py`, override only the solution stub:

```python
from .exercise import DoubleExercise


class UserSolution(DoubleExercise):
    def solve(self, value: int) -> int:
        raise NotImplementedError("Implement solve")
```

In `double_integer\model_solution.py`, import `DoubleExercise` from `.exercise`,
define `ModelSolution(DoubleExercise)`, and override `solve` with the reference
implementation. This file is never imported when listing/describing problems
or running only your implementation.

Finally, import the definition and register it in `test.py`:

```python
from double_integer.exercise import DoubleExercise

# Add this entry to EXERCISES:
"03": RegisteredExercise(DoubleExercise, "double_integer.solve_this"),
```

The CLI choices, `--list`, `--exercise all`, solution selection, and summary
automatically include registered exercises. No changes to the runner logic or
per-exercise test runners are needed.

The existing multi-function exercises use static abstract methods. Their
`UserSolution` subclasses bind these methods to the functions/classes already
in the user files with `staticmethod(...)`. Keep implementing the TODOs above
those bindings; the bindings override the abstract stubs without duplicating
your code or moving it into a file containing reference code. You can also
override methods directly in the subclass, as in the single-function example.

## Validating the framework

Run the runner's regression tests without running the ML exercises:

```bash
python -m unittest test.FrameworkTests
```

These cover abstract-method enforcement, registering a new exercise, user/model
selection, CLI validation, independent comparison, solution-loading failures,
metadata access without loading solutions, package import isolation, and
fresh/nonempty test suites.

## Suggested interview rules

- Give yourself 20 minutes to debug/implement the functions.
- Avoid opening `model_solution.py` until you're finished.
- Use the error messages and failing tests as feedback, just as you would in a coding interview.
