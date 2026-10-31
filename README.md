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
`-- pytorch_drills
    |-- __init__.py
    |-- exercise.py
    |-- solve_this.py
    `-- model_solution.py
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
generalization; the dataset is deliberately tiny and synthetic.

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
