# Exercise 06: two-part gene-sequence interview

Practice in [solve_this.py](solve_this.py). Data and a working part-1 model
are supplied in [support.py](support.py). Tests and exact contracts are in
[exercise.py](exercise.py); answers are separate in
[model_solution.py](model_solution.py).

## Mock interview format

- Use a CPU runtime; no downloads or biological datasets are required.
- Documentation is allowed. Disable AI completions/chat during the mock.
- Choose **one debugging variant and one module**, not all six tasks.
- Spend 20 minutes debugging, then **move on** to 15 minutes implementing,
  even if debugging is unfinished. Reserve five minutes to explain your work.
- Explain expected shapes, inspect a gradient, and show that your fix improves
  prediction error, not merely the reported training objective.

Try three independent mocks: signed loss + GC regressor; negative-square
loss + motif classifier; cubic loss + residual block.

## Part 1: three broken objectives

Let `r = target - prediction`, with a real-valued regression output:

| Case | Supplied objective | What to investigate |
|---|---|---|
| `signed` | `mean(r)` | Errors cancel; increasing predictions drives loss toward minus infinity. |
| `negative_square` | `-mean(r**2)` | Minimization rewards larger errors in either direction. |
| `cubic` | `mean(r**3)` | Sign-sensitive, unbounded below, and gradients grow quadratically with residual size. |

**Important distinction:** the first objective is affine, so it is both convex
and concave. The second is concave (and non-convex); the third is neither
convex nor concave over the real line. All three are unbounded below.
Non-convexity alone is not a defect: many useful neural objectives are
non-convex in model parameters. Even a convex loss in predictions may yield
a non-convex optimization problem after composition with a neural network.

For these exercises, repair each variant to **mean squared error**. MSE is
nonnegative and minimized when predictions equal targets. Merely clamping
the loss, taking its absolute value after reduction, lowering the learning
rate, or detaching predictions does not meet the contract.

The supplied training loop also contains independent training/shape bugs.
Inspect target versus prediction dimensions, gradient lifetime, and model
mode. Preserve raw model outputs; the tiny GC-fraction regression does not
need a sigmoid. Inputs are uppercase, equal-length DNA strings, encoded as
float32 `(batch, 4, length)` in A/C/G/T channel order.

Run this diagnostic from the repository root (or as a Colab cell):

```python
import torch
from gene_debugging import solve_this as candidate
from gene_debugging.support import SuppliedRegressor, regression_data

torch.manual_seed(6)
x, target = regression_data()
model = SuppliedRegressor()
print("shapes:", x.shape, target.shape, model(x).shape)
with torch.no_grad():
    print("MSE before:", (model(x).flatten() - target).square().mean().item())
losses = candidate.train_model(model, x, target, "signed", epochs=20, lr=0.1)
print("objective first/last:", losses[0], losses[-1])
with torch.no_grad():
    print("MSE after:", (model(x).flatten() - target).square().mean().item())
    print("predictions:", model(x).flatten())
```

Change `"signed"` to either other case and create a **fresh** model each
time. Exploding or non-finite losses are evidence to investigate, not a
reason to suppress errors. After fixing the loss, repeat with targets of
shape `(batch,)` and `(batch, 1)`.

## Part 2: three independent nn.Module options

Layer attribute names are unrestricted. Register layers in `__init__`,
call `super().__init__()`, and preserve autograd.

1. **GCRegressor (easy, 5-10 minutes):** average the length dimension of
   `(B, 4, L)`, then apply `Linear(4, 1)`. Output `(B, 1)`, no sigmoid.
   Test `B=1`, `L=1`, and a batch with several strings.
2. **MotifClassifier (easy-medium, 10-15 minutes):**
   `Conv1d(4, filters, kernel_size)` with valid padding, ReLU, global max
   over length, then `Linear(filters, 1)`. Use biases and return raw logits
   `(B, 1)`, suitable for `BCEWithLogitsLoss`. Length is at least the kernel
   size. Check length equal to the kernel size and longer sequences.
3. **ResidualBlock (easy-medium, 10-15 minutes):**
   `x + Linear(hidden_dim, features)(ReLU(Linear(features, hidden_dim)(x)))`.
   Use biases, no final activation. Accept `(B, features)` and
   `(B, L, features)`. With every parameter zero, output must equal input,
   including negative entries.

The tests check formulas, not only shapes: correct pooling axes, no output
squashing, registered parameters, finite gradients, and the residual skip.
The GC learning check uses tiny synthetic training data, not held-out
biological validation.

## Running tests locally

From the repository root:

```powershell
python test.py --exercise 06 --describe
python test.py --exercise 06 --solution own
# Only after finishing your attempt:
python test.py --exercise 06 --solution model
python test.py --exercise 06 --compare
```

The untouched candidate is deliberately expected to fail. To test just your
selected task without failing on other unfinished modules, use this Python
cell/script:

```python
import unittest
from gene_debugging.solve_this import UserSolution

selected = {
    "test_loss_signed",
    "test_training_matches_two_sgd_steps",
    "test_training_learns_gc_and_zero_epochs_does_not_update",
    "test_module_gc_values_shapes_and_gradients",
}
# Swap test_loss_signed for test_loss_negative_square or test_loss_cubic.
# Swap the module test for test_module_motif_matches_spec_and_gradients or
# test_module_residual_formula_shapes_and_gradients.
# Training tests cover all variants: finish all three loss repairs first,
# or omit the two training tests while isolating a single loss/module.
suite = unittest.TestSuite(
    test for test in UserSolution().build_suite()
    if test.id().rsplit(".", 1)[-1] in selected
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
```

## Google Colab

Use a CPU notebook. Upload **only** the root `exercise.py` and the new
`gene_debugging` folder as a zip archive, preserving this structure:

```text
exercise.py
gene_debugging/
    __init__.py
    exercise.py
    support.py
    solve_this.py
    model_solution.py
```

Do not include your virtual environment, Git history, or unrelated files.
Extract your uploaded archive using Colab's Files pane into `/content`,
then run:

```python
%cd /content
import torch
print(torch.__version__)
from gene_debugging.solve_this import UserSolution
print(UserSolution.problem)
```

PyTorch is normally preinstalled; if that import fails, install `torch`
with `%pip install torch`. This standalone workflow does not need
scikit-learn, the root runner, network data access, or a GPU.

Edit `gene_debugging/solve_this.py` in the Files pane, save, then restart
the Colab session before rerunning imports and the diagnostic/selected-test
cells above so Python does not reuse an old imported implementation.
For a notebook-only attempt, paste the candidate functions/classes into a
cell and rebind them to `UserSolution` as at the bottom of the candidate
file. Keep the reference file closed until reviewing.
