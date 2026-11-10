"""Reference answers; keep closed during timed practice.

What should a loss function satisfy?
-----------------------------------
There is no universal checklist for every learning algorithm. For ordinary
supervised learning by gradient-based minimization, check these properties:

1. Alignment with the task. Lower loss should reward the desired predictions,
   not an unintended shortcut. For this pointwise regression exercise, zero
   residual should minimize the loss. With noisy targets, MSE instead estimates
   the conditional mean; it cannot generally make every residual zero.
2. A meaningful lower bound on the allowed prediction domain. An objective
   that can decrease forever by sending predictions to infinity is ill-posed
   as a standalone minimization problem. Constraints or regularization may
   change that conclusion; assess the complete objective and feasible domain.
3. Well-defined, numerically stable values on valid inputs. Check domains of
   logs/divisions, finite outputs and gradients, and dtype/range compatibility.
   Even MSE can overflow for extreme finite inputs.
4. A usable optimization signal. For ordinary backpropagation, preserve the
   computation graph and provide useful derivatives or supported subgradients.
   Smoothness everywhere is not necessary (MAE has a kink at zero), but
   rounding, detaching, or saturated gradients can prevent useful learning.
   Other optimization methods need not require differentiability.
5. Correct inputs and reduction. Match the loss to the output representation
   (for example, BCEWithLogitsLoss expects logits), align prediction/target
   shapes, and reduce the intended examples to a scalar for loss.backward().
   Vector losses are also valid, but need reduction or an explicit upstream
   gradient. Square residuals before averaging so opposite errors cannot cancel.

Not universal requirements:
- Convexity: useful, but not necessary. A loss convex in predictions can still
  be non-convex in neural-network parameters.
- Nonnegativity or a minimum of exactly zero: convenient conventions, not
  necessities. Adding a constant changes neither gradients nor minimizers.
- Symmetry, uniqueness of the minimum, or unlimited growth with error:
  task-dependent choices. Asymmetric and bounded robust losses can be valid,
  although bounded losses may offer weak gradients for large residuals.

Why the supplied losses fail (r = target - prediction):
- mean(r) is affine, hence convex, but unbounded below as predictions increase.
- -mean(r**2) rewards larger errors and is unbounded below.
- mean(r**3) is non-convex over real residuals and unbounded below.
All three are unsuitable here because of what minimization rewards, not
merely because of curvature. Boundedness alone is also insufficient:
(r**2 - 1)**2 is nonnegative but rewards residuals of +/-1 instead of zero.
"""
import torch
from torch import nn

from .exercise import GeneDebuggingExercise
from .support import LOSS_CASES


def regression_loss(
    prediction: torch.Tensor, target: torch.Tensor, case: str,
) -> torch.Tensor:
    """MSE is nonnegative, minimized at matching targets, and differentiable.

    With N elements its prediction gradient is 2 * (prediction - target) / N.
    All case names use the same repair; same-shaped inputs avoid broadcasting.
    """
    if case not in LOSS_CASES:
        raise ValueError(f"Unknown loss case: {case}")
    return (prediction - target).square().mean()


def train_model(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor,
    case: str, epochs: int = 80, lr: float = 0.1,
) -> list[float]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    target = target.reshape(-1, 1)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = regression_loss(prediction, target, case)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class GCRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(dim=-1))


class MotifClassifier(nn.Module):
    def __init__(self, filters: int = 8, kernel_size: int = 3):
        super().__init__()
        self.scanner = nn.Conv1d(4, filters, kernel_size)
        self.head = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.scanner(x)).amax(dim=-1)
        return self.head(features)


class ResidualBlock(nn.Module):
    def __init__(self, features: int, hidden_dim: int):
        super().__init__()
        self.expand = nn.Linear(features, hidden_dim)
        self.project = nn.Linear(hidden_dim, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(torch.relu(self.expand(x)))


class ModelSolution(GeneDebuggingExercise):
    regression_loss = staticmethod(regression_loss)
    train_model = staticmethod(train_model)
    create_gc_regressor = staticmethod(GCRegressor)
    create_motif_classifier = staticmethod(MotifClassifier)
    create_residual_block = staticmethod(ResidualBlock)
