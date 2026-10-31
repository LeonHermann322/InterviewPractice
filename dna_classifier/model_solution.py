"""Reference solution for the PyTorch interview practice exercise."""
from collections.abc import Sequence

import torch
from torch import nn

from .exercise import DNAExercise

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(seq: str) -> torch.Tensor:
    """Convert one DNA string into a flat one-hot feature vector.

    Steps:
    1. Reject non-string input and bases outside uppercase A/C/G/T.
    2. Allocate one four-value row per base, initially all zeros.
    3. Set the position for each base to 1 using ``BASE_TO_IDX``.
    4. Flatten the rows so each base contributes four adjacent values.
    """
    if not isinstance(seq, str):
        raise TypeError("seq must be a string")
    if any(base not in BASE_TO_IDX for base in seq):
        raise ValueError("Sequence may contain only A, C, G, T")
    encoded = torch.zeros((len(seq), 4), dtype=torch.float32)
    for i, base in enumerate(seq):
        encoded[i, BASE_TO_IDX[base]] = 1.0
    return encoded.flatten()


def encode_sequences(sequences: Sequence[str]) -> torch.Tensor:
    """Encode a batch of equal-length DNA strings as a 2D tensor.

    First reject an empty batch or strings with different lengths. Then encode
    each string with :func:`encode_sequence` and stack the feature vectors into
    a tensor shaped ``(batch_size, sequence_length * 4)``.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError("All sequences must have equal length")
    return torch.stack([encode_sequence(seq) for seq in sequences])


class DNAClassifier(nn.Module):
    """Map flattened DNA features to one raw binary-classification logit.

    The first linear layer mixes the encoded base features, ReLU adds a
    nonlinearity, and the final linear layer produces one unnormalized score
    (logit) per sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        """Create the layers after checking that their dimensions are positive."""
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layers in order: input projection, ReLU, then logit projection.

        For a batch shaped ``(N, input_dim)``, return raw scores shaped
        ``(N, 1)``. Do not apply sigmoid here; the training loss consumes logits.
        """
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1,
) -> list[float]:
    """Train the classifier and return one scalar loss for each epoch.

    Steps:
    1. Validate the epoch count and learning rate, reshape labels to ``(N, 1)``,
       and check that their count matches the input batch.
    2. Set up binary cross-entropy for logits and an SGD optimizer; enable
       training mode.
    3. For each epoch, clear old gradients, compute logits, check their shape,
       calculate a finite loss, backpropagate, and update the parameters.
    4. Save the detached loss value for reporting and return all epoch losses.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    y = y.to(dtype=torch.float32).reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of examples")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        if logits.shape != y.shape:
            raise ValueError(f"Expected logits shape {tuple(y.shape)}, got {tuple(logits.shape)}")
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return losses


def predict_labels(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Convert model logits into integer class labels without tracking gradients.

    Remember the incoming train/eval state, switch to evaluation mode, and
    compute sigmoid probabilities inside ``no_grad``. Threshold at 0.5 to get
    labels, then restore training mode if it was enabled on entry.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(X)
        predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    if was_training:
        model.train()
    return predictions


class ModelSolution(DNAExercise):
    """Provide the reference functions as implementations of the exercise contract.

    The static method bindings let the shared test suite call the helpers
    through an instance, while the model factory points to ``DNAClassifier``.
    """

    encode_sequence = staticmethod(encode_sequence)
    encode_sequences = staticmethod(encode_sequences)
    create_model = staticmethod(DNAClassifier)
    train_model = staticmethod(train_model)
    predict_labels = staticmethod(predict_labels)
