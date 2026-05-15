"""
Synthetic neural network landscape benchmarks.

Generates synthetic classification datasets (Two Moons, Swiss Roll) and trains
MLPs of configurable depth and width, giving full control over the effective
dimensionality of the parameter space and the difficulty of the loss landscape.
The datasets are generated once per benchmark instance and shared across all
optimizer trials to ensure fair comparison.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler


def _make_two_moons(n_samples: int = 2000, noise: float = 0.2, seed: int = 0):
    """Generate a Two Moons binary classification dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = StandardScaler().fit_transform(X)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


def _make_swiss_roll(n_samples: int = 2000, noise: float = 0.5, seed: int = 0):
    """
    Generate a Swiss Roll dataset projected to 2D for binary classification.
    The label is derived by thresholding the roll parameter t.
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    X2d = X[:, [0, 2]]  # drop the y-axis, keep the roll plane
    X2d = StandardScaler().fit_transform(X2d)
    y = (t > t.mean()).astype(int)
    return (torch.tensor(X2d, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


class MLP(nn.Module):
    """
    Configurable multi-layer perceptron for synthetic classification tasks.
    Uses tanh activations (no BatchNorm) to preserve landscape ruggedness —
    BatchNorm would artificially smooth the loss surface.
    """

    def __init__(self, in_features: int, hidden_sizes: list, out_features: int):
        """
        Args:
            in_features (int): Input dimensionality.
            hidden_sizes (list): List of hidden layer widths.
            out_features (int): Number of output classes.
        """
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NNLandmarkBenchmark:
    """
    A neural network landscape benchmark that wraps a dataset and MLP
    configuration into a single callable object compatible with the suite runner.
    """

    def __init__(self, name: str, dataset: str, hidden_sizes: list,
                 n_samples: int = 2000, noise: float = 0.2,
                 batch_size: int = 256, seed: int = 0):
        """
        Args:
            name (str): Human-readable benchmark name.
            dataset (str): One of 'two_moons' or 'swiss_roll'.
            hidden_sizes (list): Hidden layer widths for the MLP.
            n_samples (int): Number of data points to generate.
            noise (float): Dataset noise level.
            batch_size (int): Mini-batch size for training.
            seed (int): Dataset generation seed (fixed; optimizer seed varies separately).
        """
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size

        if dataset == 'two_moons':
            self.X, self.y = _make_two_moons(n_samples, noise, seed)
        elif dataset == 'swiss_roll':
            self.X, self.y = _make_swiss_roll(n_samples, noise, seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.in_features = self.X.shape[1]
        self.out_features = len(self.y.unique())
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self, seed: int) -> MLP:
        """
        Construct a fresh MLP with a fixed random seed for reproducibility.

        Args:
            seed (int): Seed for weight initialisation.

        Returns:
            MLP: Freshly initialised model.
        """
        torch.manual_seed(seed)
        return MLP(self.in_features, self.hidden_sizes, self.out_features)

    def compute_full_loss_and_acc(self, model: MLP) -> tuple:
        """
        Compute loss and accuracy over the full dataset without gradient tracking.

        Returns:
            tuple: (loss float, accuracy float in [0, 100])
        """
        model.eval()
        with torch.no_grad():
            logits = model(self.X)
            loss = self.criterion(logits, self.y).item()
            preds = logits.argmax(dim=1)
            acc = (preds == self.y).float().mean().item() * 100.0
        model.train()
        return loss, acc

    def hessian_trace_approx(self, model: MLP, n_samples: int = 5) -> float:
        """
        Approximate the trace of the Hessian at the current model parameters
        using Hutchinson's estimator. A large trace indicates a sharp minimum
        (poor generalisation); a small trace indicates a flat minimum.

        Args:
            n_samples (int): Number of random vectors for the Hutchinson estimate.

        Returns:
            float: Estimated Hessian trace.
        """
        params = [p for p in model.parameters() if p.requires_grad]
        trace_estimates = []

        for _ in range(n_samples):
            # Sample a Rademacher random vector
            vs = [torch.randint_like(p, high=2).float() * 2 - 1 for p in params]

            # Compute gradient of loss
            model.zero_grad()
            logits = model(self.X)
            loss = self.criterion(logits, self.y)
            grads = torch.autograd.grad(loss, params, create_graph=True)

            # Compute Hv product via double backprop
            gv = sum((g * v).sum() for g, v in zip(grads, vs))
            hvs = torch.autograd.grad(gv, params, retain_graph=False)

            # Trace estimate = v^T H v
            trace_est = sum((hv * v).sum().item() for hv, v in zip(hvs, vs))
            trace_estimates.append(trace_est)

        model.zero_grad()
        return float(np.mean(trace_estimates))
