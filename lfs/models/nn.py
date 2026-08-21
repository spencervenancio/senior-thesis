"""Neural network estimators, wrapped in skorch for an sklearn-compatible API.

The selection methods refit on masked inputs whose dimensionality is constant
but whose *effective* support changes, and in the simulated-design work the
feature count varies between designs. :class:`AdaptiveNeuralNetClassifier`
therefore rebuilds its module whenever the input shape or class count changes,
so a single estimator object can be handed to ``minshap`` without the caller
tracking architecture.
"""
import numpy as np
import torch.nn as nn
from skorch import NeuralNetClassifier, NeuralNetRegressor


class MLP(nn.Module):
    """Single hidden layer. The default classifier body."""

    def __init__(self, n_input=1, n_output=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_output),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Three hidden layers, for designs the shallow net underfits.

    The conditional-interaction design in particular needs depth: it is a sum of
    gated products, which a single ReLU layer approximates poorly.
    """

    def __init__(self, n_input=1, n_output=1, hidden=256, depth=3, dropout=0.0):
        super().__init__()
        layers = []
        dim = n_input
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InteractionMLP(nn.Module):
    """Augments the input with all pairwise products before the MLP.

    For p inputs this feeds p + p(p-1)/2 features, letting the network represent
    x_i * x_j exactly rather than approximating it. Useful as a well-specified
    reference on the interaction designs -- if a method cannot recover the
    support here, the failure is the method's, not the model's.
    """

    def __init__(self, n_input=10, n_output=1, hidden=128):
        super().__init__()
        self.n_input = n_input
        n_aug = n_input + n_input * (n_input - 1) // 2
        self._iu = None
        self.net = nn.Sequential(
            nn.Linear(n_aug, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_output),
        )

    def forward(self, x):
        import torch

        if self._iu is None or self._iu[0].device != x.device:
            idx = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
            self._iu = (idx[0], idx[1])
        i, j = self._iu
        products = x[:, i] * x[:, j]
        return self.net(torch.cat([x, products], dim=1))


class _AdaptiveMixin:
    """Rebuild the module whenever the input/output shape changes."""

    module_cls = MLP

    def _coerce(self, X, y):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        X, y = self._coerce(X, y)
        n_input, n_output = X.shape[1], self._n_output_for(y)
        if (n_input, n_output) != (getattr(self, "_n_input", None),
                                   getattr(self, "_n_output", None)):
            if hasattr(self, "initialized_"):
                self.initialized_ = False  # architecture changed, must reinitialize
        self._n_input, self._n_output = n_input, n_output
        return super().fit(X, y, **kwargs)

    def initialize_module(self):
        # assigning module_ (trailing underscore) bypasses skorch's name check
        self.module_ = self.module_cls(
            n_input=getattr(self, "_n_input", 1),
            n_output=getattr(self, "_n_output", 2),
        ).to(self.device)
        return self


class AdaptiveNeuralNetClassifier(_AdaptiveMixin, NeuralNetClassifier):
    """Classifier that adapts to input dimension and class count on each fit."""

    def _n_output_for(self, y):
        return len(np.unique(y))

    def _coerce(self, X, y):
        if hasattr(X, "numpy"):
            X, y = X.numpy(), y.numpy()
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        return super().predict_proba(X)


class AdaptiveNeuralNetRegressor(_AdaptiveMixin, NeuralNetRegressor):
    """Regressor counterpart, for the continuous simulated designs."""

    module_cls = DeepMLP

    def _n_output_for(self, y):
        return 1

    def _coerce(self, X, y):
        if hasattr(X, "numpy"):
            X, y = X.numpy(), y.numpy()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return X, y

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        return super().predict(X).ravel()


def neural_net(X_train=None, y_train=None, max_epochs=20, lr=1e-3, batch_size=64,
               device="cpu", verbose=0, **kwargs):
    """Build an adaptive MLP classifier, fitting it when data is supplied.

    Call with no data to get an *unfitted* estimator, which is what the
    selection methods want -- they deepcopy and refit it themselves.
    """
    clf = AdaptiveNeuralNetClassifier(
        module=MLP, max_epochs=max_epochs, lr=lr, batch_size=batch_size,
        device=device, verbose=verbose, **kwargs,
    )
    if X_train is not None:
        clf.fit(X_train, y_train)
    return clf


def neural_net_regressor(X_train=None, y_train=None, max_epochs=50, lr=1e-3,
                         batch_size=64, device="cpu", verbose=0, module=DeepMLP,
                         **kwargs):
    """Build an adaptive MLP regressor for the continuous simulated designs."""
    reg = AdaptiveNeuralNetRegressor(
        module=module, max_epochs=max_epochs, lr=lr, batch_size=batch_size,
        device=device, verbose=verbose, **kwargs,
    )
    reg.module_cls = module
    if X_train is not None:
        reg.fit(X_train, y_train)
    return reg


__all__ = [
    "MLP", "DeepMLP", "InteractionMLP",
    "AdaptiveNeuralNetClassifier", "AdaptiveNeuralNetRegressor",
    "neural_net", "neural_net_regressor",
]
