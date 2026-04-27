import numpy as np
import torch.nn as nn
from skorch import NeuralNetClassifier


class _MLP(nn.Module):
    def __init__(self, n_input=1, n_classes=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class AdaptiveNeuralNetClassifier(NeuralNetClassifier):
    """Rebuilds the module to match the input dimension on each fit()."""

    def fit(self, X, y, **kwargs):
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        n_input, n_classes = X.shape[1], len(np.unique(y))
        if n_input != getattr(self, '_n_input', None) or n_classes != getattr(self, '_n_classes', None):
            if hasattr(self, 'initialized_'):
                self.initialized_ = False  # architecture changed — must reinitialize
        self._n_input = n_input
        self._n_classes = n_classes
        return super().fit(X, y, **kwargs)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        return super().predict_proba(X)

    def initialize_module(self):
        # module_ (trailing underscore) bypasses skorch's torch-component name check
        self.module_ = _MLP(
            n_input=getattr(self, '_n_input', 1),
            n_classes=getattr(self, '_n_classes', 2),
        ).to(self.device)
        return self


def neural_net(X_train, y_train):
    clf = AdaptiveNeuralNetClassifier(
        module=_MLP,
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        device='cpu',
        verbose=0,
    )
    clf.fit(X_train, y_train)
    return clf
