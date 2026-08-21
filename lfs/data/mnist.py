"""MNIST loading, cached under the repo-level data/ directory."""
import numpy as np

from ..paths import DATA_DIR


def load_mnist(as_tensor=False, n_train=None, n_test=None, flatten=True):
    """Load MNIST as (X_train, X_test, y_train, y_test), scaled to [0, 1]."""
    from torchvision import datasets

    train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)

    X_train, y_train = train.data, train.targets
    X_test, y_test = test.data, test.targets

    if n_train is not None:
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    if n_test is not None:
        X_test, y_test = X_test[:n_test], y_test[:n_test]

    shape = (-1, 784) if flatten else (-1, 1, 28, 28)

    if as_tensor:
        X_train = X_train.reshape(shape).float() / 255.0
        X_test = X_test.reshape(shape).float() / 255.0
    else:
        X_train = X_train.numpy().reshape(shape).astype(np.float64) / 255.0
        X_test = X_test.numpy().reshape(shape).astype(np.float64) / 255.0
        y_train = y_train.numpy()
        y_test = y_test.numpy()

    return X_train, X_test, y_train, y_test


__all__ = ["load_mnist"]
