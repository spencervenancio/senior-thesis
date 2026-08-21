"""Synthetic designs with known ground-truth support."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

FEATURE_NAMES = [f"x{i}" for i in range(1, 11)]


@dataclass
class SimulatedDataset:
    """A synthetic design together with its ground-truth support."""

    name: str
    df: pd.DataFrame
    support: np.ndarray
    _local_support: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        default=None, repr=False
    )
    feature_names: Sequence[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    @property
    def X(self) -> np.ndarray:
        return self.df[list(self.feature_names)].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.df["y"].to_numpy()

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def local_support(self, x: np.ndarray) -> np.ndarray:
        """True relevant feature indices at the point ``x``."""
        if self._local_support is None:
            return self.support
        return self._local_support(np.asarray(x))

    def support_mask(self, local_x: Optional[np.ndarray] = None) -> np.ndarray:
        """Boolean mask of the true support, globally or at ``local_x``."""
        idx = self.support if local_x is None else self.local_support(local_x)
        mask = np.zeros(self.n_features, dtype=bool)
        mask[idx] = True
        return mask


def _draw(rng, n, p=10):
    """n x p standard normal design matrix."""
    return rng.standard_normal((n, p))


def _frame(X, y, name, support, local_support=None):
    df = pd.DataFrame(X, columns=FEATURE_NAMES[: X.shape[1]])
    df["y"] = y
    return SimulatedDataset(
        name=name,
        df=df,
        support=np.asarray(support, dtype=int),
        _local_support=local_support,
        feature_names=FEATURE_NAMES[: X.shape[1]],
    )


def xor(n=1_000, noise=0.1, rng=None):
    """y = 1{x1 * x2 < 0} with x1, x2 ~ Unif(-1, 1), observed with noise."""
    rng = np.random.default_rng(rng)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = (x1 * x2) < 0
    X = np.column_stack([x1 + rng.normal(0, noise, n), x2 + rng.normal(0, noise, n)])
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y.astype(int)
    return SimulatedDataset(
        name="xor", df=df, support=np.array([0, 1]), feature_names=["x1", "x2"]
    )


def linear_additive(n=1_000, noise=0.1, rng=None):
    """y = 2x1 + x3 + 9x5 + 9x6 + 3x7 + x10 + eps."""
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    y = (
        2 * X[:, 0] + X[:, 2] + 9 * X[:, 4] + 9 * X[:, 5] + 3 * X[:, 6] + X[:, 9]
        + rng.normal(0, noise, n)
    )
    return _frame(X, y, "linear_additive", [0, 2, 4, 5, 6, 9])


def nonlinear_additive(n=1_000, noise=0.01, rng=None):
    """y = 2x1^2 + x3^3 + 9sin(x5) + 9exp(x6) + 3cos(x7) + |x10| + eps."""
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    y = (
        2 * X[:, 0] ** 2
        + X[:, 2] ** 3
        + 9 * np.sin(X[:, 4])
        + 9 * np.exp(X[:, 5])
        + 3 * np.cos(X[:, 6])
        + np.abs(X[:, 9])
        + rng.normal(0, noise, n)
    )
    return _frame(X, y, "nonlinear_additive", [0, 2, 4, 5, 6, 9])


def conditional_interaction(n=1_000, noise=0.1, rng=None):
    """y = 2 x1x2 1{x3>0} + x4x5 1{x3<0} + 9 x6x7 1{x8>0} + x9x10 1{x8<0} + eps."""
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    y = (
        2 * (X[:, 0] * X[:, 1]) * (X[:, 2] > 0)
        + (X[:, 3] * X[:, 4]) * (X[:, 2] < 0)
        + 9 * (X[:, 5] * X[:, 6]) * (X[:, 7] > 0)
        + (X[:, 8] * X[:, 9]) * (X[:, 7] < 0)
        + rng.normal(0, noise, n)
    )

    def local_support(x):
        active = [2, 7]
        active += [0, 1] if x[2] > 0 else [3, 4]
        active += [5, 6] if x[7] > 0 else [8, 9]
        return np.array(sorted(active))

    return _frame(
        X, y, "conditional_interaction", list(range(10)), local_support=local_support
    )


def logistic(n=1_000, noise=0.1, rng=None):
    """y = sigmoid(x1^2 + x3x4 + 4x6 + 7x8^2 + 2x9^3) + eps."""
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    eta = X[:, 0] ** 2 + X[:, 2] * X[:, 3] + 4 * X[:, 5] + 7 * X[:, 7] ** 2 + 2 * X[:, 8] ** 3
    y = 1 / (1 + np.exp(-eta)) + rng.normal(0, noise, n)
    return _frame(X, y, "logistic", [0, 2, 3, 5, 7, 8])


def logistic_bernoulli(n=1_000, noise=0.0, rng=None):
    """Bernoulli labels drawn from the same linear predictor as :func:`logistic`."""
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    eta = X[:, 0] ** 2 + X[:, 2] * X[:, 3] + 4 * X[:, 5] + 7 * X[:, 7] ** 2 + 2 * X[:, 8] ** 3
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p)
    return _frame(X, y, "logistic_bernoulli", [0, 2, 3, 5, 7, 8])


REGISTRY = {
    "xor": xor,
    "linear_additive": linear_additive,
    "nonlinear_additive": nonlinear_additive,
    "conditional_interaction": conditional_interaction,
    "logistic": logistic,
    "logistic_bernoulli": logistic_bernoulli,
}


def make(name, **kwargs):
    """Build a design by name, for config-driven experiments."""
    if name not in REGISTRY:
        raise KeyError(f"unknown design {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["SimulatedDataset", "REGISTRY", "make", *REGISTRY.keys()]
