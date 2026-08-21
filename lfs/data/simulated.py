"""Synthetic designs with known ground-truth support.

Each generator returns a :class:`SimulatedDataset` carrying the data *and* the
true relevant feature set, so recovery can be scored automatically instead of
comparing against an ``S_star`` dict maintained by hand in a notebook.

Two notions of support are distinguished, and the difference is the whole point
of local feature selection:

``support``
    The *global* support -- features that matter somewhere in the design.
``local_support(x)``
    The support *at a point*. For additive models this equals the global
    support for every x. For :func:`conditional_interaction` it genuinely
    varies with x, since the gating variables switch which interaction is live.

Generators are seeded through an explicit ``rng`` so results are reproducible;
see :mod:`lfs.seed`.
"""
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
    _true_gradient: Optional[Callable[[np.ndarray], np.ndarray]] = field(
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
        """True relevant feature indices at the point ``x``.

        Falls back to the global support for designs whose support is constant.
        """
        if self._local_support is None:
            return self.support
        return self._local_support(np.asarray(x))

    def true_gradient(self, x: np.ndarray) -> np.ndarray:
        """Analytic grad f(x) of the noiseless generating function.

        The saliency ceiling control: scoring this against local_support tells
        you what a *perfect* model would recover, so a shortfall is a property
        of the estimand rather than of any fitted model.
        """
        if self._true_gradient is None:
            raise NotImplementedError(
                f"design {self.name!r} carries no analytic gradient"
            )
        return self._true_gradient(np.asarray(x))

    @property
    def has_true_gradient(self) -> bool:
        return self._true_gradient is not None

    def support_mask(self, local_x: Optional[np.ndarray] = None) -> np.ndarray:
        """Boolean mask of the true support, globally or at ``local_x``."""
        idx = self.support if local_x is None else self.local_support(local_x)
        mask = np.zeros(self.n_features, dtype=bool)
        mask[idx] = True
        return mask


def _sigmoid(z):
    """Numerically stable logistic, so large |z|/tau does not overflow."""
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def _draw(rng, n, p=10):
    """n x p standard normal design matrix."""
    return rng.standard_normal((n, p))


def _frame(X, y, name, support, local_support=None, true_gradient=None):
    df = pd.DataFrame(X, columns=FEATURE_NAMES[: X.shape[1]])
    df["y"] = y
    return SimulatedDataset(
        name=name,
        df=df,
        support=np.asarray(support, dtype=int),
        _local_support=local_support,
        _true_gradient=true_gradient,
        feature_names=FEATURE_NAMES[: X.shape[1]],
    )


def xor(n=1_000, noise=0.1, rng=None):
    """y = 1{x1 * x2 < 0} with x1, x2 ~ Unif(-1, 1), observed with noise.

    Both features are individually uninformative and only matter jointly --
    the standard failure case for marginal screening.
    """
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
    """y = 2 x1x2 1{x3>0} + x4x5 1{x3<0} + 9 x6x7 1{x8>0} + x9x10 1{x8<0} + eps.

    The design where local and global support genuinely differ: x3 and x8 gate
    which pair is active, so at any given x only 2 of the 4 interacting pairs
    contribute. Any method reporting a single global ranking must blur these
    together; a local method should recover the active pairs for that x.
    """
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
        # gates x3 (index 2) and x8 (index 7) are always relevant
        active = [2, 7]
        active += [0, 1] if x[2] > 0 else [3, 4]
        active += [5, 6] if x[7] > 0 else [8, 9]
        return np.array(sorted(active))

    def true_gradient(x):
        g = np.zeros(10)
        if x[2] > 0:
            g[0], g[1] = 2 * x[1], 2 * x[0]
        else:
            g[3], g[4] = x[4], x[3]
        if x[7] > 0:
            g[5], g[6] = 9 * x[6], 9 * x[5]
        else:
            g[8], g[9] = x[9], x[8]
        return g

    return _frame(
        X, y, "conditional_interaction", list(range(10)),
        local_support=local_support, true_gradient=true_gradient,
    )


def smooth_conditional_interaction(n=1_000, noise=0.1, tau=0.1, rng=None):
    """conditional_interaction with the indicator gates replaced by sigmoids.

    The hard-gate design is unusable as a saliency benchmark: an indicator is
    flat off a measure-zero set, so d y / d x3 = 0 almost everywhere and the two
    gating features sit in local_support while carrying no gradient signal at
    all. Two of six target features are then unrecoverable by *any* gradient
    method at *any* model complexity, which caps the achievable F1 near 0.75 for
    reasons that have nothing to do with the model.

    Replacing 1{x3>0} with sigmoid(x3/tau) makes the gate differentiable:
    d y / d x3 is proportional to sigmoid'(x3/tau)/tau, peaking at the gate
    boundary and decaying with |x3|. The ceiling becomes attainable and tau is a
    difficulty knob -- small tau approaches the hard design, large tau blurs
    which branch is live until local_support stops being well defined.
    """
    rng = np.random.default_rng(rng)
    X = _draw(rng, n)
    g3 = _sigmoid(X[:, 2] / tau)
    g8 = _sigmoid(X[:, 7] / tau)
    y = (
        2 * (X[:, 0] * X[:, 1]) * g3
        + (X[:, 3] * X[:, 4]) * (1 - g3)
        + 9 * (X[:, 5] * X[:, 6]) * g8
        + (X[:, 8] * X[:, 9]) * (1 - g8)
        + rng.normal(0, noise, n)
    )

    def local_support(x):
        active = [2, 7]
        active += [0, 1] if x[2] > 0 else [3, 4]
        active += [5, 6] if x[7] > 0 else [8, 9]
        return np.array(sorted(active))

    def true_gradient(x):
        s3, s8 = _sigmoid(x[2] / tau), _sigmoid(x[7] / tau)
        g = np.zeros(10)
        g[0], g[1] = 2 * x[1] * s3, 2 * x[0] * s3
        g[3], g[4] = x[4] * (1 - s3), x[3] * (1 - s3)
        g[5], g[6] = 9 * x[6] * s8, 9 * x[5] * s8
        g[8], g[9] = x[9] * (1 - s8), x[8] * (1 - s8)
        g[2] = (2 * x[0] * x[1] - x[3] * x[4]) * s3 * (1 - s3) / tau
        g[7] = (9 * x[5] * x[6] - x[8] * x[9]) * s8 * (1 - s8) / tau
        return g

    return _frame(
        X, y, "smooth_conditional_interaction", list(range(10)),
        local_support=local_support, true_gradient=true_gradient,
    )


def logistic(n=1_000, noise=0.1, rng=None):
    """y = sigmoid(x1^2 + x3x4 + 4x6 + 7x8^2 + 2x9^3) + eps.

    Note this is a *continuous* response in (0, 1) plus noise, not a Bernoulli
    draw -- treat it as regression. Use :func:`logistic_bernoulli` if you want
    actual binary labels.
    """
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


#: Lookup used by the experiment runner so configs can name a design as a string.
REGISTRY = {
    "xor": xor,
    "linear_additive": linear_additive,
    "nonlinear_additive": nonlinear_additive,
    "conditional_interaction": conditional_interaction,
    "smooth_conditional_interaction": smooth_conditional_interaction,
    "logistic": logistic,
    "logistic_bernoulli": logistic_bernoulli,
}


def make(name, **kwargs):
    """Build a design by name, for config-driven experiments."""
    if name not in REGISTRY:
        raise KeyError(f"unknown design {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["SimulatedDataset", "REGISTRY", "make", *REGISTRY.keys()]
