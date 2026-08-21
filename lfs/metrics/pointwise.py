"""Per-sample losses.

Why this module exists
----------------------
MinShap's rejection threshold is built from ``sigma_j``, the variance of the
*per-sample* change in loss when patch j is added. That is only coherent if the
value function ``V`` and the variance ``sigma`` come from the same loss.

The original implementation computed ``V`` with an arbitrary sklearn metric
(e.g. ``accuracy_score``) but always computed ``sigma`` from
``(y_true - y_pred) ** 2``. For a classifier whose ``predict`` returns integer
class labels, that squared term treats a 7-predicted-as-1 error as 36x worse
than 7-predicted-as-6, which is not a meaningful loss on a nominal label -- and
it is not the accuracy whose change ``V`` was tracking either.

Defining the loss pointwise fixes both: ``V = mean(loss)`` and
``sigma = var(loss_curr - loss_new) / n`` are guaranteed consistent.
"""
import numpy as np


def squared_error(y_true, y_pred, proba=None):
    """Per-sample squared error. The natural loss for regression."""
    return (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2


def zero_one(y_true, y_pred, proba=None):
    """Per-sample 0/1 misclassification loss. Mean equals 1 - accuracy."""
    return (np.asarray(y_true) != np.asarray(y_pred)).astype(float)


def cross_entropy(y_true, y_pred, proba=None, eps=1e-12):
    """Per-sample negative log-likelihood of the true class.

    Requires class probabilities. Lower variance than 0/1 loss, so it usually
    gives a tighter threshold, but needs a model exposing ``predict_proba``.
    """
    if proba is None:
        raise ValueError("cross_entropy requires predicted probabilities")
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    p_true = proba[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(p_true, eps, 1.0))


#: Names usable from an experiment config.
REGISTRY = {
    "squared_error": squared_error,
    "zero_one": zero_one,
    "cross_entropy": cross_entropy,
}

NEEDS_PROBA = {"cross_entropy"}


def resolve(loss, y=None):
    """Turn a name, callable, or None into a (loss_fn, needs_proba) pair.

    Passing None infers from ``y``: integer-valued targets with few distinct
    values are treated as classification (0/1 loss), otherwise regression
    (squared error).
    """
    if callable(loss):
        return loss, False
    if isinstance(loss, str):
        if loss not in REGISTRY:
            raise KeyError(f"unknown loss {loss!r}; available: {sorted(REGISTRY)}")
        return REGISTRY[loss], loss in NEEDS_PROBA
    if loss is None:
        if y is None:
            raise ValueError("cannot infer loss without y; pass loss= explicitly")
        y = np.asarray(y)
        is_classification = y.dtype.kind in "biu" and len(np.unique(y)) <= max(
            20, int(0.05 * len(y))
        )
        return (zero_one, False) if is_classification else (squared_error, False)
    raise TypeError(f"loss must be a name, callable, or None; got {type(loss)}")


__all__ = ["squared_error", "zero_one", "cross_entropy", "resolve", "REGISTRY"]
