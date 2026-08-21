"""LOCO -- Leave One Covariate Out.

Zero out one patch in both train and test, refit, and measure how much worse
the model gets. Unlike MinShap this makes no attempt at a conditional
statement: a patch with a highly correlated partner can look unimportant
because the partner absorbs its signal. That is precisely the weakness the
permutation methods are meant to address, so LOCO is kept as the baseline.

Importance is reported on the same scale as MinShap: **increase in loss**, so
larger means more important, for every loss.
"""

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..metrics import pointwise
from ..metrics.local import Neighborhood
from ._permutation import _evaluate, _prepare_estimator, is_skorch_model


def _loco_one_patch(patch_idx, model, X_train, X_test, y_train, y_test,
                    full_loss, loss_fn, needs_proba, eval_idx,
                    early_stopping_patience, is_skorch):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[:, patch_idx] = 0
    X_te[:, patch_idx] = 0

    est = _prepare_estimator(model, is_skorch, early_stopping_patience)
    est.fit(X_tr, y_train)
    losses = _evaluate(est, X_te, y_test, loss_fn, needs_proba)

    sel = slice(None) if eval_idx is None else eval_idx
    return float(np.mean(losses[sel]) - full_loss)


def loco(model, patches, X_train, X_test, y_train, y_test, loss=None,
         early_stopping_patience=5, n_jobs=-1, local=False, x_S=None, k=50):
    """LOCO importance for each patch.

    Parameters
    ----------
    model : sklearn estimator or skorch NeuralNet
        Refit once per patch; passed by deepcopy, never mutated.
    patches : list of np.ndarray
        Index arrays from :func:`lfs.data.patches.get_patches`.
    X_train, X_test, y_train, y_test : np.ndarray
    loss : str, callable, or None
        Per-sample loss; None infers from ``y_train``. See
        :mod:`lfs.metrics.pointwise`.
    n_jobs : int
        joblib parallelism. -1 = all cores, 1 = serial.
    local : bool
        Score on a fixed k-NN neighborhood of ``x_S`` within the *test* set.
    x_S : np.ndarray
        Query point, required when ``local=True``.
    k : int
        Neighborhood size.

    Returns
    -------
    np.ndarray, shape (len(patches),)
        Increase in loss when the patch is removed. Larger = more important.
        Values near zero (or negative) mean the model recovers without it.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    is_skorch = is_skorch_model(model)
    if is_skorch:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    loss_fn, needs_proba = pointwise.resolve(loss, y_train)
    eval_idx = Neighborhood(X_test, x_S, k).indices if local else None

    baseline = _evaluate(model, X_test, y_test, loss_fn, needs_proba)
    sel = slice(None) if eval_idx is None else eval_idx
    full_loss = float(np.mean(baseline[sel]))

    prefer = "threads" if is_skorch else "processes"
    results = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
            delayed(_loco_one_patch)(
                patch, model, X_train, X_test, y_train, y_test, full_loss,
                loss_fn, needs_proba, eval_idx, early_stopping_patience, is_skorch,
            )
            for patch in patches
        ),
        total=len(patches), desc="LOCO",
    ))
    return np.array(results)


__all__ = ["loco"]
