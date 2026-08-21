"""Shared machinery for permutation-based Shapley feature selection.

MinShap and Max-p differ only in how they turn the per-permutation statistics
``(phi, sigma)`` into a rejection decision, so the expensive part -- refitting
the model along a random patch ordering -- lives here and is run once.

Convention
----------
``phi[j]`` is the **reduction in loss** obtained by adding patch j to the active
set at its position in the permutation. Larger is more important, always,
regardless of which loss is in play. This replaces the old ``higher_is_better``
flag, which had to be interpreted differently in local and global mode.
"""
import copy

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..metrics import pointwise
from ..metrics.local import Neighborhood


def is_skorch_model(model):
    """True if ``model`` is a skorch NeuralNet (needs threads, not processes)."""
    try:
        from skorch import NeuralNet

        return isinstance(model, NeuralNet)
    except ImportError:
        return False


def _prepare_estimator(model, is_skorch, early_stopping_patience):
    m = copy.deepcopy(model)
    if is_skorch:
        from skorch.callbacks import EarlyStopping

        m.warm_start = True
        m.callbacks = [EarlyStopping(patience=early_stopping_patience)]
    return m


def _evaluate(model, X, y, loss_fn, needs_proba):
    """Return per-sample losses of ``model`` on ``(X, y)``."""
    preds = model.predict(X)
    proba = model.predict_proba(X) if needs_proba else None
    return np.asarray(loss_fn(y, preds, proba), dtype=float)


def _mask_to(X, patches, active):
    """Copy of X with every feature outside the active patches set to zero."""
    X_new = np.zeros_like(X)
    for idx in active:
        X_new[:, patches[idx]] = X[:, patches[idx]]
    return X_new


def permutation_pass(model, patches, X, y, loss_fn, needs_proba, eval_idx,
                     null_losses, rng, is_skorch, early_stopping_patience,
                     show_patch_bar=False):
    """One random-order pass, accumulating marginal contributions.

    Parameters
    ----------
    eval_idx : np.ndarray or None
        Restrict evaluation to these row indices (local mode). None uses all rows.
    null_losses : np.ndarray
        Per-sample losses of the all-zeros model, computed once by the caller.
    rng : np.random.Generator
        This permutation's own stream -- see :func:`lfs.seed.spawn`.

    Returns
    -------
    phi, sigma : np.ndarray, each shape (len(patches),)
    """
    p = len(patches)
    est = _prepare_estimator(model, is_skorch, early_stopping_patience)

    order = rng.permutation(p)
    losses_curr = null_losses.copy()
    sel = slice(None) if eval_idx is None else eval_idx
    n_eval = len(y) if eval_idx is None else len(eval_idx)

    phi = np.zeros(p)
    sigma = np.zeros(p)

    for pos in tqdm(range(p), desc="patches", leave=False, disable=not show_patch_bar):
        patch_j = order[pos]
        X_new = _mask_to(X, patches, order[: pos + 1])

        fitted = est.fit(X_new, y)
        losses_new = _evaluate(fitted, X_new, y, loss_fn, needs_proba)

        delta = losses_curr[sel] - losses_new[sel]
        phi[patch_j] = float(np.mean(delta))
        # Variance of the mean reduction; ddof=1 needs at least 2 eval points.
        sigma[patch_j] = float(np.var(delta, ddof=1) / n_eval) if n_eval > 1 else 0.0

        losses_curr = losses_new

    return phi, sigma


def run_permutations(model, patches, X, y, loss=None, K=100,
                     early_stopping_patience=5, n_jobs=-1,
                     local=False, x_S=None, k=50, rng=None, desc="permutations"):
    """Run K independent permutation passes.

    Returns
    -------
    dict with 'phi' and 'sigma', each shape (K, len(patches)), plus 'eval_idx'
    and 'n_eval' describing the evaluation set.
    """
    from ..seed import spawn

    X = np.asarray(X)
    y = np.asarray(y)

    if len(patches) == 0:
        raise ValueError("patches is empty")

    is_skorch = is_skorch_model(model)
    if is_skorch:
        X = X.astype(np.float32)

    loss_fn, needs_proba = pointwise.resolve(loss, y)
    if needs_proba and not hasattr(model, "predict_proba"):
        raise ValueError(
            f"loss requires predict_proba but {type(model).__name__} does not provide it"
        )

    eval_idx = Neighborhood(X, x_S, k).indices if local else None
    if local and len(eval_idx) < 2:
        raise ValueError(
            f"local neighborhood has {len(eval_idx)} point(s); need >= 2 for a "
            "variance estimate -- increase k"
        )

    # Null model: everything zeroed. Fit once and share across permutations.
    X_null = np.zeros_like(X)
    null_model = _prepare_estimator(model, is_skorch, early_stopping_patience)
    null_model.fit(X_null, y)
    null_losses = _evaluate(null_model, X_null, y, loss_fn, needs_proba)

    streams = spawn(rng if rng is not None else np.random.default_rng(), K)

    def one(stream, show_bar):
        return permutation_pass(
            model, patches, X, y, loss_fn, needs_proba, eval_idx, null_losses,
            stream, is_skorch, early_stopping_patience, show_patch_bar=show_bar,
        )

    if n_jobs == 1:
        results = [one(s, True) for s in tqdm(streams, desc=desc)]
    else:
        # threads avoid torch multiprocessing issues; processes are fine for sklearn
        prefer = "threads" if is_skorch else "processes"
        results = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
                delayed(one)(s, False) for s in streams
            ),
            total=K, desc=desc,
        ))

    return {
        "phi": np.array([r[0] for r in results]),
        "sigma": np.array([r[1] for r in results]),
        "eval_idx": eval_idx,
        "n_eval": len(y) if eval_idx is None else len(eval_idx),
    }


__all__ = ["run_permutations", "permutation_pass", "is_skorch_model"]
