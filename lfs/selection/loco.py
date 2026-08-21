"""LOCO -- Leave One Covariate Out."""

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
    """LOCO importance for each patch."""
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
