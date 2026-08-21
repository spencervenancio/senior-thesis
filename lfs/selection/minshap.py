"""MinShap: selection via the minimum marginal contribution across permutations.

For each patch j, take the smallest loss-reduction ``phi_j`` observed over K
random orderings, and reject the null "patch j is irrelevant" when that minimum
exceeds a threshold ``t_j`` calibrated from the per-permutation variance.

Taking the *minimum* is what makes the statement conditional: a patch only
survives if it helps regardless of which other patches are already present, so
it cannot be carried by a correlated companion.
"""
import numpy as np

from ._permutation import run_permutations


class SelectionResult(dict):
    """Selection output. A dict, plus convenience accessors.

    Kept dict-shaped so existing ``result['rejected']`` code keeps working.
    """

    @property
    def selected(self):
        """Indices of the selected patches."""
        return np.flatnonzero(self["rejected"])

    @property
    def n_selected(self):
        return int(self["rejected"].sum())

    def __repr__(self):
        keys = ", ".join(sorted(self))
        return f"<SelectionResult {self.n_selected} selected; keys: {keys}>"


def minshap(model, patches, X_train, y_train, loss=None, K=100, alpha=0.05,
            early_stopping_patience=5, n_jobs=-1, local=False, x_S=None,
            k=50, rng=None, metric=None, higher_is_better=None):
    """Select patches whose minimum marginal contribution clears the threshold.

    Parameters
    ----------
    model : sklearn estimator or skorch NeuralNet
        Refit many times; passed by deepcopy, never mutated.
    patches : list of np.ndarray
        Index arrays from :func:`lfs.data.patches.get_patches`, or
        :func:`lfs.data.patches.single_features` for ungrouped features.
    X_train, y_train : np.ndarray
    loss : str, callable, or None
        Per-sample loss -- ``'squared_error'``, ``'zero_one'``,
        ``'cross_entropy'``, or a callable ``(y_true, y_pred, proba) -> array``.
        None infers from ``y_train``. See :mod:`lfs.metrics.pointwise`.
    K : int
        Number of random permutations. The minimum over K is conservative and
        gets stricter as K grows, so K trades power for confidence.
    alpha : float
        Significance level entering the threshold.
    n_jobs : int
        joblib parallelism. -1 = all cores, 1 = serial (needed for readable
        progress bars and for debugging).
    local : bool
        Evaluate on a fixed k-NN neighborhood of ``x_S`` instead of the full
        training sample. The neighborhood is built once from all features and
        held constant across every refit.
    x_S : np.ndarray
        Query point, required when ``local=True``.
    k : int
        Neighborhood size.
    rng : int, Generator, or None
        Seed or generator. Each permutation gets an independent child stream.
    metric, higher_is_better : deprecated
        Present so old calls fail loudly rather than silently changing meaning.

    Returns
    -------
    SelectionResult with keys:
        phi_min   -- (n_patches,) minimum loss reduction across permutations
        t_j       -- (n_patches,) rejection threshold
        rejected  -- (n_patches,) bool, phi_min >= t_j
        phi, sigma -- (K, n_patches) raw per-permutation statistics
    """
    if metric is not None or higher_is_better is not None:
        raise TypeError(
            "minshap() no longer takes metric=/higher_is_better=. Pass loss= "
            "instead (e.g. loss='zero_one' for classification, "
            "loss='squared_error' for regression). See lfs.metrics.pointwise "
            "for why the value function and its variance must share a loss."
        )

    stats = run_permutations(
        model, patches, X_train, y_train, loss=loss, K=K,
        early_stopping_patience=early_stopping_patience, n_jobs=n_jobs,
        local=local, x_S=x_S, k=k, rng=rng, desc="MinShap",
    )

    phi, sigma = stats["phi"], stats["sigma"]
    phi_min = phi.min(axis=0)
    t = np.sqrt(-2 * np.log(alpha) * sigma.mean(axis=0))

    return SelectionResult(
        phi_min=phi_min,
        t_j=t,
        rejected=phi_min >= t,
        phi=phi,
        sigma=sigma,
        alpha=alpha,
        K=K,
        n_eval=stats["n_eval"],
        eval_idx=stats["eval_idx"],
    )


__all__ = ["minshap", "SelectionResult"]
