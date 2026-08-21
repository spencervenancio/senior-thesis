"""MinShap: selection via the minimum marginal contribution across permutations."""
import numpy as np

from ._permutation import run_permutations


class SelectionResult(dict):
    """Selection output. A dict, plus convenience accessors."""

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
    """Select patches whose minimum marginal contribution clears the threshold."""
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
