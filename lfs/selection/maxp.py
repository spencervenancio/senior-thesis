"""Max-p: reject when the *largest* p-value across permutations is small."""
import numpy as np
from scipy.stats import norm

from ._permutation import run_permutations
from .minshap import SelectionResult


def max_p(model, patches, X_train, y_train, loss=None, K=100, alpha=0.05,
          early_stopping_patience=5, n_jobs=-1, local=False, x_S=None,
          k=50, rng=None, two_sided=True, metric=None):
    """Select patches whose maximum p-value across permutations is below alpha."""
    if metric is not None:
        raise TypeError(
            "max_p() no longer takes metric=. Pass loss= instead "
            "(see lfs.metrics.pointwise)."
        )

    stats = run_permutations(
        model, patches, X_train, y_train, loss=loss, K=K,
        early_stopping_patience=early_stopping_patience, n_jobs=n_jobs,
        local=local, x_S=x_S, k=k, rng=rng, desc="Max-p",
    )

    phi, sigma = stats["phi"], stats["sigma"]

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, phi / np.sqrt(np.where(sigma > 0, sigma, 1.0)), 0.0)

    p = 2 * norm.sf(np.abs(z)) if two_sided else norm.sf(z)
    p_max = p.max(axis=0)

    return SelectionResult(
        p_max=p_max,
        rejected=p_max < alpha,
        z=z,
        phi=phi,
        sigma=sigma,
        alpha=alpha,
        K=K,
        n_degenerate=int((sigma == 0).sum()),
        n_eval=stats["n_eval"],
        eval_idx=stats["eval_idx"],
    )


__all__ = ["max_p"]
