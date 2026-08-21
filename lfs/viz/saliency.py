"""Plots for gradient attributions."""
import matplotlib.pyplot as plt
import numpy as np

from ..selection.saliency import dropout_curve


def plot_saliency(attribution, image=None, img_size=28, ax=None, cmap="hot",
                  alpha=0.5, title=None):
    """Attribution heatmap, optionally overlaid on the source image.

    Parameters
    ----------
    attribution : np.ndarray, shape (n_features,)
    image : np.ndarray, optional
        Source image to show underneath. Accepts torch tensors.
    """
    attr = np.asarray(attribution.detach().cpu() if hasattr(attribution, "detach")
                      else attribution).reshape(img_size, img_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if image is not None:
        img = np.asarray(image.detach().cpu() if hasattr(image, "detach")
                         else image).reshape(img_size, img_size)
        ax.imshow(img, cmap="gray")
        ax.imshow(attr, cmap=cmap, alpha=alpha)
    else:
        ax.imshow(attr, cmap=cmap)

    if title:
        ax.set_title(title)
    ax.axis("off")
    return fig, ax


def plot_saliency_comparison(attributions, image=None, img_size=28, titles=None):
    """Side-by-side attributions from several methods on the same input."""
    n = len(attributions)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    axes = np.atleast_1d(axes)
    names = titles or [f"method {i}" for i in range(n)]
    for ax, attr, name in zip(axes, attributions, names):
        plot_saliency(attr, image=image, img_size=img_size, ax=ax, title=name)
    fig.tight_layout()
    return fig, axes


def plot_dropout_thresholds(attribution, true_support=None, steps=20, ax=None,
                            title=None):
    """Number of features retained as the attribution threshold sweeps upward.

    A horizontal line marks the true support size when known: the useful
    diagnostic is whether the curve *plateaus* there, meaning some threshold
    recovers the right support, or passes straight through, meaning no cutoff
    on this attribution does.
    """
    thresholds, counts = dropout_curve(attribution, steps=steps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure

    ax.step(thresholds, counts, where="post", lw=1.8)
    if true_support is not None:
        ax.axhline(len(np.atleast_1d(true_support)), color="tab:red", ls="--",
                   label=f"|S*| = {len(np.atleast_1d(true_support))}")
        ax.legend()
    ax.set_xlabel("attribution threshold")
    ax.set_ylabel("features retained")
    ax.set_title(title or "support size vs. threshold")
    return fig, ax


__all__ = ["plot_saliency", "plot_saliency_comparison", "plot_dropout_thresholds"]
