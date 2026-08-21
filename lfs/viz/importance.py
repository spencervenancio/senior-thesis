"""Plots for patch importance and selected supports.

Every function returns ``(fig, ax)`` and does **not** call ``plt.show()``, so
the same call works in a notebook (where inline display happens anyway) and in
a headless experiment run that needs ``fig.savefig(...)``.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..data.patches import patches_to_pixels


def plot_patch_importance(importances, patches, img_size=28, ax=None,
                          cmap="viridis", label="importance", title=None):
    """Heatmap of per-patch importance mapped back onto the pixel grid."""
    grid = patches_to_pixels(importances, patches, img_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    im = ax.imshow(grid, cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title or f"{len(patches)} patches")
    ax.axis("off")
    return fig, ax


def plot_selected(result, patches, img_size=28, ax=None, title=None):
    """Binary map of which patches were selected."""
    rejected = np.asarray(result["rejected"])
    grid = patches_to_pixels(rejected.astype(float), patches, img_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    ax.imshow(grid, cmap="binary", vmin=0, vmax=1)
    ax.set_title(title or f"{int(rejected.sum())} / {len(patches)} patches selected")
    ax.axis("off")
    return fig, ax


def plot_local_selected(result, patches, x_S, img_size=28, ax=None, title=None,
                        alpha=0.5):
    """Query image with its selected patches overlaid in red."""
    rejected = np.asarray(result["rejected"])
    image = np.asarray(x_S).reshape(img_size, img_size)
    mask = patches_to_pixels(rejected.astype(float), patches, img_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    ax.imshow(image, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="Reds", alpha=alpha,
              vmin=0, vmax=1)
    ax.set_title(title or f"{int(rejected.sum())} / {len(patches)} patches selected")
    ax.axis("off")
    return fig, ax


def plot_threshold_diagnostic(result, ax=None, top_n=None):
    """phi_min against its threshold t_j, per patch -- the MinShap decision plot.

    Shows *how close* each call was, which a binary selection map hides. Patches
    sitting just under the line are the ones whose fate would flip with a
    different alpha or a larger K.
    """
    phi_min = np.asarray(result["phi_min"])
    t = np.asarray(result["t_j"])
    order = np.argsort(phi_min - t)[::-1]
    if top_n:
        order = order[:top_n]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    pos = np.arange(len(order))
    ax.bar(pos, phi_min[order], color=["tab:green" if phi_min[i] >= t[i]
                                       else "tab:gray" for i in order],
           label="phi_min")
    ax.plot(pos, t[order], "r.-", lw=1, ms=4, label="threshold $t_j$")
    ax.set_xlabel("patch (sorted by margin)")
    ax.set_ylabel("minimum loss reduction")
    ax.set_title(f"MinShap decisions (alpha={result.get('alpha', '?')})")
    ax.legend()
    ax.margins(x=0.01)
    return fig, ax


def plot_importance_bars(importances, feature_names=None, true_support=None, ax=None,
                         title="feature importance"):
    """Bar chart for tabular designs, optionally coloring the true support.

    Pass ``true_support`` from a :class:`lfs.data.simulated.SimulatedDataset` to
    see at a glance whether the method found the right features.
    """
    importances = np.asarray(importances)
    n = len(importances)
    names = feature_names if feature_names is not None else [f"x{i+1}" for i in range(n)]

    colors = ["tab:gray"] * n
    if true_support is not None:
        for j in np.asarray(true_support, dtype=int):
            colors[j] = "tab:blue"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    ax.bar(np.arange(n), importances, color=colors)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title + ("  (blue = true support)" if true_support is not None else ""))
    ax.set_ylabel("importance")
    return fig, ax


__all__ = [
    "plot_patch_importance", "plot_selected", "plot_local_selected",
    "plot_threshold_diagnostic", "plot_importance_bars",
]
