"""Plotting. Every function returns (fig, ax) and never calls plt.show()."""
from .importance import (
    plot_importance_bars,
    plot_local_selected,
    plot_patch_importance,
    plot_selected,
    plot_threshold_diagnostic,
)
from .saliency import plot_dropout_thresholds, plot_saliency, plot_saliency_comparison

__all__ = [
    "plot_patch_importance", "plot_selected", "plot_local_selected",
    "plot_threshold_diagnostic", "plot_importance_bars",
    "plot_saliency", "plot_saliency_comparison", "plot_dropout_thresholds",
]
