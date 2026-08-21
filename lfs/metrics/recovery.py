"""Scoring a selected feature set against a known ground-truth support."""
import numpy as np


def _as_mask(x, n_features):
    x = np.asarray(x)
    if x.dtype == bool:
        if len(x) != n_features:
            raise ValueError(f"mask length {len(x)} != n_features {n_features}")
        return x
    mask = np.zeros(n_features, dtype=bool)
    mask[x.astype(int)] = True
    return mask


def recovery_scores(selected, truth, n_features):
    """Precision / recall / F1 / exact-match of ``selected`` against ``truth``."""
    sel = _as_mask(selected, n_features)
    tru = _as_mask(truth, n_features)

    tp = int(np.sum(sel & tru))
    fp = int(np.sum(sel & ~tru))
    fn = int(np.sum(~sel & tru))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": bool(np.array_equal(sel, tru)),
        "n_selected": int(sel.sum()),
        "n_true": int(tru.sum()),
        "false_positives": fp,
        "false_negatives": fn,
    }


def false_discovery_rate(selected, truth, n_features):
    """Realized FDR: fraction of selected features that are truly null."""
    sel = _as_mask(selected, n_features)
    tru = _as_mask(truth, n_features)
    if sel.sum() == 0:
        return 0.0
    return float(np.sum(sel & ~tru) / sel.sum())


def power(selected, truth, n_features):
    """Fraction of the true support that was recovered."""
    sel = _as_mask(selected, n_features)
    tru = _as_mask(truth, n_features)
    if tru.sum() == 0:
        return 1.0
    return float(np.sum(sel & tru) / tru.sum())


__all__ = ["recovery_scores", "false_discovery_rate", "power"]
