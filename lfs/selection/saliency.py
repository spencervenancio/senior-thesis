"""Gradient-based attribution, wrapped for comparison against MinShap/LOCO.

These are the cheap local baselines: one backward pass gives a per-feature
score at a single point, with no refitting and no statistical guarantee. The
research question they exist to sharpen is how much is lost by that trade.

All functions return plain numpy arrays so downstream scoring
(:mod:`lfs.metrics.recovery`) is identical across methods.
"""
import numpy as np

METHODS = ("saliency", "input_x_gradient", "integrated_gradients")


def _unwrap(model):
    """Get the torch module out of a skorch wrapper, if needed."""
    return getattr(model, "module_", model)


def attribute(model, x, target=None, method="saliency", n_steps=50, abs_value=True):
    """Attribution for a single input.

    Parameters
    ----------
    model : torch.nn.Module or skorch NeuralNet
        skorch wrappers are unwrapped automatically.
    x : torch.Tensor or np.ndarray, shape (n_features,) or (1, n_features)
    target : int, optional
        Output index to attribute. Defaults to the model's predicted class.
    method : {'saliency', 'input_x_gradient', 'integrated_gradients'}
    n_steps : int
        Riemann steps, integrated gradients only.
    abs_value : bool
        Return |attribution|. Magnitude is what feature *selection* needs;
        set False to keep signed direction for visualization.

    Returns
    -------
    np.ndarray, shape (n_features,)
    """
    import torch
    from captum.attr import InputXGradient, IntegratedGradients, Saliency

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    net = _unwrap(model)
    net.eval()

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.asarray(x), dtype=torch.float32)
    x = x.reshape(1, -1).float().clone().requires_grad_(True)

    if target is None:
        with torch.no_grad():
            out = net(x)
        target = int(out.argmax(dim=1)) if out.ndim > 1 and out.shape[1] > 1 else 0

    attributor = {
        "saliency": Saliency,
        "input_x_gradient": InputXGradient,
        "integrated_gradients": IntegratedGradients,
    }[method](net)

    kwargs = {"n_steps": n_steps} if method == "integrated_gradients" else {}
    attr = attributor.attribute(x, target=target, **kwargs)

    attr = attr.squeeze().detach().cpu().numpy()
    return np.abs(attr) if abs_value else attr


def attribute_batch(model, X, targets=None, method="saliency", **kwargs):
    """Attributions for every row of ``X``. Returns shape (n_samples, n_features)."""
    X = np.asarray(X) if not hasattr(X, "detach") else X
    out = []
    for i in range(len(X)):
        t = None if targets is None else int(targets[i])
        out.append(attribute(model, X[i], target=t, method=method, **kwargs))
    return np.vstack(out)


def select_top_k(attribution, k, rng=None):
    """Boolean mask of the k largest attributions -- the fixed-budget rule.

    Ties are broken uniformly at random. np.argsort breaks them by feature
    index, which biases selection toward high-index features and manufactures
    apparent recall whenever many attributions are exactly equal -- the true
    gradient of an indicator gate is exactly zero, so a design with hard gates
    hits this immediately. Pass ``rng`` for a reproducible tie-break.
    """
    attribution = np.asarray(attribution)
    rng = np.random.default_rng(rng)
    order = np.lexsort((rng.random(len(attribution)), attribution))
    mask = np.zeros(len(attribution), dtype=bool)
    mask[order[-k:]] = True
    return mask


def select_threshold(attribution, frac=0.05):
    """Boolean mask of attributions above ``frac`` of the maximum.

    Note this has no error control: the cutoff is a heuristic on an arbitrary
    scale, which is exactly the gap MinShap's calibrated threshold fills.
    """
    attribution = np.asarray(attribution)
    if attribution.max() <= 0:
        return np.zeros(len(attribution), dtype=bool)
    return attribution >= frac * attribution.max()


def path_gradients(model, x, target=None, baseline=None, n_steps=50):
    """Per-feature gradients at each point along the straight-line IG path.

    Returns
    -------
    grads : np.ndarray, shape (n_steps, n_features)
        Row k is grad f_c evaluated at ``baseline + (k/m)(x - baseline)``.
    delta : np.ndarray, shape (n_features,)
        ``x - baseline``, the factor IG multiplies its path average by.

    Integrated gradients averages ``grads`` over the path. Exposing the raw
    path lets other reductions be tried -- see :func:`reduce_path`.
    """
    import torch

    net = _unwrap(model)
    net.eval()

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.asarray(x), dtype=torch.float32)
    x = x.reshape(1, -1).float()

    if baseline is None:
        baseline = torch.zeros_like(x)
    elif not isinstance(baseline, torch.Tensor):
        baseline = torch.tensor(np.asarray(baseline), dtype=torch.float32).reshape(1, -1)

    if target is None:
        with torch.no_grad():
            out = net(x)
        target = int(out.argmax(dim=1)) if out.ndim > 1 and out.shape[1] > 1 else 0

    grads = []
    for k in range(1, n_steps + 1):
        point = (baseline + (k / n_steps) * (x - baseline)).clone().requires_grad_(True)
        out = net(point)
        score = out[:, target] if out.ndim > 1 and out.shape[1] > 1 else out.squeeze()
        (grad,) = torch.autograd.grad(score.sum(), point)
        grads.append(grad.squeeze().detach().cpu().numpy())

    return np.vstack(grads), (x - baseline).squeeze().detach().cpu().numpy()


def reduce_path(grads, delta, reduction="mean"):
    """Collapse path gradients into one attribution per feature.

    Parameters
    ----------
    reduction : {'mean', 'min', 'min_abs', 'max_abs'}
        ``'mean'`` reproduces standard integrated gradients (the Riemann sum
        approximating the path integral).

        ``'min_abs'`` is the selection-oriented variant: it reports the
        *smallest* gradient magnitude seen anywhere along the path, so a
        feature scores highly only if it matters at every point between the
        baseline and the input. This is the path analogue of MinShap's minimum
        over permutations -- in both cases the minimum is what converts a
        contribution measure into evidence that survives an adversarial choice
        of context.

    Notes
    -----
    Open question from the 05-08-2026 meeting notes. The reason to expect this
    to behave differently from mean-IG: a feature whose gradient is large near
    the input but ~0 near the baseline gets a healthy IG score, yet contributes
    nothing over most of the path. Mean-IG cannot distinguish that from a
    feature that matters uniformly; the minimum can.
    """
    if reduction == "mean":
        reduced = grads.mean(axis=0)
    elif reduction == "min":
        reduced = grads.min(axis=0)
    elif reduction == "min_abs":
        reduced = np.abs(grads).min(axis=0)
    elif reduction == "max_abs":
        reduced = np.abs(grads).max(axis=0)
    else:
        raise ValueError(
            f"reduction must be mean, min, min_abs, or max_abs; got {reduction!r}"
        )
    return reduced * delta


def min_integrated_gradients(model, x, target=None, baseline=None, n_steps=50,
                             reduction="min_abs", abs_value=True):
    """Integrated gradients with the path integral replaced by a minimum.

    Convenience wrapper over :func:`path_gradients` + :func:`reduce_path`.
    Set ``reduction='mean'`` to recover standard IG for a like-for-like
    comparison on the same path samples.
    """
    grads, delta = path_gradients(model, x, target=target, baseline=baseline,
                                  n_steps=n_steps)
    attr = reduce_path(grads, delta, reduction=reduction)
    return np.abs(attr) if abs_value else attr


def dropout_curve(attribution, steps=20):
    """Support size as the selection threshold sweeps from 0 to max.

    Returns ``(thresholds, n_selected)``. A design whose truly-relevant features
    dominate shows a long plateau at the true support size; a diffuse
    attribution decays smoothly with no stable reading.
    """
    attribution = np.asarray(attribution)
    hi = attribution.max()
    if hi <= 0:
        return np.zeros(steps), np.zeros(steps, dtype=int)
    thresholds = np.linspace(0, hi, steps)
    counts = np.array([(attribution >= t).sum() for t in thresholds])
    return thresholds, counts


__all__ = [
    "attribute", "attribute_batch", "select_top_k", "select_threshold",
    "dropout_curve", "path_gradients", "reduce_path", "min_integrated_gradients",
    "METHODS",
]
