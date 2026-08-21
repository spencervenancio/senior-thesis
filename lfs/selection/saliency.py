"""Gradient-based attribution, wrapped for comparison against MinShap/LOCO."""
import numpy as np

METHODS = ("saliency", "input_x_gradient", "integrated_gradients")


def _unwrap(model):
    """Get the torch module out of a skorch wrapper, if needed."""
    return getattr(model, "module_", model)


def attribute(model, x, target=None, method="saliency", n_steps=50, abs_value=True):
    """Attribution for a single input."""
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


def select_top_k(attribution, k):
    """Boolean mask of the k largest attributions -- the fixed-budget rule."""
    attribution = np.asarray(attribution)
    mask = np.zeros(len(attribution), dtype=bool)
    mask[np.argsort(attribution)[-k:]] = True
    return mask


def select_threshold(attribution, frac=0.05):
    """Boolean mask of attributions above ``frac`` of the maximum."""
    attribution = np.asarray(attribution)
    if attribution.max() <= 0:
        return np.zeros(len(attribution), dtype=bool)
    return attribution >= frac * attribution.max()


def path_gradients(model, x, target=None, baseline=None, n_steps=50):
    """Per-feature gradients at each point along the straight-line IG path."""
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
    """Collapse path gradients into one attribution per feature."""
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
    """Integrated gradients with the path integral replaced by a minimum."""
    grads, delta = path_gradients(model, x, target=target, baseline=baseline,
                                  n_steps=n_steps)
    attr = reduce_path(grads, delta, reduction=reduction)
    return np.abs(attr) if abs_value else attr


def dropout_curve(attribution, steps=20):
    """Support size as the selection threshold sweeps from 0 to max."""
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
