"""Config-driven experiment runner."""
import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lfs import seed as seed_mod  # noqa: E402
from lfs.data import get_patches, load_mnist, simulated, single_features  # noqa: E402
from lfs.metrics import recovery  # noqa: E402
from lfs.paths import RESULTS_DIR, ROOT  # noqa: E402
from lfs.selection.loco import loco as loco_fn  # noqa: E402
from lfs.selection.maxp import max_p as max_p_fn  # noqa: E402
from lfs.selection.minshap import minshap as minshap_fn  # noqa: E402
from lfs.viz import importance as viz  # noqa: E402


def _build_model(spec):
    """Instantiate an *unfitted* estimator from a {kind, params} spec."""
    from lfs import models

    kind = spec["kind"]
    params = spec.get("params", {}) or {}
    builders = {
        "logistic_regression": models.logistic_regression,
        "linear_regression": models.linear_regression,
        "neural_net": models.neural_net,
        "neural_net_regressor": models.neural_net_regressor,
    }
    if kind not in builders:
        raise KeyError(f"unknown model kind {kind!r}; available: {sorted(builders)}")
    return builders[kind](**params)


def _build_data(spec, rng):
    """Return (X_train, X_test, y_train, y_test, patches, design_or_None)."""
    source = spec["source"]

    if source == "mnist":
        n_train = spec.get("n_train", 5000)
        n_test = spec.get("n_test", 1000)
        X_train, X_test, y_train, y_test = load_mnist(n_train=n_train, n_test=n_test)
        patch_size = spec.get("patch_size", 4)
        patches = get_patches(patch_size, img_size=spec.get("img_size", 28))
        return X_train, X_test, y_train, y_test, patches, None

    if source == "simulated":
        design_name = spec["design"]
        n = spec.get("n", 2000)
        noise = spec.get("noise", 0.1)
        design = simulated.make(design_name, n=n, noise=noise, rng=rng)
        X, y = design.X, design.y
        split = int(len(X) * (1 - spec.get("test_frac", 0.3)))
        patches = single_features(X.shape[1])
        return X[:split], X[split:], y[:split], y[split:], patches, design

    raise KeyError(f"unknown data source {source!r}; expected 'mnist' or 'simulated'")


def _git_info():
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "sklearn", "torch", "skorch", "captum"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _set_in(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def expand_sweep(cfg):
    """Expand a ``sweep`` block into one config per parameter combination."""
    sweep = cfg.pop("sweep", None)
    if not sweep:
        return [cfg]

    keys = list(sweep)
    combos = list(itertools.product(*(sweep[k] for k in keys)))
    configs = []
    for combo in combos:
        c = deepcopy(cfg)
        label = []
        for key, value in zip(keys, combo):
            _set_in(c, key, value)
            label.append(f"{key.split('.')[-1]}{value}")
        c["_sweep_label"] = "_".join(str(x) for x in label)
        configs.append(c)
    return configs


def run_one(cfg, out_root=RESULTS_DIR, dry_run=False):
    """Execute a single resolved config; return the output directory."""
    name = cfg.get("name", "run")
    label = cfg.get("_sweep_label")
    seed = int(cfg.get("seed", 0))

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dirname = f"{stamp}_{name}" + (f"_{label}" if label else "") + f"_s{seed}"
    out_dir = Path(out_root) / dirname

    if dry_run:
        print(f"[dry-run] would write {out_dir}")
        print(yaml.safe_dump(cfg, sort_keys=False))
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = seed_mod.set_seed(seed)
    t0 = time.time()

    X_train, X_test, y_train, y_test, patches, design = _build_data(cfg["data"], rng)
    model = _build_model(cfg["model"])

    method_spec = cfg["method"]
    kind = method_spec["kind"]
    params = dict(method_spec.get("params", {}) or {})

    local_spec = cfg.get("local") or {}
    if local_spec.get("enabled"):
        qi = int(local_spec.get("query_index", 0))
        params.update(local=True, x_S=X_train[qi], k=local_spec.get("k", 50))

    print(f"==> {dirname}: {kind} on {cfg['data']['source']}, "
          f"{len(patches)} patches, seed={seed}")

    if kind == "loco":
        model.fit(X_train, y_train)
        importances = loco_fn(
            model, patches, X_train, X_test, y_train, y_test, **params
        )
        result = {"importances": importances}
        selected_mask = None
    elif kind == "minshap":
        result = minshap_fn(model, patches, X_train, y_train, rng=rng, **params)
        selected_mask = np.asarray(result["rejected"])
    elif kind == "max_p":
        result = max_p_fn(model, patches, X_train, y_train, rng=rng, **params)
        selected_mask = np.asarray(result["rejected"])
    else:
        raise KeyError(f"unknown method {kind!r}; expected minshap, max_p, or loco")

    elapsed = time.time() - t0

    arrays = {k: np.asarray(v) for k, v in result.items()
              if isinstance(v, (np.ndarray, list)) and v is not None}
    np.savez_compressed(out_dir / "arrays.npz", **arrays)

    (out_dir / "config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                       sort_keys=False)
    )

    meta = {
        "name": name, "seed": seed, "method": kind,
        "n_patches": len(patches), "elapsed_sec": round(elapsed, 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(), "versions": _versions(),
        "sweep_label": label,
    }

    metrics = {}
    if design is not None and selected_mask is not None:
        truth = design.support
        if local_spec.get("enabled"):
            truth = design.local_support(X_train[int(local_spec.get("query_index", 0))])
        metrics = recovery.recovery_scores(selected_mask, truth, design.n_features)
        metrics["fdr"] = recovery.false_discovery_rate(
            selected_mask, truth, design.n_features
        )
        metrics["true_support"] = np.asarray(truth).tolist()
        print(f"    recovery: F1={metrics['f1']:.3f} "
              f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}")

    if metrics:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    try:
        _write_figures(cfg, result, patches, design, X_train, local_spec, out_dir, kind)
    except Exception as exc:
        print(f"    [warn] figure generation failed: {exc}")

    print(f"    wrote {out_dir}  ({elapsed:.1f}s)")
    return out_dir


def _write_figures(cfg, result, patches, design, X_train, local_spec, out_dir, kind):
    is_image = cfg["data"]["source"] == "mnist"

    if kind == "loco":
        if is_image:
            fig, _ = viz.plot_patch_importance(result["importances"], patches,
                                               label="LOCO importance")
        else:
            fig, _ = viz.plot_importance_bars(
                result["importances"],
                feature_names=design.feature_names if design else None,
                true_support=design.support if design else None,
                title="LOCO importance",
            )
        fig.savefig(out_dir / "importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    if is_image:
        if local_spec.get("enabled"):
            x_S = X_train[int(local_spec.get("query_index", 0))]
            fig, _ = viz.plot_local_selected(result, patches, x_S)
        else:
            fig, _ = viz.plot_selected(result, patches)
        fig.savefig(out_dir / "selected.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        stat = result.get("phi_min", result.get("p_max"))
        fig, _ = viz.plot_importance_bars(
            stat, feature_names=design.feature_names if design else None,
            true_support=design.support if design else None,
            title=f"{kind} statistic",
        )
        fig.savefig(out_dir / "importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "phi_min" in result:
        fig, _ = viz.plot_threshold_diagnostic(result)
        fig.savefig(out_dir / "thresholds.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="path to a YAML experiment config")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR,
                    help="output root (default: results/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print resolved configs without running")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config's seed")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed

    configs = expand_sweep(cfg)
    print(f"{len(configs)} run(s) from {args.config}")

    out_dirs = [run_one(c, out_root=args.out, dry_run=args.dry_run) for c in configs]

    if not args.dry_run:
        print(f"\nDone. {len(out_dirs)} run(s) under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
