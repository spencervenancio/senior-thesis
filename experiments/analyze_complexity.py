"""Readouts for the complexity-vs-local-selection sweep.

Recomputes every selection rule offline from the attributions stored by
``experiments.run``, so the threshold grid, the oracle-k budget, and the
controls all come from one set of fits.

    python -m experiments.analyze_complexity [results_root]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lfs.metrics.recovery import recovery_scores  # noqa: E402
from lfs.paths import RESULTS_DIR, ROOT  # noqa: E402
from lfs.selection.saliency import select_threshold, select_top_k  # noqa: E402

GATES = (2, 7)
TAUS = np.round(np.concatenate([np.logspace(-3, -1.1, 12),
                                np.linspace(0.1, 0.95, 25)]), 5)


def load_runs(root, name="complexity_saliency"):
    runs = []
    for d in sorted(Path(root).glob(f"*_{name}_*")):
        if not (d / "arrays.npz").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        met = json.loads((d / "metrics.json").read_text())
        runs.append({
            "dir": d,
            "hidden": int(cfg["model"]["params"]["module__hidden"]),
            "seed": int(cfg["seed"]),
            "methods": met["methods"],
            "mse_test": met["mse_test"],
            "arrays": np.load(d / "arrays.npz"),
        })
    return runs


def score_point(a, truth, rng):
    """All selection rules for one attribution vector at one point."""
    k = int(truth.sum())
    out = {"oracle_k": recovery_scores(select_top_k(a, k, rng=rng), truth, len(a))}
    curve = []
    for t in TAUS:
        m = select_threshold(a, frac=t)
        curve.append(recovery_scores(m, truth, len(a)))
    out["frac"] = curve
    return out


def role_scores(a, truth, rng):
    """Recall split by feature role: the gates vs the live interaction pair."""
    k = int(truth.sum())
    sel = select_top_k(a, k, rng=rng)
    gate_idx = [j for j in GATES if truth[j]]
    pair_idx = [j for j in range(len(a)) if truth[j] and j not in GATES]
    return (float(sel[gate_idx].mean()) if gate_idx else np.nan,
            float(sel[pair_idx].mean()) if pair_idx else np.nan)


def analyse(runs, rng):
    rows = defaultdict(list)
    for r in runs:
        A = r["arrays"]
        attr, truth, Xq = A["attributions"], A["truth"], A["X_query"]
        margin = np.abs(Xq[:, list(GATES)]).min(axis=1)

        sources = list(zip(r["methods"], attr))
        if "true_gradient" in A:                       # C1 ceiling
            sources.append(("true_gradient", A["true_gradient"]))
        sources.append(("random", rng.random(attr.shape[1:])))   # C0 floor

        for mname, mat in sources:
            f1_o, f1_curve, gate_r, pair_r, tau_star = [], [], [], [], []
            prec_o, rec_o = [], []
            for j in range(len(truth)):
                sc = score_point(mat[j], truth[j], rng)
                f1_o.append(sc["oracle_k"]["f1"])
                prec_o.append(sc["oracle_k"]["precision"])
                rec_o.append(sc["oracle_k"]["recall"])
                curve = np.array([c["f1"] for c in sc["frac"]])
                f1_curve.append(curve)
                tau_star.append(TAUS[int(curve.argmax())])
                g, p = role_scores(mat[j], truth[j], rng)
                gate_r.append(g)
                pair_r.append(p)
            f1_curve = np.vstack(f1_curve)
            rows[mname].append({
                "hidden": r["hidden"], "seed": r["seed"], "mse_test": r["mse_test"],
                "f1_oracle": float(np.mean(f1_o)),
                "precision_oracle": float(np.mean(prec_o)),
                "recall_oracle": float(np.mean(rec_o)),
                "f1_frac_best": float(f1_curve.mean(axis=0).max()),
                "tau_best": float(TAUS[int(f1_curve.mean(axis=0).argmax())]),
                "tau_star_sd": float(np.std(tau_star)),
                "gate_recall": float(np.nanmean(gate_r)),
                "pair_recall": float(np.nanmean(pair_r)),
                "f1_by_margin": [
                    float(np.mean(np.array(f1_o)[margin <= 0.25])),
                    float(np.mean(np.array(f1_o)[(margin > 0.25) & (margin <= 0.75)])),
                    float(np.mean(np.array(f1_o)[margin > 0.75])),
                ],
                "gate_recall_by_margin": [
                    float(np.nanmean(np.array(gate_r)[margin <= 0.25])),
                    float(np.nanmean(np.array(gate_r)[(margin > 0.25) & (margin <= 0.75)])),
                    float(np.nanmean(np.array(gate_r)[margin > 0.75])),
                ],
            })
    return rows


def agg(rows, key):
    """Mean and paired-bootstrap SE across seeds, per width."""
    by_w = defaultdict(list)
    for r in rows:
        by_w[r["hidden"]].append(r[key])
    widths = sorted(by_w)
    mean = np.array([np.mean(by_w[w]) for w in widths])
    se = np.array([np.std(by_w[w], ddof=1) / np.sqrt(len(by_w[w]))
                   if len(by_w[w]) > 1 else 0.0 for w in widths])
    return np.array(widths), mean, se


def main(argv=None):
    root = Path(argv[0]) if argv else RESULTS_DIR
    rng = np.random.default_rng(0)
    runs = load_runs(root)
    if not runs:
        print(f"no complexity_saliency runs under {root}")
        return 1
    print(f"{len(runs)} runs, widths {sorted({r['hidden'] for r in runs})}, "
          f"seeds {sorted({r['seed'] for r in runs})}\n")

    rows = analyse(runs, rng)
    methods = [m for m in rows if m not in ("random", "true_gradient")]

    floor = np.mean([r["f1_oracle"] for r in rows["random"]])
    ceil = np.mean([r["f1_oracle"] for r in rows["true_gradient"]])

    print(f"C0 random floor        F1 = {floor:.3f}")
    print(f"C1 true-gradient ceil  F1 = {ceil:.3f}\n")

    for m in methods:
        w, mu, se = agg(rows[m], "f1_oracle")
        _, gap_o, _ = agg(rows[m], "f1_oracle")
        _, gap_f, _ = agg(rows[m], "f1_frac_best")
        _, gr, _ = agg(rows[m], "gate_recall")
        _, pr, _ = agg(rows[m], "pair_recall")
        _, mse, _ = agg(rows[m], "mse_test")
        _, tb, _ = agg(rows[m], "tau_best")
        print(f"--- {m} ---")
        print(f"{'width':>7} {'F1(oracle)':>12} {'+-SE':>7} {'F1(frac)':>9} "
              f"{'R3 gap':>8} {'gate':>6} {'pair':>6} {'mse':>8} {'tau*':>6}")
        for i, ww in enumerate(w):
            print(f"{ww:>7} {mu[i]:>12.3f} {se[i]:>7.3f} {gap_f[i]:>9.3f} "
                  f"{gap_o[i]-gap_f[i]:>8.3f} {gr[i]:>6.3f} {pr[i]:>6.3f} "
                  f"{mse[i]:>8.4f} {tb[i]:>6.3f}")
        print()

    _plot(rows, methods, floor, ceil)
    return 0


def _plot(rows, methods, floor, ceil):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = dict(zip(methods, ["tab:blue", "tab:orange", "tab:green"]))

    ax = axes[0, 0]
    for m in methods:
        w, mu, se = agg(rows[m], "f1_oracle")
        ax.errorbar(w, mu, yerr=se, marker="o", label=m, color=colors[m], capsize=3)
    ax.axhline(ceil, ls="--", c="k", lw=1.2, label=f"C1 true-gradient ceiling {ceil:.3f}")
    ax.axhline(floor, ls=":", c="r", lw=1.2, label=f"C0 random floor {floor:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("mean local F1 (oracle-k)")
    ax.set_title("R1  g(c)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[0, 1]
    for m in methods:
        w, o, _ = agg(rows[m], "f1_oracle")
        _, f, _ = agg(rows[m], "f1_frac_best")
        ax.plot(w, o - f, marker="s", label=m, color=colors[m])
    ax.axhline(0, c="k", lw=.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("F1(oracle-k) - F1(best frac)")
    ax.set_title("R3  threshold gap")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[1, 0]
    for m in methods:
        w, g, _ = agg(rows[m], "gate_recall")
        _, p, _ = agg(rows[m], "pair_recall")
        ax.plot(w, g, marker="o", ls="-", color=colors[m], label=f"{m} gates")
        ax.plot(w, p, marker="^", ls="--", color=colors[m], alpha=.6,
                label=f"{m} pair")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden width")
    ax.set_ylabel("recall")
    ax.set_title("R5  by feature role")
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    for m in methods:
        w, mse, _ = agg(rows[m], "mse_test")
        _, f1, _ = agg(rows[m], "f1_oracle")
        ax.plot(mse, f1, marker="o", color=colors[m], label=m)
        for i, ww in enumerate(w):
            ax.annotate(str(ww), (mse[i], f1[i]), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xscale("log")
    ax.set_xlabel("test MSE (achieved fit)")
    ax.set_ylabel("mean local F1")
    ax.set_title("R6  F1 vs achieved fit")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    fig.suptitle("Model complexity vs. local feature-selection capability "
                 "(smooth_conditional_interaction, tau=0.25)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "complexity_saliency.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / 'complexity_saliency.png'}")
    return fig, axes


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
