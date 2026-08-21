# Local Feature Selection with Statistical Guarantees

Senior thesis — Spencer Venancio.

Model-agnostic feature selection that stays honest under **correlated features**,
and that can make a claim about a *single observation* rather than an average
over the dataset.

The motivating picture is MNIST: no individual pixel is useful, but a patch is.
The motivating question is what happens when features explain each other away —
LOCO drops an entire correlated cluster, LASSO keeps one arbitrary
representative, and neither is obviously the right answer.

## Where the research stands

Chronology lives in [`notes/meeting_notes.tex`](notes/meeting_notes.tex). In short:

| Thread | Status |
|---|---|
| Patch-size effects on LOCO / MinShap over MNIST | explored |
| Local value function `E[(Y - f_S(X))² \| X_S = x_S]` via k-NN | implemented (`local=True`) |
| Saliency / input×gradient / integrated gradients | implemented, compared |
| Alignment (GTV-style) in a wrapper method | **open** |
| Threshold for selection | **open** — the central statistical question |
| `min(·)` in place of the IG path integral | **open** — scaffolded, see below |
| Generalized covariance measure (GCM) | not started |

### Two known problems, written down so they stay visible

**Non-monotonicity of the local value function.** MinShap's guarantee leans on
`V` improving as features are added. That holds globally but *not* at a single
point: adding a feature can make the prediction at `x_S` worse. Reported
`phi_min` values can therefore be negative in local mode. This is a property of
the estimand, not a bug in the code.

**Loss consistency.** The rejection threshold is built from `sigma`, the
variance of the per-sample change in loss, so `V` and `sigma` must come from the
same loss. See [`lfs/metrics/pointwise.py`](lfs/metrics/pointwise.py) — the
earlier implementation measured `V` with `accuracy_score` but `sigma` with
squared error on integer class labels, which is not a loss on a nominal label.
Pass `loss=` explicitly.

## Install

```bash
pip install -e ".[torch,dev]"     # torch extra covers skorch/captum/torchvision
pytest                            # 102 tests, ~4s
```

The `torch` extra is optional: the core selection code, the simulated designs,
and most of the test suite run without it.

## Layout

```
lfs/                    the package
  data/                 MNIST loading, synthetic designs, patch groupings
  models/               sklearn + skorch estimators
  selection/            loco, minshap, maxp, saliency  (_permutation = shared engine)
  metrics/              pointwise losses, k-NN neighborhoods, recovery scoring
  viz/                  plots; every function returns (fig, ax), none call show()
experiments/
  configs/*.yaml        declarative experiment definitions
  run.py                the runner
results/                one directory per run (gitignored)
notebooks/              exploration only — promote anything reusable into lfs/
notes/                  meeting notes (LaTeX) and figures for the writeup
lit/                    papers
tests/
```

## Running an experiment

```bash
python -m experiments.run experiments/configs/mnist_local_minshap.yaml
python -m experiments.run experiments/configs/mnist_patch_sweep.yaml   # 8 runs
python -m experiments.run experiments/configs/simulated_recovery.yaml --seed 7
```

Each run writes `results/<timestamp>_<name>_s<seed>/` containing:

| file | what |
|---|---|
| `config.yaml` | the exact resolved parameters |
| `meta.json` | git SHA, dirty flag, seed, elapsed time, library versions |
| `arrays.npz` | `phi_min`, `t_j`, `rejected`, and the raw `phi` / `sigma` |
| `metrics.json` | precision / recall / F1 / FDR against `S*`, on simulated designs |
| `*.png` | figures |

This exists so that a plot you find in six months can be traced to the code and
parameters that made it. `plots/` and loose PNGs cannot do that; prefer a config
over a notebook cell for anything you might want to cite.

A `sweep:` block maps dotted config paths to lists and expands to one run per
combination:

```yaml
sweep:
  data.patch_size: [2, 4, 7, 14]
  method.kind: [minshap, max_p]
```

## Using the library

```python
from lfs import minshap, set_seed
from lfs.data import get_patches, load_mnist
from lfs.models import neural_net

rng = set_seed(0)
X_train, X_test, y_train, y_test = load_mnist(n_train=5000, n_test=1000)
patches = get_patches(patch_size=4)          # 49 patches

result = minshap(
    neural_net(),                            # unfitted: minshap refits it
    patches, X_train, y_train,
    loss="zero_one", K=50, alpha=0.05,
    local=True, x_S=X_train[42], k=50,       # local mode
    rng=rng,
)
result.selected          # indices of selected patches
result["phi_min"]        # minimum contribution across the K permutations
```

Scoring against a known support:

```python
from lfs.data import simulated
from lfs.metrics.recovery import recovery_scores

design = simulated.conditional_interaction(n=20000, rng=rng)
truth = design.local_support(design.X[0])    # S*(x) — varies by observation here
recovery_scores(result["rejected"], truth, design.n_features)
```

### The synthetic designs

`lfs.data.simulated` carries ground-truth support with the data, so recovery is
scored automatically instead of against a hand-maintained dict.

| design | support | note |
|---|---|---|
| `xor` | {x1, x2} | marginally uninformative — defeats screening |
| `linear_additive` | {x1,x3,x5,x6,x7,x10} | coefficients 2,1,9,9,3,1 |
| `nonlinear_additive` | same | squares, cubes, sin/exp/cos |
| `conditional_interaction` | all 10 globally | **`S*(x)` varies** — x3, x8 gate which pair is live |
| `logistic` | {x1,x3,x4,x6,x8,x9} | continuous response in (0,1) |
| `logistic_bernoulli` | same | actual binary labels |

`conditional_interaction` is the design that separates local from global
selection: only 2 of the 4 interacting pairs are active at any `x`, so a single
global ranking is necessarily wrong.

## The `min(·)` integrated-gradients question

From the 05-08-2026 notes: *can the integral operator in IG be replaced with a
`min(·)` to get a feature selection method?* The scaffolding is in
[`lfs/selection/saliency.py`](lfs/selection/saliency.py):

```python
from lfs.selection.saliency import path_gradients, reduce_path

grads, delta = path_gradients(model, x)          # (n_steps, n_features)
ig      = reduce_path(grads, delta, "mean")      # standard IG
min_ig  = reduce_path(grads, delta, "min_abs")   # the proposed variant
```

`path_gradients` returns the raw per-step gradients so reductions can be
compared on identical path samples. The logic mirrors MinShap: a minimum
survives an adversarial choice of context, where a mean does not. A feature
whose gradient is large near the input but ~0 near the baseline scores well
under mean-IG despite contributing over almost none of the path; the minimum
separates those cases.

What's still missing is the same thing that's missing for saliency generally —
a calibrated threshold. `select_threshold` is an uncalibrated heuristic, and
naming that gap is the point.

## Conventions worth knowing

- **Importance is always "increase in loss".** Larger = more important, for
  every method and every loss. The old `higher_is_better` flag is gone; it meant
  different things in local and global mode.
- **Estimators are passed unfitted.** `minshap` / `max_p` deepcopy and refit
  internally. Only `loco` needs a fitted model, for its baseline.
- **Randomness is explicit.** Pass `rng=`. Each permutation gets an independent
  child stream via `lfs.seed.spawn`, so parallel workers don't duplicate draws.
- **Plots return `(fig, ax)`** and never call `plt.show()`.
- **`n_jobs=1` for skorch models** until a run is known-good; threads and torch
  interact badly under load.

## Notebooks

`notebooks/` is for exploration. Anything reusable belongs in `lfs/`, and
anything you want to cite belongs in an experiment config. The current notebooks
predate this restructure and still import from the old `src.*` layout — see
`notebooks/README.md`.

Strip outputs before committing:

```bash
nbstripout --install     # once, installs a git filter
```
