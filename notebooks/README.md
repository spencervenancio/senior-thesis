# Notebooks

Exploration only. Anything reusable belongs in `lfs/`; anything you want to cite
belongs in an `experiments/configs/*.yaml` so it carries a seed and a git SHA.

## State of each notebook

| notebook | status |
|---|---|
| `local_loss.ipynb` | imports migrated; **API changed**, see below |
| `saliency_maps.ipynb` | imports migrated; runs as-is |
| `saliency_w_simulated_data.ipynb` | imports migrated; model classes now in `lfs.models.nn` |
| `testing_loco_src.ipynb` | imports migrated; `metric=` calls rewritten to `loss=` |
| `simulated_datasets.ipynb` | **superseded** by `lfs/data/simulated.py` — kept for reference |
| `mnist_exp.ipynb` | **historical**. Predates the package; defines its own `get_patches`, `loco`, and plotting inline. Superseded by `experiments/configs/mnist_patch_sweep.yaml` |
| `local_loan.ipynb` | empty placeholder (loan-approval dataset, from the 04-30 notes) |

Outputs were left intact — they are your recorded results. To stop committing
new ones:

```bash
pip install nbstripout && nbstripout --install
```

## What changed in the restructure

Imports moved from `src.*` to `lfs.*` and were rewritten automatically:

| old | new |
|---|---|
| `from src.minshap import minshap, max_p` | `from lfs.selection import minshap, max_p` |
| `from src.loco import loco` | `from lfs.selection import loco` |
| `from src.utils import load_mnist, get_patches` | `from lfs.data import load_mnist, get_patches` |
| `from src.plotting import ...` | `from lfs.viz import ...` |
| `from src.models.nn import neural_net` | `from lfs.models import neural_net` |
| `import src.dataset as data` | `from lfs.data import simulated as data` |

`os.chdir('/Users/.../thesis')` cells were deleted. The package is installed
with `pip install -e .`, so imports resolve from any working directory.

### API changes you must apply by hand

**`metric=` is now `loss=`.** The rejection threshold is calibrated from the
variance of a *per-sample* loss, so the value function and the variance have to
come from the same loss. Passing `metric=` now raises `TypeError` rather than
silently computing the two from different quantities.

```python
# before
minshap(model, patches, X, y, metric=accuracy_score, K=10)

# after
minshap(model, patches, X, y, loss='zero_one', K=10)      # classification
minshap(model, patches, X, y, loss='squared_error', K=10) # regression
```

Automatic rewrites were applied for `accuracy_score` and `mean_squared_error`.
Any other metric needs a decision, not a substitution — see
`lfs/metrics/pointwise.py`.

**`higher_is_better=` is gone.** Importance is now always "reduction in loss",
so larger is more important for every method and every loss. The old flag was
interpreted differently in local and global mode.

**Plot functions return `(fig, ax)` and no longer call `plt.show()`.** In a
notebook the figure still displays. `fig, axes = plot_selected(...)` now
actually works — previously those functions returned `None`.

**Models are passed unfitted.** `neural_net()` with no arguments returns an
unfitted estimator, which is what `minshap`/`max_p` want. Only `loco` needs a
fitted model.

**Pass `rng=`** for reproducibility. Without it every run draws fresh
permutations.
