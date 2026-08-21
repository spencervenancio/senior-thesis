# Maintenance Notebook

Chronology of maintenance passes on this repo: what was deleted or consolidated,
what evidence justified it, and what was validated. Research chronology lives in
`notes/meeting_notes.tex`; this file is only about the health of the code.

## Chronology

### 2026-08-21 — repo integrity: sources excluded by over-broad ignore rules

**Theme.** A fresh clone of the public repo did not import or test. Fixing that,
plus one duplicate-owner consolidation found along the way.

**Boundary.** Starting boundary is the existing `HEAD` `04762e3` (clean
worktree, no pre-pass checkpoint was needed or created). Branch
`maintenance-pass-2026-08-21`, per `CLAUDE.md`.

**Baseline.** 28 tracked `.py` files / 2740 LOC before; 30 files / 2790 LOC
after. The pass *adds* tracked lines because two package source files were
missing from version control, not because code grew.

**What changed.**

1. `.gitignore` rule `data/` was unanchored, so it matched `lfs/data/` as well
   as the repo-root MNIST cache. `lfs/data/__init__.py` and
   `lfs/data/patches.py` were therefore never committed. Consequences on a
   clean clone: `from lfs.data import get_patches, load_mnist` (the README
   quickstart) raised `ImportError`, `tests/test_patches.py` failed at import,
   and `pip install .` dropped the `lfs.data` package entirely, since
   `[tool.setuptools.packages.find]` needs an `__init__.py`. Anchored the rule
   to `/data/` and committed both files.
2. `!results/.gitkeep` could never fire — git does not descend into an excluded
   directory, so a negation under `results/` is dead. Changed to `/results/*` +
   `!/results/.gitkeep` and committed the placeholder.
3. Anchored the remaining root-only rules (`/env/`, `/bin/`, `/include/`,
   `/lib/`, `/share/`, `/dist/`, `/build/`, `/plots/`) so the same class of
   accident cannot recur — `lib/` in particular would swallow any future
   `lfs/lib/`.
4. Consolidated `lfs/metrics/local.py` onto `Neighborhood`. `local_score`,
   `local_mse`, and `local_neighborhood` predate `lfs/metrics/pointwise.py`.
5. Fixed a pre-existing `UP035` so the documented `ruff check` command passes.

**Liveness evidence.** `local_score` had zero references anywhere in the repo —
package, tests, configs, notebooks, notes — and its `metric=` argument is the
interface this repo deliberately removed elsewhere (`minshap`/`max_p` raise
`TypeError` on `metric=`). `local_mse` and `local_neighborhood` were referenced
only by their own tests; both live call sites (`selection/_permutation.py`,
`selection/loco.py`) already use `Neighborhood(...).indices` directly, and the
loss itself now comes from `pointwise`. `notes/meeting_notes.tex` mentions none
of the three.

Note the deletion was *not* pure removal: `local_neighborhood` carried an
`x_S is None` guard that the live path lacked, so `minshap(local=True)` with no
`x_S` failed inside sklearn with a misleading "Input X contains NaN". The guard
moved onto `Neighborhood.__init__` and kept a test.

**Deliberately left alone.** The `min(·)`-vs-IG scaffolding in
`lfs/selection/saliency.py` (`path_gradients`, `reduce_path`,
`min_integrated_gradients`, `select_top_k`, `attribute_batch`) has no callers,
but `CLAUDE.md` and `README.md` both name it as open research scaffolding.
Uncalled is the expected state there, not evidence of death. Same for the
historical notebooks (`mnist_exp.ipynb`, `simulated_datasets.ipynb`), which
`CLAUDE.md` marks as expected-stale.

**Validation.** `ruff check lfs experiments tests` clean; `pytest` 102 passed
(103 before — two tests for the removed wrappers replaced by one for the
relocated guard); `python -m experiments.run ... --dry-run` resolves configs; a
fresh `git clone` into a temp dir imports `lfs.data` and passes the suite.

**Residual risk.** Low. The one behavior change is a new `ValueError` from
`Neighborhood` when `x_S is None`, which previously reached sklearn and failed
anyway. No completed-run artifacts, configs, or provenance were touched.

**Next-pass candidate.** No CI workflow exists, so nothing would have caught the
missing-source bug automatically. A minimal GitHub Actions job that clones,
`pip install -e ".[dev]"`, and runs `pytest -m "not slow"` plus `ruff` is the
highest-yield next step; it is additive, so blast radius is nil.
