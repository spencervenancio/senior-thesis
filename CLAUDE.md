# CLAUDE.md

Senior thesis on **local feature selection with statistical guarantees** —
model-agnostic selection that survives correlated features and can make a claim
about a single observation rather than a dataset average.

`README.md` is the orientation doc. `notes/meeting_notes.tex` is the research
chronology and the source of truth for what is actually open — read it before
proposing direction, not just code.

## Commands

```bash
pip install -e ".[torch,dev]"          # torch extra = skorch/captum/torchvision
pytest                                  # 102 tests, ~4s
pytest -m "not slow"                    # skip end-to-end recovery runs
ruff check lfs experiments tests
python -m experiments.run experiments/configs/<name>.yaml
python -m experiments.run <config> --dry-run   # print resolved configs, run nothing
```

## Conventions that are easy to get wrong

These are deliberate. Changing them silently corrupts results.

- **`loss=`, never `metric=`.** The rejection threshold is calibrated from the
  variance of a *per-sample* loss, so the value function and its variance must
  come from the same loss. `metric=` and `higher_is_better=` raise `TypeError`
  on purpose — do not "restore" them. See `lfs/metrics/pointwise.py`.
- **Importance is always "increase in loss."** Larger = more important, for
  every method and every loss. Do not reintroduce a direction flag.
- **Estimators are passed unfitted.** `minshap` / `max_p` deepcopy and refit
  internally. Only `loco` takes a fitted model, for its baseline.
- **Pass `rng=`.** Each permutation gets an independent child stream via
  `lfs.seed.spawn`. Sharing one Generator across joblib workers duplicates
  draws and silently correlates the K permutations.
- **Plot functions return `(fig, ax)` and never call `plt.show()`** — the same
  call has to work in a notebook and in a headless run.
- **`n_jobs=1` for skorch models** until a run is known-good; torch and threads
  interact badly under load.
- **Paths come from `lfs.paths`.** No `os.chdir`, no absolute paths. The old
  notebooks were machine-specific because of this.

## Never commit

**This repo is public** (`github.com/spencervenancio/senior-thesis`).

- `lit/*.pdf` — copyrighted, and `Shap_Feature_Selection.pdf` is an
  **unpublished manuscript** (Zheng & Raskutti). Gitignored *and* purged from
  history. Do not re-add. `lit/README.md` + `notes/ref.bib` carry the metadata.
- `data/` — 63MB of MNIST; torchvision re-downloads on demand.
- `results/` — every run is reproducible from its config + seed. Commit the
  config, not the artifact.
- `.claude/settings.local.json` — per-machine.

`git add -A` has already swept a misfiled personal document into a commit once
in this repo. Check `git status` before staging broadly.

## Research work, not just code

- New experiments go in `experiments/configs/*.yaml`, not notebook cells.
  Anything you might want to cite needs a seed and a git SHA behind it.
- `notebooks/` is exploration. Promote anything reusable into `lfs/`.
  `mnist_exp.ipynb` and `simulated_datasets.ipynb` are kept as history and are
  *expected* to be stale — do not "fix" or delete them in a cleanup pass.
- Recovery is scored automatically: the designs in `lfs/data/simulated.py`
  carry their true support, including a point-dependent `local_support(x)` for
  `conditional_interaction`.

## Open problems — do not paper over

Both are real and unresolved. Code that hides them is worse than code that
exposes them.

1. **Non-monotonicity of the local value function.** MinShap's guarantee needs
   `V` monotone in the feature set. That holds globally but not at a point:
   adding a feature can worsen the prediction at `x_S`. So `phi_min` can be
   negative in local mode. This is a property of the estimand. Do not clamp it.
2. **Threshold calibration.** The central open statistical question. The
   saliency `select_threshold` is an admitted heuristic with no error control —
   that gap is the point, not a bug to patch over.

The `min(·)`-in-place-of-the-IG-integral question is scaffolded in
`lfs/selection/saliency.py` (`path_gradients` + `reduce_path`). It is
scaffolding, not a result — no claim has been tested.

## Skills

Installed from `uchicago-dsi/ai-sci-skills` (`~/ai-sci-skills`, symlinked into
`~/.claude/skills/`; update with `git -C ~/ai-sci-skills pull`).

- `$experiment-design` — before proposing any run, sweep, or ablation. Ask
  whether each cell changes a decision; the existing sweeps are not sacred.
- `$sensemaking` — before trusting or reporting a result. Name a falsifier.
- `$skeptical-labmate` — before claiming a method works. This is the one that
  catches a loss mismatch or a missing baseline.
- `$lab-notebook` — experiment logging; complements the provenance stamped by
  `experiments/run.py`.
- `$pi-progress-synthesis` — advisor meetings and lab talks.
- `$handoff` — ending a session with work in flight.
- `$slurm` — only once runs outgrow this laptop.

Run `$maintenance-pass` on a branch here, not `main`: it reads deliberately-kept
historical notebooks as dead code.
