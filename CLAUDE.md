# CLAUDE.md

Senior thesis on **local feature selection with statistical guarantees** —
model-agnostic selection that survives correlated features and can make a claim
about a single observation rather than a dataset average.

## Philosophy — read this before writing anything

This is early-stage research code. **All of it is replaceable.** Optimize for a
small surface area someone can hold in their head, not for preservation.

- **`notes/meeting_notes.tex` is the source of truth** for what is open, what
  was tried, and where this is going. It is written by hand. Read it before
  proposing direction. Do not edit it, do not restate it elsewhere, and do not
  maintain a second status list that will drift out of sync with it.
- **Delete ruthlessly.** If a direction is recorded in the notes, the code that
  explored it does not need to survive to preserve it. Git history is the
  archive. Do not keep something "for reference".
- **Documentation is written by the human.** If Spencer did not think a thing
  was worth writing down, do not write it down for him. Prose you generate is
  the default thing to cut, not the thing to protect.
- **The exception is configuration** — this file, `pyproject.toml`,
  `experiments/configs/*.yaml`. Config is documentation that has teeth, because
  it is executed or enforced. Put durable decisions here, not in prose.
- **One-line docstrings. No comments.** Public functions get a single line; the
  signature and the code carry the rest. A decision that genuinely needs an
  explanation is a convention — record it in the section below, where it is
  read once, instead of in a comment that is read never.
- Errors are the place to be wordy. A loud `TypeError` that explains a removed
  interface is worth ten paragraphs of docstring.

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

Deliberate. Changing them silently corrupts results.

- **`loss=`, never `metric=`.** The rejection threshold is calibrated from the
  variance of a *per-sample* loss, so the value function and its variance must
  come from the same loss. `metric=` and `higher_is_better=` raise `TypeError`
  on purpose — do not "restore" them.
- **Importance is always "increase in loss."** Larger = more important, for
  every method and every loss. Do not reintroduce a direction flag.
- **Estimators are passed unfitted.** `minshap` / `max_p` deepcopy and refit
  internally. Only `loco` takes a fitted model, for its baseline.
- **Pass `rng=`.** Each permutation gets an independent child stream via
  `lfs.seed.spawn`. Sharing one Generator across joblib workers duplicates
  draws and silently correlates the K permutations.
- **A degenerate permutation (`sigma == 0`) gets `z = 0`, so `p = 1`.** It
  carries no evidence and must block rejection. An epsilon floor instead turns
  numerical noise into significance.
- **Plot functions return `(fig, ax)` and never call `plt.show()`** — the same
  call has to work in a notebook and in a headless run.
- **`n_jobs=1` for skorch models** until a run is known-good; torch and threads
  interact badly under load.
- **Paths come from `lfs.paths`.** No `os.chdir`, no absolute paths.
- **`experiments/run.py` imports selection callables directly**
  (`from lfs.selection.minshap import minshap`). Going through the package
  binds the re-exported *function* where the *module* was meant.

## Never commit

**This repo is public** (`github.com/spencervenancio/senior-thesis`).

- `lit/*.pdf` — copyrighted, and `Shap_Feature_Selection.pdf` is an
  **unpublished manuscript**. Gitignored *and* purged from history. `notes/ref.bib`
  carries the metadata.
- `data/` (63MB of MNIST), `results/` (reproducible from config + seed),
  `.claude/settings.local.json`.
- **`.gitignore` rules are unanchored on purpose.** `data/` therefore also
  matches `lfs/data/`, so a new file there needs `git add -f` and will not show
  up in `git status`. Anchoring the rules was tried and rejected as noise — do
  not re-add leading slashes.
- `git add -A` has swept a misfiled personal document into a commit here once.
  Check `git status` before staging broadly.

## Research work

- New experiments go in `experiments/configs/*.yaml`, not notebook cells.
  Anything you might want to cite needs a seed and a git SHA behind it.
- `notebooks/` is scratch. Promote anything reusable into `lfs/`; delete the
  rest rather than letting it rot.
- Recovery is scored automatically: the designs in `lfs/data/simulated.py`
  carry their true support, including a point-dependent `local_support(x)`.
- Two open problems are stated in the notes and must stay visible in the code:
  `phi_min` can be negative in local mode (non-monotonicity of the local value
  function — a property of the estimand, do not clamp it), and
  `select_threshold` is an admitted heuristic with no error control. Code that
  hides either is worse than code that exposes it.

## Skills

Installed from `uchicago-dsi/ai-sci-skills` (`~/ai-sci-skills`, symlinked into
`~/.claude/skills/`; update with `git -C ~/ai-sci-skills pull`).

`$experiment-design` before proposing a run · `$sensemaking` before trusting a
result · `$skeptical-labmate` before claiming a method works · `$lab-notebook`
for experiment logging · `$pi-progress-synthesis` for advisor meetings ·
`$handoff` when ending with work in flight · `$slurm` once runs outgrow this
laptop.

Run `$maintenance-pass` on a branch, not `main`. It will offer to create
`docs/maintenance_notebook.md` — decline. The commit log is the record, and a
generated changelog is exactly the bloat this file exists to prevent.
