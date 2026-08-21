# Design: model complexity vs. local feature-selection capability

Status: **proposed.** Stage 0 (§4) has been run; nothing else has.
Written against `main` @ `04762e3`. Target config
`experiments/configs/complexity_saliency.yaml` does not exist yet.

---

## 1. The question

Characterise

$$ g(c) \;=\; \mathbb{E}_{\mathbf{x}}\Big[\, \mathrm{F1}\big(\hat S_{\mathbf{x}}(f_c),\; S^*_{\mathbf{x}}\big) \Big] $$

- $c$ — a scalar model-complexity index (§5),
- $f_c$ — a model of complexity $c$ fit to a design whose support varies by point,
- $S^*_{\mathbf{x}}$ — the true local support, from
  `SimulatedDataset.local_support(x)` (`lfs/data/simulated.py:150`),
- $\hat S_{\mathbf{x}}$ — what a saliency rule selects at $\mathbf{x}$.

Motivating hypothesis: if $g$ saturates at $c^*$, then (a) selection stops
being capacity-limited past $c^*$, and (b) whatever governs the plateau
plausibly also governs the *threshold* — the open question in `CLAUDE.md`.

### $g$ is a composition, and that is the design problem

$$ c \;\longrightarrow\; \text{achieved fit} \;\longrightarrow\; \text{attribution quality} \;\longrightarrow\; \hat S_{\mathbf{x}} \ \text{(via a rule)} $$

A plateau can be produced at any arrow, and each implies a different follow-up:

| Saturating arrow | Meaning | Follow-up |
|---|---|---|
| $c \to$ fit | the design stopped being hard | uninteresting — use a harder design |
| fit $\to$ attribution | gradients stop improving though the fit does | interesting; a property of saliency |
| attribution $\to \hat S$ | the ranking is right, the *rule* is the bottleneck | **this is the threshold problem** |

So the decision-critical readout is not $g(c)$ but the **gap between selection
at an oracle budget and selection at a threshold** (§8, R3).

---

## 2. Decision to make

**Is gradient attribution worth pursuing as a local selector, and is a fixed
threshold rule even plausible?** One of:

- **(a)** viable past some $c^*$ → next project is calibrating the rule.
- **(b)** ranks correctly, no fixed rule works → saliency is a *screen*; the
  threshold is the thesis contribution.
- **(c)** never ranks correctly → negative result; local MinShap stays the only
  candidate.

---

## 3. Live hypotheses

- **H1 Saturating** — rises, plateaus at $c^*$, height $< 1$.
- **H2 Inverted-U** — rises then falls: capacity fits noise, and the gradient
  *field* degrades faster than the function value. Live. The design must be able
  to see it; do not fit a monotone saturating curve by default.
- **H3 Estimand-limited ceiling** — $g$ plateaus low for reasons unrelated to
  $c$. **Confirmed for `conditional_interaction`, see §4.**
- **H4 No plateau in budget** — still climbing at the largest $c$ affordable.
  Then $c^*$ is not identified and reporting one would be fabrication.

---

## 4. Stage 0 — RUN, and it changes the design

The cheap discriminative check: score the **analytic gradient of the true $f$**
against $S^*_{\mathbf{x}}$. No training, no torch, seconds. It bounds what any
gradient method can achieve on this design, independent of $c$.

For `conditional_interaction`,
$y = 2x_1x_2\mathbb{1}\{x_3>0\} + x_4x_5\mathbb{1}\{x_3<0\} + 9x_6x_7\mathbb{1}\{x_8>0\} + x_9x_{10}\mathbb{1}\{x_8<0\}$:

200 query points, balanced across the four gate quadrants, seed 0, $n=4000$:

| selector | F1 | precision | recall |
|---|---|---|---|
| random (floor) | **0.594** | 0.594 | 0.594 |
| true-gradient, oracle-$k$ | **0.750** | 0.750 | 0.750 |
| true-gradient, frac-of-max 0.05 | 0.746 | 1.000 | 0.602 |
| true-gradient, oracle-$k$, gates removed from $S^*$ | **1.000** | 1.000 | 1.000 |

Diagnostics:

```
features with strictly-positive true |grad| : 4.00 of 10, at every x   (|S*_x| = 6)
gate {x3,x8} recall, argsort tie-break      : 0.240
gate {x3,x8} recall, random tie-break       : 0.355
gate recall expected by pure chance         : 0.333
active-pair recall                          : 1.000
```

### What this establishes

1. **The gates are gradient-invisible, exactly.** $\partial f/\partial x_3 = 0$
   almost everywhere — an indicator is flat off a measure-zero set. Exactly 4 of
   the 6 features in $S^*_{\mathbf{x}}$ carry any gradient signal at all, at
   every single point. The apparent 0.24 gate recall is **pure `argsort`
   tie-breaking among exact zeros**: randomising the tie-break gives 0.355,
   matching the 0.333 expected by chance. Zero signal.
2. **The ceiling is 0.750 and the floor is 0.594.** The entire usable dynamic
   range of $g(c)$ on this design is **0.156 of F1**, and 2/6 of the target is
   unreachable *in principle*. A curve rising from 0.60 to 0.72 would look like
   a beautiful saturating result and would mean almost nothing.
3. **Conditional on the gates, saliency is perfect.** F1 = 1.000 on the active
   interaction pairs. So H3 is confirmed, and the failure is entirely
   attributable to the estimand, not to any model.

So **H3 is settled without spending a single fit** — which is the whole point of
running Stage 0 first. Reproduce:
`scratchpad/stage0.py`, seed 0, `main` @ `04762e3` (promote to
`experiments/` per W6).

### Consequences — the design must change

- **The main sweep moves to a smooth-gate design** (W4): replace
  $\mathbb{1}\{x_3>0\}$ with $\sigma(x_3/\tau)$. Then
  $\partial f/\partial x_3 = (2x_1x_2 - x_4x_5)\,\sigma'(x_3/\tau)/\tau \ne 0$,
  peaking at $x_3 = 0$ and decaying with $|x_3|$. Gate recall becomes a real
  function of margin, the ceiling becomes attainable, and $\tau$ is a difficulty
  knob commensurable with $c$.
  *Caveat to state, not hide:* as $\tau$ grows the inactive branch never fully
  switches off, so $S^*_{\mathbf{x}}$ stops being sharply defined. $\tau$ must
  stay small enough that this is still a local-*support* problem;
  $\tau \in \{0.1, 0.25\}$, and report the induced contamination.
- **`conditional_interaction` is retained as a labelled control**, not the main
  design, and always reported with the 0.594 floor and 0.750 ceiling drawn on
  the plot.
- **Score against both** $S^*_{\mathbf{x}}$ and the *reachable* support
  $S^*_{\mathbf{x}} \setminus \{\text{gates}\}$. Reporting only the former
  confounds an estimand limitation with a capacity limitation.
- **Break ties randomly in `select_top_k`.** The current `np.argsort` is
  deterministic in feature index, which manufactures ~0.24 of fake gate recall.
  Any design with exact-zero attributions inherits this bias (W7).

---

## 5. The complexity axis

Width only, depth frozen — `DeepMLP(hidden=h, depth=2)`:

$$ h \in \{4, 8, 16, 32, 64, 128, 256\}, \qquad c := \log_{10}(\#\text{params}) $$

Seven log-spaced levels, one variable. Depth is a *secondary* sweep
($h=64$, depth $\in\{1,2,3,4\}$), run only if the width curve is ambiguous.

Frozen: design, $n$, noise, split, query points, optimiser, lr, batch size,
epoch budget, attribution method, init-seed set.

**Training budget is a confound, so measure it rather than assume it away.**
`max_epochs: 300` with early stopping on a held-out validation split; record
achieved train/test MSE for every arm; plot F1 against achieved test MSE as well
as against $c$. That is the arrow-1 vs. arrow-2 separation from §1.

---

## 6. Data and query points

Main: `smooth_conditional_interaction(tau=0.1)` (W4). Control:
`conditional_interaction`. $n = 20{,}000$, noise 0.1, 30% test.

$M = 200$ query points from the test split, **balanced 50/50/50/50 across the
four gate quadrants** (signs of $x_3, x_8$) so all four distinct supports are
equally represented, stratified within quadrant over gate margin. Same 200
points for every arm — paired design.

$|S^*_{\mathbf{x}}| = 6$ for every $\mathbf{x}$ in both designs. Constant
support size makes the thresholding question easier than it really is (oracle-$k$
is a single fixed $k$), so **no threshold claim from this experiment
generalises** until a variable-sparsity design exists (W5). Treat it as a
deliberate simplification and say so in the writeup.

---

## 7. Arms

Attribution is one backward pass; fitting is the expensive part. Selection rules
are crossed over stored attributions for free.

**Model arms** — M1–M7, `DeepMLP(hidden=h, depth=2)`, 7 × 5 init seeds = 35 fits.

**Controls** (cheap, and they bound the curve):

| ID | What | Purpose |
|---|---|---|
| C0 | random attribution | F1 floor — **0.594**, high, must be on every plot |
| C1 | analytic gradient of true $f$ | ceiling of saliency-as-estimand — **done, §4** |
| C2 | linear regression, $a_j = \lvert\beta_j\rvert$ | zero-complexity anchor; constant in $\mathbf{x}$, so local F1 = global F1 |

**Baseline** — B1: local MinShap (`conditional_local.yaml` settings) at
$h \in \{16, 256\}$. The in-repo method saliency must beat. Two cells only; it
dominates wall-clock, run it last.

**Selection rules**, crossed over every arm at no extra fitting cost:

| ID | Rule | Isolates |
|---|---|---|
| S-oracle | top-$k$, $k=\lvert S^*_{\mathbf{x}}\rvert$ (`select_top_k`, random tie-break) | ranking quality alone |
| S-tang | normalise to $[0,1]$, select $>0.5$ | the published rule, transplanted |
| S-frac | $a_j \ge \tau\max_k a_k$, $\tau \in \{0.01,\dots,0.9\}$ (`select_threshold`) | full precision/recall trade-off, $\tau^*$ per point |

Attribution methods: `saliency`, `input_x_gradient`, `integrated_gradients` —
all present in `lfs/selection/saliency.py`. Three backward passes, not three
fits, so sweep all three.

---

## 8. Readouts

- **R1 — $g(c)$**: mean local F1 vs. $\log_{10}(\#\text{params})$, one line per
  rule, with C0/C1/C2 as horizontal reference lines. *Primary plot.*
- **R2 — Coverage / Localization** (§9): the precision/recall decomposition of
  R1. Says whether a plateau is a recall wall or a precision wall.
- **R3 — the threshold gap**,
  $\mathrm{F1}@\text{oracle-}k - \mathrm{F1}@S\text{-frac}(\tau^{\text{best}})$ vs. $c$.
  **Decision-critical.** Plateau + gap $\to 0$ ⇒ the model was the bottleneck
  (outcome **a**). Plateau + gap stays large ⇒ the *rule* is the bottleneck
  (outcome **b**) — the thesis's open problem, quantified.
- **R4 — $\tau^*(\mathbf{x}, c)$ stability**: per-point optimal $\tau$, plotted
  against $c$. Concentrating and settling past $c^*$ ⇒ a fixed rule is
  plausible. Drifting or wide at every $c$ ⇒ a frac-of-max rule **cannot** have
  error control, and we say so.
- **R5 — F1 by feature role** (gates vs. active pair) and **stratified by gate
  margin** $|x_3|$. Guards §4; on the smooth design, gate recall must decrease
  in margin or the pipeline is wrong.
- **R6 — achieved train/test MSE per arm**, and F1 re-plotted against it.
- **R7 — visual**: $200 \times 10$ attribution heatmap (points × features) with
  true local support outlined, one panel per complexity level. Attribution
  bleeding onto the *inactive* branch is obvious by eye and invisible in a mean.

Plot helpers return `(fig, ax)` and never call `plt.show()`.

---

## 9. The Tang et al. metrics

Tang et al., ICLR 2022, *Self-Supervised Graph Neural Networks for Improved
Electroencephalographic Seizure Analysis* (arXiv:2104.08336). Over an occlusion
map $M$ and binary annotation $M^{\text{annot}}$ on channels × seconds:

$$ \text{Coverage} = \frac{\sum_{ij}\mathbb{1}\{M_{ij}>0.5\}M^{\text{annot}}_{ij}}{\sum_{ij}M^{\text{annot}}_{ij}}, \qquad \text{Localization} = \frac{\sum_{ij}\mathbb{1}\{M_{ij}>0.5\}M^{\text{annot}}_{ij}}{\sum_{ij}\mathbb{1}\{M_{ij}>0.5\}} $$

These are **exactly recall and precision** of a thresholded importance map
against a ground-truth support. `lfs/metrics/recovery.recovery_scores` already
returns both, so the "F1-type calculation" is the harmonic mean of Tang's two
numbers and **needs no new metric code** — only a name-mapping in the writeup.
Report all three; F1 alone hides which side is failing.

Three differences to state rather than paper over:

1. **Occlusion, not gradients.** Their $M$ is a zero-fill occlusion map. The
   metric transfers; the attribution does not. `loco` on single features is the
   occlusion analogue and is already implemented — one arm if reviewers ask.
2. **The 0.5 is not scale-free.** It cuts a relative-logit-change scale. Saliency
   has no comparable scale, so the transplant needs a normalisation choice, and
   frac-of-max is one. This is the open threshold problem wearing a different
   hat, so **S-tang is reported as one point on the S-frac curve**, not as an
   independent method.
3. **Grid vs. point.** They aggregate over channels × time; we score one
   $\mathbf{x}$ against $S^*_{\mathbf{x}}$ then average over $M$ points. The unit
   of variation is the query point (§10).

Not in `notes/ref.bib` — add before citing (W6).

---

## 10. Stage 1, stop rule, error bars

35 fits. Predicted signatures:

| Hyp. | R1 | R3 | R4 | Decision |
|---|---|---|---|---|
| H1 | rises, plateaus at $c^*$ | $\to 0$ | concentrates past $c^*$ | **(a)** calibrate the threshold next |
| H1′ | rises, plateaus | stays large | wide / drifting | **(b)** saliency screens; rule is the problem |
| H2 | inverted-U | grows past the peak | drifts up | report $\arg\max$; capacity control joins the method |
| H3 | flat, near the C1 ceiling | — | — | estimand-limited — **already observed on the control design** |
| H4 | still climbing at $h=256$ | — | — | **report no $c^*$**; extend to $h=1024$ once, then stop |

**Stop rule, hypothesis-scoped.** Failure of saliency retires *saliency as a
selector*, not local MinShap (B1), which keeps its standing under existing
evidence. Failure of the smooth-gate design retires *that design*, not the
question. If Stage 1 returns H4 after one extension, stop and record "complexity
range insufficient at laptop scale" rather than widening the grid.

**On recovering the asymptote.** Seven levels × five seeds cannot support
estimating an asymptote parameter from a fitted saturating curve — plateau and
rate are badly confounded at this resolution and a fitted $c^*$ would carry a
fake standard error. Use a **pre-registered plateau rule**: $c^*$ is the
smallest $c$ whose paired difference from the largest $c$ has a 95% bootstrap CI
inside $(-\varepsilon, \varepsilon)$, $\varepsilon = 0.02$ F1, fixed before
looking. Report "no $c^*$ identified" when nothing qualifies.

**Error bars.** The unit of variation is the **query point**, not the seed — 200
points share one trained model, so their F1s are correlated. Bootstrap over
query points, clustered; use **paired** differences across levels (same points,
same data seed, different init seed). Five init seeds give the between-seed
component; report both. Averaging over points also hides the margin dependence
(§4), so always show the distribution (R7), never only the mean.

---

## 11. Work items

| # | Work | Why |
|---|---|---|
| **W1** | Add `kind: saliency` to `experiments/run.py` — build, fit, attribute over a *batch* of query points | Runner dispatches only `minshap`/`max_p`/`loco`. |
| **W2** | Batch local scoring: `local.query_index` is a single int and sweeping it writes one result dir per point. Add `local.query_indices` with one `metrics.json` of per-point rows | 200 dirs per arm is unusable. |
| **W3** | Store raw attributions in `arrays.npz` ($M \times p$) so the $\tau$ grid, S-tang, S-oracle, R4 and R7 recompute offline from one run | Makes the rule-crossing genuinely free and re-analysable without refitting. |
| **W4** | `smooth_conditional_interaction(tau=...)` in `lfs/data/simulated.py` with matching `local_support` | §4. The main design. |
| **W5** | A variable-$\lvert S^*_{\mathbf{x}}\rvert$ design | §6. Needed before any threshold claim generalises. |
| **W6** | Promote `scratchpad/stage0.py` into `experiments/`; add Tang et al. to `notes/ref.bib` | §4 is now a citable result and needs a seed + SHA behind it. |
| **W7** | Random tie-breaking in `select_top_k`; tests for quadrant balance and F1 of a known mask | §4 — deterministic argsort manufactures fake recall on exact-zero attributions. |

Conventions in force: pass `rng=`, spawn child streams via `lfs.seed.spawn`;
estimators go in **unfitted**; paths from `lfs.paths`; `n_jobs=1` for skorch
until known-good; commit the config, not `results/`.

The `loss=` / `metric=` convention does **not** bind here: saliency differentiates
the model output, not a per-sample loss, so there is no variance-calibrated
threshold in play. Say this out loud — it is precisely *why* saliency has no
error control, and why R3/R4 are the interesting readouts. MinShap
non-monotonicity does not apply either (no value function), except in arm B1,
where `phi_min` may be negative and **must not be clamped**.

---

## 12. Cost

35 fits of a ≤2×256 MLP on $n=20{,}000$: order 20–40 min on this laptop.
Attribution: $200 \times 35 \times 3$ single backward passes — seconds. Threshold
grid, S-tang, S-oracle, all plots: offline from `arrays.npz`, free. B1 dominates
wall-clock; run last. No cluster; `$slurm` stays unopened.

Environment: `/opt/anaconda3/bin/python` — torch 2.11.0, captum 0.9.0,
skorch 1.3.1, sklearn 1.5.1.

---

## 13. Deliberately not running

- **`min_integrated_gradients` / `reduce_path`.** Present in
  `lfs/selection/saliency.py` and a separate open question. One unvalidated
  thing per experiment.
- **MNIST.** No ground-truth $S^*_{\mathbf{x}}$, so no F1. Qualitative-demo
  dataset, not the scoring dataset.
- **The depth sweep**, unless the width curve is ambiguous. Two capacity knobs at
  once is a grid without a decision rule.
- **`max_p`, `loco`** as full arms. `loco` enters only for the occlusion-analogue
  comparison in §9(1).
- **Any claim about a calibrated threshold.** This design can say whether a fixed
  rule is *plausible* (R4). It cannot produce one with error control, and a
  positive R4 is not a guarantee.
