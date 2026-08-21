# Literature

PDFs backing the citations in `../notes/ref.bib`.

| file | citekey | relevance |
|---|---|---|
| `Shap_Feature_Selection.pdf` | `zheng_minshap` | **The method this thesis extends.** Minimum marginal contribution across permutations instead of the average; Type-I error guarantee under DAG faithfulness. Compares against LOCO, GCM, Lasso. |
| `Discovering Conditionally Salient_Features_with_Statistical_Guarantees.pdf` | `gimenez2019discovering...` | Local/conditional feature selection with guarantees — the framing for `S*(x)` varying by observation. |
| `Graph-based regularization ... highly-correlated designs.pdf` | `li2019graphbased` | The GTV paper. Alignment assumption spreads coefficient weight across a correlated cluster — the effect we want in a *model-agnostic* wrapper. |
| `Deep Inside Convolutional Networks ...pdf` | `simonyan2014deepinside` | Original saliency maps (vanilla gradients). |

## Gaps worth filling

Cited or discussed in the notes but not yet in `lit/`:

- **Integrated Gradients** — Sundararajan, Taly, Yan (2017), arXiv:1703.01365. Directly relevant to the open `min(·)`-in-place-of-the-integral question.
- **GCM** (generalized covariance measure) — Shah & Peters. Named at the end of the 06-04-2026 meeting and used as a baseline in the MinShap paper.

## Note

`Youth Protection & Engagement Participant Interacting with Minors Code of
Conduct.pdf` is in this directory but is not research literature — it looks
misfiled. It is untracked in git; move or delete it as you see fit.
