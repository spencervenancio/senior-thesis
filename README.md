# Local Feature Selection with Statistical Guarantees

Senior thesis — Spencer Venancio.

Model-agnostic feature selection that survives correlated features and can make
a claim about a single observation rather than a dataset average.

Research chronology and open questions:
[`notes/meeting_notes.tex`](notes/meeting_notes.tex).
Conventions and philosophy: [`CLAUDE.md`](CLAUDE.md).

## Setup

```bash
pip install -e ".[torch,dev]"
pre-commit install                      # once, per clone
pytest
python -m experiments.run experiments/configs/<name>.yaml
```

## Documentation rules

This is early-stage research code: all of it is replaceable, `notes/` is the
hand-written source of truth, and generated prose is the first thing cut.
Enforced by `tools/lint_docs.py` in pre-commit — violations fail the commit.

| | rule | scope |
|---|---|---|
| D1 | docstrings are a single line | `lfs/`, `experiments/` |
| D2 | no comments, except directives like `# noqa` | `lfs/`, `experiments/` |
| D3 | no markdown outside `README.md`, `CLAUDE.md`, `notes/` | repo-wide |

`tests/` and `tools/` are exempt from D1/D2 — a test docstring states the bug it
guards. To add a markdown file, add it to `ALLOWED_MARKDOWN` in
`tools/lint_docs.py` in the same commit.
