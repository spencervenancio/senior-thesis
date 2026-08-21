#!/usr/bin/env python3
"""Enforce the documentation rules stated in CLAUDE.md.

Run over a set of paths (pre-commit passes the staged ones):

    python tools/lint_docs.py $(git ls-files '*.py' '*.md')

Rules
-----
D1  Docstrings in CODE_DIRS are a single line.
D2  No comments in CODE_DIRS except tooling directives (see DIRECTIVES).
D3  No markdown outside ALLOWED_MARKDOWN.

Why these and not a docstring-style linter: the failure mode here is not badly
formatted prose, it is generated prose accumulating faster than anyone reads
it. These three rules are the ones that cap volume rather than police style.
"""
from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

# Directories whose docstrings and comments are capped. Tests are deliberately
# exempt: a test docstring states the bug the test guards, which is the one
# place prose earns its keep.
CODE_DIRS = ("lfs", "experiments")

# Comments that survive because a tool reads them, not a human.
DIRECTIVES = ("noqa", "type:", "pragma:", "ruff:", "fmt:", "isort:", "mypy:")

# Markdown that is allowed to exist. README/CLAUDE are configuration; notes/ is
# the human's. Anything else is the bloat this rule exists to stop -- if you
# genuinely need a new one, add it here in the same commit and say why.
ALLOWED_MARKDOWN = ("README.md", "CLAUDE.md")
ALLOWED_MARKDOWN_DIRS = ("notes",)


def _in(path: Path, roots: tuple[str, ...]) -> bool:
    return path.parts and path.parts[0] in roots


def check_docstrings(path: Path, src: str) -> list[str]:
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: could not parse: {exc.msg}"]

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        expr = node.body[0]
        if "\n" in doc.strip() or expr.end_lineno != expr.lineno:
            name = getattr(node, "name", "<module>")
            problems.append(
                f"{path}:{expr.lineno}: D1 multi-line docstring on {name!r}. "
                f"One line; move any real decision into CLAUDE.md."
            )
    return problems


def check_comments(path: Path, src: str) -> list[str]:
    problems = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        comments = [t for t in tokens if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError):
        return problems

    for tok in comments:
        text = tok.string.lstrip("#").strip()
        if any(d in tok.string for d in DIRECTIVES) or tok.string.startswith("#!"):
            continue
        problems.append(
            f"{path}:{tok.start[0]}: D2 comment: {text[:60]!r}. "
            f"Delete it, or record the decision in CLAUDE.md."
        )
    return problems


def check_markdown(path: Path) -> list[str]:
    if path.as_posix() in ALLOWED_MARKDOWN or _in(path, ALLOWED_MARKDOWN_DIRS):
        return []
    return [
        f"{path}:1: D3 new markdown file. Documentation is written by the human. "
        f"If this is genuinely needed, add it to ALLOWED_MARKDOWN in "
        f"tools/lint_docs.py in the same commit."
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args(argv)

    problems: list[str] = []
    for path in args.paths:
        if not path.is_file():
            continue
        if path.suffix == ".md":
            problems += check_markdown(path)
            continue
        if path.suffix != ".py" or not _in(path, CODE_DIRS):
            continue
        src = path.read_text()
        problems += check_docstrings(path, src)
        problems += check_comments(path, src)

    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} documentation-rule violation(s). "
              f"The rules are stated in CLAUDE.md.", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
