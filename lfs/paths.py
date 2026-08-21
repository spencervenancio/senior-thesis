"""Canonical filesystem locations, resolved relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LIT_DIR = ROOT / "lit"

__all__ = ["ROOT", "DATA_DIR", "RESULTS_DIR", "LIT_DIR"]
