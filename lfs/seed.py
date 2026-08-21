"""Reproducibility helpers."""
import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed python, numpy, and torch; return a fresh Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    return np.random.default_rng(seed)


def spawn(rng, n):
    """Split ``rng`` into ``n`` independent child generators."""
    rng = np.random.default_rng(rng)
    return [np.random.default_rng(s) for s in rng.bit_generator._seed_seq.spawn(n)]


__all__ = ["set_seed", "spawn"]
