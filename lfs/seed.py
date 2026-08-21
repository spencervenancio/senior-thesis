"""Reproducibility helpers.

Call :func:`set_seed` once at the top of a notebook or experiment. Prefer
passing an explicit ``rng`` (a :class:`numpy.random.Generator`) into functions
that need randomness -- global state is a fallback for libraries we do not
control, such as skorch's weight initialization.
"""
import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed python, numpy, and torch; return a fresh Generator for local use.

    Parameters
    ----------
    seed : int
    deterministic_torch : bool
        Force deterministic cuDNN kernels. Slower, and some ops have no
        deterministic implementation, but required for bitwise-reproducible
        neural net training.

    Returns
    -------
    np.random.Generator
    """
    random.seed(seed)
    np.random.seed(seed)  # legacy global, for anything still using np.random.*
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
    """Split ``rng`` into ``n`` independent child generators.

    Use this to give each parallel permutation its own stream. Sharing one
    Generator across joblib workers silently duplicates draws, which would make
    the K permutations in MinShap correlated.
    """
    rng = np.random.default_rng(rng)
    return [np.random.default_rng(s) for s in rng.bit_generator._seed_seq.spawn(n)]


__all__ = ["set_seed", "spawn"]
