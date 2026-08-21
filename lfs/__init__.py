"""Local feature selection with statistical guarantees.

Top-level convenience imports::

    from lfs import minshap, max_p, loco, set_seed
    from lfs.data import simulated, get_patches, load_mnist
"""
from .paths import DATA_DIR, RESULTS_DIR, ROOT
from .seed import set_seed, spawn
from .selection.loco import loco
from .selection.maxp import max_p
from .selection.minshap import SelectionResult, minshap

__version__ = "0.2.0"

__all__ = [
    "minshap", "max_p", "loco", "SelectionResult",
    "set_seed", "spawn",
    "ROOT", "DATA_DIR", "RESULTS_DIR",
    "__version__",
]
