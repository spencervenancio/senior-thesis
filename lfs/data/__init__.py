"""Data sources: MNIST, synthetic designs, and patch groupings."""
from . import simulated
from .mnist import load_mnist
from .patches import get_patches, patches_to_pixels, single_features

__all__ = [
    "simulated", "load_mnist", "get_patches", "single_features", "patches_to_pixels",
]
