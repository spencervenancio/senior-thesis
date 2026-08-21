"""Patch construction for image-shaped feature vectors.

A "patch" is an index array into a *flattened* image, so the same selection
code works for single pixels ([[0], [1], ...]) and for square blocks.
"""
import numpy as np


def get_patches(patch_size, img_size=28):
    """Return a list of patch index arrays for a flattened img_size x img_size vector.

    Parameters
    ----------
    patch_size : int
        Side length of each square patch. Must divide img_size evenly, otherwise
        the trailing partial patches are ragged (see note below).
    img_size : int
        Side length of the source image.

    Returns
    -------
    list of np.ndarray
        Each array holds the flat indices belonging to one patch.

    Notes
    -----
    When patch_size does not divide img_size, edge patches are clipped rather
    than padded, so they contain fewer indices. Every pixel still appears in
    exactly one patch, which is what the selection methods require.
    """
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if patch_size > img_size:
        raise ValueError(f"patch_size {patch_size} exceeds img_size {img_size}")

    patches = []
    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            idx = [
                (i + di) * img_size + (j + dj)
                for di in range(min(patch_size, img_size - i))
                for dj in range(min(patch_size, img_size - j))
            ]
            patches.append(np.array(idx))
    return patches


def single_features(n_features):
    """Return one patch per feature -- the ungrouped / per-pixel case."""
    return [np.array([j]) for j in range(n_features)]


def patches_to_pixels(values, patches, img_size=28):
    """Broadcast one value per patch back onto the pixel grid, for plotting."""
    pixels = np.zeros(img_size * img_size)
    for patch_idx, patch in enumerate(patches):
        pixels[patch] = values[patch_idx]
    return pixels.reshape(img_size, img_size)


__all__ = ["get_patches", "single_features", "patches_to_pixels"]
