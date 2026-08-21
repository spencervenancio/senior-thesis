import numpy as np
import pytest

from lfs.data.patches import get_patches, patches_to_pixels, single_features


@pytest.mark.parametrize("patch_size", [1, 2, 4, 7, 14, 28])
def test_patches_partition_the_image(patch_size):
    """Every pixel belongs to exactly one patch -- required by the masking logic."""
    patches = get_patches(patch_size, img_size=28)
    all_idx = np.concatenate(patches)
    assert len(all_idx) == 784
    assert set(all_idx.tolist()) == set(range(784))


@pytest.mark.parametrize("patch_size,expected", [(1, 784), (2, 196), (4, 49), (7, 16), (14, 4)])
def test_patch_count(patch_size, expected):
    assert len(get_patches(patch_size, img_size=28)) == expected


def test_patch_indices_are_a_square_block():
    """Patch 0 of size 3 must be the top-left 3x3 block, in row-major order."""
    patch = get_patches(3, img_size=9)[0]
    expected = np.array([0, 1, 2, 9, 10, 11, 18, 19, 20])
    np.testing.assert_array_equal(patch, expected)


def test_non_dividing_patch_size_still_partitions():
    """28 / 3 is ragged; edge patches must clip rather than wrap or overlap."""
    patches = get_patches(3, img_size=28)
    all_idx = np.concatenate(patches)
    assert sorted(all_idx.tolist()) == list(range(784))
    assert len(all_idx) == len(set(all_idx.tolist()))


def test_invalid_patch_size_rejected():
    with pytest.raises(ValueError):
        get_patches(0)
    with pytest.raises(ValueError):
        get_patches(29, img_size=28)


def test_single_features():
    patches = single_features(5)
    assert len(patches) == 5
    np.testing.assert_array_equal(np.concatenate(patches), np.arange(5))


def test_patches_to_pixels_roundtrip():
    patches = get_patches(7, img_size=28)
    values = np.arange(len(patches), dtype=float)
    grid = patches_to_pixels(values, patches, 28)
    assert grid.shape == (28, 28)
    # every pixel in patch j carries value j
    for j, patch in enumerate(patches):
        np.testing.assert_allclose(grid.ravel()[patch], values[j])
