import numpy as np

def get_patches(patch_size, img_size=28):
    """Returns list of patch index arrays for flattened img_size x img_size vector."""
    patches = []
    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            idx = [
                (i + di) * img_size + (j + dj)
                for di in range(patch_size)
                for dj in range(patch_size)
            ]
            patches.append(np.array(idx))
    return patches