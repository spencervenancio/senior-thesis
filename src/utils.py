import numpy as np
import os

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

def load_mnist(as_tensor=False):
    from torchvision import datasets
    import torch

    _data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train = datasets.MNIST(root=_data_dir, train=True, download=True)
    test = datasets.MNIST(root=_data_dir, train=False, download=True)

    if as_tensor:
        X_train = train.data.reshape(60000, -1).float() / 255.0
        y_train = train.targets
        X_test  = test.data.reshape(10000, -1).float() / 255.0
        y_test  = test.targets
    else:
        X_train = train.data.numpy().reshape(60000, -1) / 255.0
        y_train = train.targets.numpy()
        X_test  = test.data.numpy().reshape(10000, -1) / 255.0
        y_test  = test.targets.numpy()

    return X_train, X_test, y_train, y_test