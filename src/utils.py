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

def load_mnist():
    from torchvision import datasets
    
    train = datasets.MNIST(root='data', train=True, download=True)
    test = datasets.MNIST(root='data', train=False, download=True)


    X_train = train.data.numpy().reshape(60000, -1) / 255.0  # (60000, 784)
    y_train = train.targets.numpy()                           # (60000,)

    X_test  = test.data.numpy().reshape(10000, -1) / 255.0   # (10000, 784)
    y_test  = test.targets.numpy()                            # (10000,)
    
    return X_train, X_test, y_train, y_test