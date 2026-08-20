import numpy as np
from sklearn.neighbors import NearestNeighbors


def local_neighborhood(X, x_S, k=50):
    """Find indices of k nearest neighbors of x_S in X (all features)."""
    nn = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X)
    return nn.kneighbors(x_S.reshape(1, -1), return_distance=False)[0]


def local_mse(Y_hat, Y, indices):
    """MSE of Y_hat vs Y restricted to a precomputed neighborhood."""
    return np.mean((Y[indices] - Y_hat[indices]) ** 2)
