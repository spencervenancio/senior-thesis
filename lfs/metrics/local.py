"""Neighborhood construction for local (single-observation) scoring.

Local feature selection scores a model on a neighborhood of a query point x_S
rather than on the whole sample. The neighborhood is built **once** from the
full feature set and then held fixed across every refit, so that changes in the
score reflect the model rather than a moving evaluation set.

The loss evaluated on that neighborhood comes from :mod:`lfs.metrics.pointwise`;
this module only decides *which rows* are scored.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Neighborhood:
    """A fixed k-NN neighborhood around a query point.

    Fitting once and reusing the index matters: MinShap refits the model
    ``K * n_patches`` times, and rebuilding the neighbor index inside that loop
    both wastes time and risks the neighborhood drifting between steps.
    """

    def __init__(self, X, x_S, k=50, metric="euclidean"):
        if x_S is None:
            raise ValueError("x_S is required for local mode")
        self.k = int(min(k, len(X)))
        self.x_S = np.asarray(x_S).reshape(1, -1)
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=metric).fit(X)
        self.distances, idx = self._nn.kneighbors(self.x_S, return_distance=True)
        self.indices = idx[0]
        self.distances = self.distances[0]

    def __len__(self):
        return len(self.indices)

    def __array__(self):
        return self.indices

    @property
    def radius(self):
        """Distance to the furthest included neighbor."""
        return float(self.distances[-1])


__all__ = ["Neighborhood"]
