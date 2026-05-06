import numpy as np

def local_expecation(X, x_S, Y_hat, Y, active_cols, tolerance=0.005):
    """Calculates the E((Y-f_s(X))^2 | X_S = x_s) using a KNN approach
    """
    abs_diff = np.abs(X[:, active_cols] - x_S[active_cols])
    local_indices = np.where(np.all(abs_diff < tolerance, axis=1))[0]
    is_self = np.all(abs_diff[local_indices] == 0, axis=1)
    local_indices = local_indices[~is_self]
    local_predictions = Y_hat[local_indices]
    return np.mean((Y[local_indices] - local_predictions) ** 2)