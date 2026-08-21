"""Linear baselines."""
from sklearn.linear_model import LinearRegression, LogisticRegression


def logistic_regression(X_train=None, y_train=None, max_iter=300, solver="lbfgs",
                        **kwargs):
    """Logistic regression, fitted when data is supplied.

    Call with no data to get an unfitted estimator for the selection methods to
    refit themselves.
    """
    clf = LogisticRegression(max_iter=max_iter, solver=solver, **kwargs)
    if X_train is not None:
        clf.fit(X_train, y_train)
    return clf


def linear_regression(X_train=None, y_train=None, **kwargs):
    """Ordinary least squares, for the continuous simulated designs.

    Worth running as a baseline even on the nonlinear designs: where a linear
    model already recovers the support, a neural net adds nothing but variance.
    """
    reg = LinearRegression(**kwargs)
    if X_train is not None:
        reg.fit(X_train, y_train)
    return reg


__all__ = ["logistic_regression", "linear_regression"]
