import copy

import numpy as np
from typing import Callable
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _minshap_one_permutation(model, X_train, y_train, patches, metric,
                              higher_is_better, V_null, e2_null,
                              early_stopping_patience, is_skorch,
                              show_patch_bar=False):
    n = X_train.shape[0]
    p = len(patches)
    model = copy.deepcopy(model)
    if is_skorch:
        from skorch.callbacks import EarlyStopping
        model.warm_start = True
        model.callbacks = [EarlyStopping(patience=early_stopping_patience)]

    pi_k = np.random.permutation(p)
    V_curr = V_null
    e2_curr = e2_null.copy()

    phi_k = np.zeros(p)
    sigma_k = np.zeros(p)

    for pos in tqdm(range(p), desc="patches", leave=False, disable=not show_patch_bar):
        patch_j = pi_k[pos]
        active_patches = pi_k[:pos + 1]

        X_new = np.zeros_like(X_train)
        for idx in active_patches:
            X_new[:, patches[idx]] = X_train[:, patches[idx]]

        fn_new = model.fit(X_new, y_train)
        preds_new = fn_new.predict(X_new)
        V_new = metric(y_train, preds_new)
        e2_new = (y_train - preds_new) ** 2

        phi_k[patch_j] = V_new - V_curr if higher_is_better else V_curr - V_new
        sigma_k[patch_j] = np.var(e2_curr - e2_new, ddof=1) / n

        V_curr = V_new
        e2_curr = e2_new

    return phi_k, sigma_k


def minshap(model, patches, X_train: np.ndarray, y_train: np.ndarray,
            metric: Callable, K: int = 100, alpha: float = 0.05,
            higher_is_better: bool = True, early_stopping_patience: int = 5,
            n_jobs: int = -1):
    """
    MinShap feature selection via minimum-SHAP marginal contributions.

    Parameters
    ----------
    model : fitted sklearn estimator or skorch NeuralNet
    patches : list of np.ndarray
        Index arrays from get_patches(), or [[0],[1],...] for individual pixels.
    X_train, y_train : np.ndarray
    metric : callable(y_true, y_pred) -> float
        Default convention matches sklearn (higher_is_better=True, e.g. accuracy_score).
    K : int
        Number of random permutations.
    alpha : float
        Significance level for rejection threshold t_j.
    higher_is_better : bool
        True  -> importance = V_new - V_curr  (e.g. R², accuracy)
        False -> importance = V_curr - V_new  (e.g. MSE, loss)
    early_stopping_patience : int
        Patience for skorch EarlyStopping callback. Ignored for sklearn models.
    n_jobs : int
        Parallelism passed to joblib. -1 = all cores, 1 = serial.

    Returns
    -------
    dict with keys 'phi_min', 't_j', 'rejected' — each an np.ndarray of shape (len(patches),)
    """
    try:
        from skorch import NeuralNet
        is_skorch = isinstance(model, NeuralNet)
    except ImportError:
        is_skorch = False

    # threads avoid torch/CUDA multiprocessing issues; processes are fine for sklearn
    prefer = "threads" if is_skorch else "processes"

    X_null = np.zeros_like(X_train)
    null_model = copy.deepcopy(model).fit(X_null, y_train)
    null_preds = null_model.predict(X_null)
    V_null = metric(y_train, null_preds)
    e2_null = (y_train - null_preds) ** 2

    if n_jobs == 1:
        results = []
        for _ in tqdm(range(K), desc="MinShap"):
            results.append(_minshap_one_permutation(
                model, X_train, y_train, patches, metric, higher_is_better,
                V_null, e2_null, early_stopping_patience, is_skorch,
                show_patch_bar=True
            ))
    else:
        results = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
                delayed(_minshap_one_permutation)(
                    model, X_train, y_train, patches, metric, higher_is_better,
                    V_null, e2_null, early_stopping_patience, is_skorch,
                    show_patch_bar=False
                )
                for _ in range(K)
            ),
            total=K,
            desc="MinShap",
        ))

    phi = np.array([r[0] for r in results])    # (K, len(patches))
    sigma = np.array([r[1] for r in results])  # (K, len(patches))

    phi_min = phi.min(axis=0)
    t = np.sqrt(-2 * np.log(alpha) * sigma[0, :])
    rejected = phi_min >= t

    return {'phi_min': phi_min, 't_j': t, 'rejected': rejected}
