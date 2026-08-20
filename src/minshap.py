import copy

import numpy as np
from typing import Callable
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import norm

from .loss import local_neighborhood, local_mse


def _minshap_one_permutation(model, X_train, y_train, patches, metric,
                              higher_is_better, V_null, e2_null, null_preds,
                              early_stopping_patience, is_skorch,
                              show_patch_bar=False, local_idx=None):
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
    preds_curr = null_preds.copy()

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
        e2_new = (y_train - preds_new) ** 2

        if local_idx is not None:
            V_new = local_mse(preds_new, y_train, local_idx)
            local_e2_curr = (y_train[local_idx] - preds_curr[local_idx]) ** 2
            local_e2_new  = (y_train[local_idx] - preds_new[local_idx])  ** 2
            sigma_k[patch_j] = np.var(local_e2_curr - local_e2_new, ddof=1) / len(local_idx)
        else:
            V_new = metric(y_train, preds_new)
            sigma_k[patch_j] = np.var(e2_curr - e2_new, ddof=1) / n

        phi_k[patch_j] = V_curr - V_new  # reduction in MSE; higher_is_better handled below
        if local_idx is None and higher_is_better:
            phi_k[patch_j] = -phi_k[patch_j]

        V_curr = V_new
        e2_curr = e2_new
        preds_curr = preds_new

    return phi_k, sigma_k


def minshap(model, patches, X_train: np.ndarray, y_train: np.ndarray,
            metric: Callable = None, K: int = 100, alpha: float = 0.05,
            higher_is_better: bool = True, early_stopping_patience: int = 5,
            n_jobs: int = -1, local=False, x_S=None, k: int = 50):
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
    local : bool
        If True, evaluate on the fixed k-NN neighborhood of x_S instead of the
        full training set.  The neighborhood is precomputed once (using all features)
        and held constant across every permutation step, so phi = V_curr - V_new is
        always non-negative when the model improves on that neighborhood.
    x_S : np.ndarray, shape (n_features,)
        Query point for local mode.
    k : int
        Neighborhood size for local mode.

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

    # Precompute fixed local neighborhood once (all features, all permutations reuse it)
    local_idx = local_neighborhood(X_train, x_S, k) if local else None

    X_null = np.zeros_like(X_train)
    null_model = copy.deepcopy(model).fit(X_null, y_train)
    null_preds = null_model.predict(X_null)

    if local_idx is not None:
        V_null = local_mse(null_preds, y_train, local_idx)
    else:
        V_null = metric(y_train, null_preds)
    e2_null = (y_train - null_preds) ** 2

    shared_args = dict(
        model=model, X_train=X_train, y_train=y_train, patches=patches,
        metric=metric, higher_is_better=higher_is_better,
        V_null=V_null, e2_null=e2_null, null_preds=null_preds,
        early_stopping_patience=early_stopping_patience, is_skorch=is_skorch,
        local_idx=local_idx,
    )

    if n_jobs == 1:
        results = [
            _minshap_one_permutation(**shared_args, show_patch_bar=True)
            for _ in tqdm(range(K), desc="MinShap")
        ]
    else:
        results = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
                delayed(_minshap_one_permutation)(**shared_args, show_patch_bar=False)
                for _ in range(K)
            ),
            total=K,
            desc="MinShap",
        ))

    phi   = np.array([r[0] for r in results])   # (K, len(patches))
    sigma = np.array([r[1] for r in results])   # (K, len(patches))

    phi_min = phi.min(axis=0)
    t = np.sqrt(-2 * np.log(alpha) * sigma.mean(axis=0))
    rejected = phi_min >= t

    return {'phi_min': phi_min, 't_j': t, 'rejected': rejected}

def max_p(model, patches, X_train: np.ndarray, y_train: np.ndarray,
          metric: Callable = None, K: int = 100, alpha: float = 0.05,
          early_stopping_patience: int = 5, n_jobs: int = -1,
          local=False, x_S=None, k: int = 50):
    """
    Max-p feature selection: per-permutation two-sided z-tests on V_curr - V_new,
    reject when the max p-value across permutations is below alpha. Same params
    as minshap() except higher_is_better is dropped (|z| is sign-agnostic).
    """
    from scipy.stats import norm

    try:
        from skorch import NeuralNet
        is_skorch = isinstance(model, NeuralNet)
    except ImportError:
        is_skorch = False

    prefer = "threads" if is_skorch else "processes"
    local_idx = local_neighborhood(X_train, x_S, k) if local else None

    X_null = np.zeros_like(X_train)
    null_model = copy.deepcopy(model).fit(X_null, y_train)
    null_preds = null_model.predict(X_null)

    V_null = (local_mse(null_preds, y_train, local_idx) if local_idx is not None
              else metric(y_train, null_preds))
    e2_null = (y_train - null_preds) ** 2

    shared_args = dict(
        model=model, X_train=X_train, y_train=y_train, patches=patches,
        metric=metric, higher_is_better=False,
        V_null=V_null, e2_null=e2_null, null_preds=null_preds,
        early_stopping_patience=early_stopping_patience, is_skorch=is_skorch,
        local_idx=local_idx,
    )

    if n_jobs == 1:
        results = [_minshap_one_permutation(**shared_args, show_patch_bar=True)
                   for _ in tqdm(range(K), desc="Max-p")]
    else:
        results = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
                delayed(_minshap_one_permutation)(**shared_args, show_patch_bar=False)
                for _ in range(K)
            ),
            total=K, desc="Max-p",
        ))

    phi   = np.array([r[0] for r in results])
    sigma = np.array([r[1] for r in results])

    z = phi / np.sqrt(np.maximum(sigma, 1e-12))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    p_max = p.max(axis=0)

    return {'p_max': p_max, 'rejected': p_max < alpha}