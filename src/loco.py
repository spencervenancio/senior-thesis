import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def _loco_one_patch(patch_idx, model, X_train, X_test, y_train, y_test,
                    full_score, metric, higher_is_better,
                    early_stopping_patience, is_skorch):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[:, patch_idx] = 0
    X_te[:, patch_idx] = 0

    m = copy.deepcopy(model)
    if is_skorch:
        from skorch.callbacks import EarlyStopping
        m.warm_start = True
        m.callbacks = [EarlyStopping(patience=early_stopping_patience)]

    m.fit(X_tr, y_train)
    score = metric(y_test, m.predict(X_te))
    return full_score - score if higher_is_better else score - full_score


def loco(model, patches, X_train, X_test, y_train, y_test,
         full_score=None, metric=accuracy_score, higher_is_better=True,
         early_stopping_patience=5, n_jobs=-1):
    """
    LOCO (Leave One Covariate Out) importance over a list of patch index arrays.

    Each patch is zeroed out in both train and test, the model is retrained
    (warm-started for skorch), and the importance is the change in metric.

    Parameters
    ----------
    model : fitted sklearn estimator or skorch NeuralNet
    patches : list of np.ndarray
        Index arrays from get_patches(), or [[0],[1],...] for individual pixels.
    X_train, X_test : np.ndarray
    y_train, y_test : array-like
    full_score : float, optional
        Pre-computed baseline score. Computed from X_test if None.
    metric : callable(y_true, y_pred) -> float
        Default: accuracy_score.
    higher_is_better : bool
        True  -> importance = full_score - patch_score  (e.g. accuracy)
        False -> importance = patch_score - full_score  (e.g. loss)
    early_stopping_patience : int
        Patience for skorch EarlyStopping callback. Ignored for sklearn models.
    n_jobs : int
        Parallelism passed to joblib. -1 = all cores, 1 = serial.

    Returns
    -------
    np.ndarray of shape (len(patches),)
    """
    try:
        from skorch import NeuralNet
        is_skorch = isinstance(model, NeuralNet)
    except ImportError:
        is_skorch = False

    if full_score is None:
        full_score = metric(y_test, model.predict(X_test))

    # threads avoid torch/CUDA multiprocessing issues; processes are fine for sklearn
    prefer = "threads" if is_skorch else "processes"

    results = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(
            delayed(_loco_one_patch)(
                patch, model, X_train, X_test, y_train, y_test,
                full_score, metric, higher_is_better, early_stopping_patience, is_skorch
            )
            for patch in patches
        ),
        total=len(patches),
        desc="LOCO",
    ))
    return np.array(results)