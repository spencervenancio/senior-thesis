import numpy as np
import pytest

from lfs.metrics import pointwise
from lfs.metrics.local import Neighborhood, local_mse, local_neighborhood
from lfs.metrics.recovery import false_discovery_rate, power, recovery_scores

# --- pointwise losses -------------------------------------------------------

def test_squared_error():
    np.testing.assert_allclose(
        pointwise.squared_error([1.0, 2.0], [1.5, 2.0]), [0.25, 0.0]
    )


def test_zero_one_mean_is_one_minus_accuracy():
    y = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 9])
    assert pointwise.zero_one(y, pred).mean() == pytest.approx(0.25)


def test_cross_entropy_needs_proba():
    with pytest.raises(ValueError):
        pointwise.cross_entropy([0], [0], proba=None)


def test_cross_entropy_values():
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    loss = pointwise.cross_entropy([0, 1], [0, 1], proba=proba)
    np.testing.assert_allclose(loss, [-np.log(0.9), -np.log(0.8)])


def test_zero_one_is_scale_free_on_labels():
    """The bug this loss layer fixes: squared error on nominal labels is not a loss.

    Predicting class 9 instead of 0 is one mistake, exactly like predicting 1
    instead of 0 -- but squared error calls it 81x worse.
    """
    y = np.array([0, 0])
    near, far = np.array([1, 0]), np.array([9, 0])
    assert pointwise.zero_one(y, near).sum() == pointwise.zero_one(y, far).sum()
    assert pointwise.squared_error(y, near).sum() != pointwise.squared_error(y, far).sum()


def test_resolve_infers_classification_from_int_labels():
    fn, needs_proba = pointwise.resolve(None, np.array([0, 1, 2, 1, 0] * 10))
    assert fn is pointwise.zero_one and not needs_proba


def test_resolve_infers_regression_from_floats():
    fn, _ = pointwise.resolve(None, np.random.default_rng(0).normal(size=100))
    assert fn is pointwise.squared_error


def test_resolve_by_name_and_callable():
    assert pointwise.resolve("zero_one")[0] is pointwise.zero_one
    assert pointwise.resolve("cross_entropy")[1] is True
    custom = lambda a, b, c=None: np.zeros(len(a))  # noqa: E731
    assert pointwise.resolve(custom)[0] is custom


def test_resolve_rejects_unknown():
    with pytest.raises(KeyError):
        pointwise.resolve("nope")
    with pytest.raises(ValueError):
        pointwise.resolve(None, None)


# --- neighborhoods ----------------------------------------------------------

def test_neighborhood_includes_query_point_itself():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    nb = Neighborhood(X, X[5], k=3)
    assert 5 in nb.indices
    assert len(nb) == 3


def test_neighborhood_picks_nearest():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    nb = Neighborhood(X, np.array([5.0]), k=3)
    assert sorted(nb.indices) == [4, 5, 6]


def test_neighborhood_k_clamped_to_sample_size():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    assert len(Neighborhood(X, X[0], k=100)) == 4


def test_neighborhood_radius_is_furthest_distance():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    assert Neighborhood(X, np.array([5.0]), k=3).radius == pytest.approx(1.0)


def test_local_neighborhood_requires_query():
    with pytest.raises(ValueError):
        local_neighborhood(np.zeros((5, 2)), None)


def test_local_mse_restricts_to_indices():
    y = np.array([0.0, 0.0, 10.0])
    y_hat = np.array([1.0, 1.0, 0.0])
    assert local_mse(y_hat, y, np.array([0, 1])) == pytest.approx(1.0)
    assert local_mse(y_hat, y, np.array([2])) == pytest.approx(100.0)


# --- recovery scoring -------------------------------------------------------

def test_recovery_perfect():
    s = recovery_scores([0, 2], [0, 2], 5)
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["exact"]


def test_recovery_counts():
    # selected {0,1}, truth {1,2} -> tp=1, fp=1, fn=1
    s = recovery_scores([0, 1], [1, 2], 4)
    assert (s["false_positives"], s["false_negatives"]) == (1, 1)
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(0.5)


def test_recovery_accepts_masks_and_indices_alike():
    by_idx = recovery_scores([0, 2], [0, 2], 4)
    by_mask = recovery_scores(
        np.array([True, False, True, False]), np.array([True, False, True, False]), 4
    )
    assert by_idx == by_mask


def test_recovery_empty_selection():
    s = recovery_scores([], [0, 1], 4)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["n_selected"] == 0


def test_fdr_and_power():
    # selected {0,1,2}, truth {0,1} -> 1 of 3 selected is null
    assert false_discovery_rate([0, 1, 2], [0, 1], 5) == pytest.approx(1 / 3)
    assert power([0, 1, 2], [0, 1], 5) == pytest.approx(1.0)
    assert power([0], [0, 1], 5) == pytest.approx(0.5)


def test_fdr_of_empty_selection_is_zero():
    assert false_discovery_rate([], [0, 1], 5) == 0.0


def test_mask_length_validated():
    with pytest.raises(ValueError):
        recovery_scores(np.array([True, False]), [0], 5)
