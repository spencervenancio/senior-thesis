"""Tests for gradient attribution and the (uncalibrated) selection rules."""
import numpy as np
import pytest

from lfs.selection.saliency import (
    attribute,
    attribute_batch,
    dropout_curve,
    select_threshold,
    select_top_k,
)


def _linear_net(weights):
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    net = torch.nn.Linear(len(weights), 1, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([weights], dtype=torch.float32))
    return net


# --- selection rules --------------------------------------------------------

def test_select_threshold_keeps_above_frac_of_max():
    attr = np.array([1.0, 0.5, 0.04, 0.0])
    np.testing.assert_array_equal(
        select_threshold(attr, frac=0.05), [True, True, False, False]
    )


def test_select_threshold_empty_when_nothing_is_positive():
    assert not select_threshold(np.array([0.0, -1.0, -2.0])).any()


def test_select_threshold_scales_with_the_max_not_the_values():
    """The heuristic is relative, so a rescaled attribution selects identically."""
    attr = np.array([1.0, 0.5, 0.01])
    np.testing.assert_array_equal(
        select_threshold(attr, frac=0.1), select_threshold(attr * 1000, frac=0.1)
    )


def test_select_top_k_picks_the_k_largest():
    attr = np.array([0.1, 0.9, 0.5, 0.3])
    np.testing.assert_array_equal(select_top_k(attr, 2), [False, True, True, False])


def test_select_top_k_zero_selects_nothing():
    """`[-0:]` is `[0:]`, so a naive slice selects everything instead."""
    assert not select_top_k(np.array([0.1, 0.9, 0.5]), 0).any()


def test_select_top_k_beyond_length_selects_everything():
    assert select_top_k(np.array([0.1, 0.9]), 10).all()


# --- threshold diagnostic ---------------------------------------------------

def test_dropout_curve_is_monotone_non_increasing():
    rng = np.random.default_rng(0)
    _, counts = dropout_curve(np.abs(rng.normal(size=50)), steps=25)
    assert np.all(np.diff(counts) <= 0)


def test_dropout_curve_starts_at_full_support():
    attr = np.abs(np.random.default_rng(1).normal(size=12))
    thresholds, counts = dropout_curve(attr, steps=10)
    assert thresholds[0] == 0.0
    assert counts[0] == 12


def test_dropout_curve_degenerate_on_all_zero_attribution():
    thresholds, counts = dropout_curve(np.zeros(6), steps=4)
    assert len(thresholds) == len(counts) == 4
    assert not counts.any()


# --- attribution ------------------------------------------------------------

def test_saliency_of_a_linear_model_is_the_absolute_weight():
    net = _linear_net([3.0, -1.0, 0.0])
    attr = attribute(net, np.array([2.0, 1.0, 5.0]), method="saliency")
    np.testing.assert_allclose(attr, [3.0, 1.0, 0.0], atol=1e-5)


def test_input_x_gradient_of_a_linear_model_is_weight_times_input():
    net = _linear_net([3.0, -1.0, 2.0])
    x = np.array([2.0, 1.0, 0.0])
    attr = attribute(net, x, method="input_x_gradient", abs_value=False)
    np.testing.assert_allclose(attr, [6.0, -1.0, 0.0], atol=1e-5)


def test_abs_value_false_keeps_sign():
    net = _linear_net([-4.0, 1.0])
    signed = attribute(net, np.array([1.0, 1.0]), method="input_x_gradient",
                       abs_value=False)
    assert signed[0] < 0


def test_attribute_rejects_unknown_method():
    net = _linear_net([1.0, 1.0])
    with pytest.raises(ValueError, match="method must be one of"):
        attribute(net, np.array([1.0, 1.0]), method="nope")


def test_attribute_batch_shape():
    net = _linear_net([1.0, -2.0, 0.5])
    X = np.random.default_rng(2).normal(size=(5, 3))
    assert attribute_batch(net, X, method="saliency").shape == (5, 3)
