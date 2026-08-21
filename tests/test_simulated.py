import numpy as np
import pytest

from lfs.data import simulated


@pytest.mark.parametrize("name", sorted(simulated.REGISTRY))
def test_designs_are_reproducible(name):
    """Same seed, same data -- the property the old np.random.randn code lacked."""
    a = simulated.make(name, n=200, rng=np.random.default_rng(0))
    b = simulated.make(name, n=200, rng=np.random.default_rng(0))
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.y, b.y)


@pytest.mark.parametrize("name", sorted(simulated.REGISTRY))
def test_different_seeds_differ(name):
    a = simulated.make(name, n=200, rng=np.random.default_rng(0))
    b = simulated.make(name, n=200, rng=np.random.default_rng(1))
    assert not np.allclose(a.X, b.X)


@pytest.mark.parametrize("name", sorted(simulated.REGISTRY))
def test_shapes_and_support_bounds(name):
    d = simulated.make(name, n=150, rng=np.random.default_rng(0))
    assert d.X.shape[0] == 150
    assert d.y.shape == (150,)
    assert d.X.shape[1] == d.n_features
    assert d.support.min() >= 0
    assert d.support.max() < d.n_features


def test_linear_additive_support_matches_formula():
    """y = 2x1 + x3 + 9x5 + 9x6 + 3x7 + x10 -> 0-based indices 0,2,4,5,6,9."""
    d = simulated.linear_additive(n=100, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(d.support, [0, 2, 4, 5, 6, 9])


def test_null_features_are_uncorrelated_with_y():
    """x2, x4, x8, x9 carry no signal in the linear design."""
    d = simulated.linear_additive(n=20000, noise=0.1, rng=np.random.default_rng(0))
    for null_j in [1, 3, 7, 8]:
        corr = np.corrcoef(d.X[:, null_j], d.y)[0, 1]
        assert abs(corr) < 0.05, f"x{null_j+1} should be null, got corr={corr:.3f}"
    for sig_j in d.support:
        corr = np.corrcoef(d.X[:, sig_j], d.y)[0, 1]
        assert abs(corr) > 0.01


def test_conditional_interaction_local_support_tracks_gates():
    """S*(x) must follow the gating variables x3 and x8."""
    d = simulated.conditional_interaction(n=100, rng=np.random.default_rng(0))

    x = np.zeros(10)
    x[2], x[7] = 1.0, 1.0        # both gates positive -> pairs (x1,x2), (x6,x7)
    np.testing.assert_array_equal(d.local_support(x), [0, 1, 2, 5, 6, 7])

    x[2], x[7] = -1.0, -1.0      # both negative -> pairs (x4,x5), (x9,x10)
    np.testing.assert_array_equal(d.local_support(x), [2, 3, 4, 7, 8, 9])

    x[2], x[7] = 1.0, -1.0       # mixed
    np.testing.assert_array_equal(d.local_support(x), [0, 1, 2, 7, 8, 9])


def test_local_support_actually_varies():
    """The design is only interesting if S*(x) is not constant."""
    d = simulated.conditional_interaction(n=200, rng=np.random.default_rng(0))
    supports = {tuple(d.local_support(x)) for x in d.X[:50]}
    assert len(supports) > 1, "conditional design must have point-dependent support"


def test_additive_local_support_is_constant():
    d = simulated.linear_additive(n=50, rng=np.random.default_rng(0))
    for x in d.X[:10]:
        np.testing.assert_array_equal(d.local_support(x), d.support)


def test_support_mask():
    d = simulated.linear_additive(n=50, rng=np.random.default_rng(0))
    mask = d.support_mask()
    assert mask.dtype == bool and mask.sum() == 6
    assert np.flatnonzero(mask).tolist() == d.support.tolist()


def test_logistic_bernoulli_is_binary():
    d = simulated.logistic_bernoulli(n=300, rng=np.random.default_rng(0))
    assert set(np.unique(d.y)).issubset({0, 1})


def test_xor_is_marginally_uninformative():
    """Neither XOR feature correlates with y alone -- the point of the design."""
    d = simulated.xor(n=20000, noise=0.01, rng=np.random.default_rng(0))
    for j in range(2):
        assert abs(np.corrcoef(d.X[:, j], d.y)[0, 1]) < 0.05


def test_unknown_design_raises():
    with pytest.raises(KeyError):
        simulated.make("not_a_design")
