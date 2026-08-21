"""Behavioral tests for the selection methods.

These deliberately use fast linear models rather than neural nets: the point is
to pin down the *estimator's* conventions and arithmetic, not to benchmark a
network. Anything slow or stochastic belongs in an experiment, not a test.
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from lfs.data import simulated
from lfs.data.patches import single_features
from lfs.metrics.recovery import recovery_scores
from lfs.selection import loco, max_p, minshap
from lfs.selection._permutation import run_permutations


@pytest.fixture
def linear_design():
    return simulated.linear_additive(n=400, noise=0.1, rng=np.random.default_rng(0))


# --- API guards -------------------------------------------------------------

def test_minshap_rejects_removed_kwargs(linear_design):
    """Old calls must fail loudly, not silently change meaning."""
    from sklearn.metrics import accuracy_score

    with pytest.raises(TypeError, match="loss="):
        minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, metric=accuracy_score)
    with pytest.raises(TypeError, match="loss="):
        minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, higher_is_better=True)


def test_max_p_rejects_removed_kwargs(linear_design):
    with pytest.raises(TypeError, match="loss="):
        max_p(LinearRegression(), single_features(10), linear_design.X,
              linear_design.y, metric=lambda a, b: 0.0)


def test_empty_patches_rejected(linear_design):
    with pytest.raises(ValueError, match="empty"):
        minshap(LinearRegression(), [], linear_design.X, linear_design.y, K=2)


def test_local_requires_enough_neighbors(linear_design):
    """A 1-point neighborhood cannot support a ddof=1 variance estimate."""
    with pytest.raises(ValueError, match="need >= 2"):
        minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, K=2, local=True, x_S=linear_design.X[0], k=1)


def test_cross_entropy_requires_predict_proba(linear_design):
    with pytest.raises(ValueError, match="predict_proba"):
        minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, loss="cross_entropy", K=2)


# --- sign conventions -------------------------------------------------------

def test_phi_is_positive_for_signal_features(linear_design):
    """phi = reduction in loss, so relevant features must score above null ones."""
    stats = run_permutations(
        LinearRegression(), single_features(10), linear_design.X, linear_design.y,
        loss="squared_error", K=4, n_jobs=1, rng=np.random.default_rng(0),
    )
    mean_phi = stats["phi"].mean(axis=0)
    # x5 and x6 have coefficient 9; x2/x4 are pure noise
    assert mean_phi[4] > 0 and mean_phi[5] > 0
    assert mean_phi[4] > mean_phi[1]
    assert mean_phi[5] > mean_phi[3]


def test_phi_ordering_follows_effect_size(linear_design):
    """Coefficients are 2,1,9,9,3,1 on x1,x3,x5,x6,x7,x10 -- x5/x6 must dominate."""
    stats = run_permutations(
        LinearRegression(), single_features(10), linear_design.X, linear_design.y,
        loss="squared_error", K=6, n_jobs=1, rng=np.random.default_rng(0),
    )
    mean_phi = stats["phi"].mean(axis=0)
    assert mean_phi[4] > mean_phi[0]   # x5 (9) beats x1 (2)
    assert mean_phi[5] > mean_phi[6]   # x6 (9) beats x7 (3)


def test_sigma_is_non_negative(linear_design):
    stats = run_permutations(
        LinearRegression(), single_features(10), linear_design.X, linear_design.y,
        loss="squared_error", K=3, n_jobs=1, rng=np.random.default_rng(0),
    )
    assert (stats["sigma"] >= 0).all()


def test_shapes(linear_design):
    K, p = 3, 10
    stats = run_permutations(
        LinearRegression(), single_features(p), linear_design.X, linear_design.y,
        loss="squared_error", K=K, n_jobs=1, rng=np.random.default_rng(0),
    )
    assert stats["phi"].shape == (K, p)
    assert stats["sigma"].shape == (K, p)


# --- reproducibility --------------------------------------------------------

def test_same_rng_gives_same_result(linear_design):
    kwargs = dict(loss="squared_error", K=3, n_jobs=1)
    a = minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, rng=np.random.default_rng(7), **kwargs)
    b = minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, rng=np.random.default_rng(7), **kwargs)
    np.testing.assert_allclose(a["phi_min"], b["phi_min"])
    np.testing.assert_array_equal(a["rejected"], b["rejected"])


def test_different_rng_gives_different_permutations(linear_design):
    kwargs = dict(loss="squared_error", K=3, n_jobs=1)
    a = minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, rng=np.random.default_rng(0), **kwargs)
    b = minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, rng=np.random.default_rng(1), **kwargs)
    assert not np.allclose(a["phi"], b["phi"])


# --- result object ----------------------------------------------------------

def test_result_is_dict_compatible(linear_design):
    r = minshap(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, loss="squared_error", K=2, n_jobs=1,
                rng=np.random.default_rng(0))
    assert set(["phi_min", "t_j", "rejected"]).issubset(r)   # old-style access
    assert r.n_selected == int(r["rejected"].sum())          # new-style access
    np.testing.assert_array_equal(r.selected, np.flatnonzero(r["rejected"]))


def test_threshold_grows_with_smaller_alpha(linear_design):
    """t = sqrt(-2 log(alpha) sigma) must be stricter for smaller alpha."""
    common = dict(loss="squared_error", K=3, n_jobs=1)
    loose = minshap(LinearRegression(), single_features(10), linear_design.X,
                    linear_design.y, alpha=0.20, rng=np.random.default_rng(0), **common)
    tight = minshap(LinearRegression(), single_features(10), linear_design.X,
                    linear_design.y, alpha=0.01, rng=np.random.default_rng(0), **common)
    assert (tight["t_j"] >= loose["t_j"]).all()
    assert tight.n_selected <= loose.n_selected


# --- max_p ------------------------------------------------------------------

def test_max_p_degenerate_sigma_is_conservative():
    """sigma == 0 carries no evidence; it must not manufacture significance.

    The previous implementation floored sigma at 1e-12, turning a zero-variance
    permutation into an enormous z and a p-value of 0 -- the opposite of the
    conservative reading.
    """
    X = np.zeros((60, 3))
    y = np.zeros(60)
    r = max_p(LinearRegression(), single_features(3), X, y,
              loss="squared_error", K=2, n_jobs=1, rng=np.random.default_rng(0))
    assert not r["rejected"].any()
    np.testing.assert_allclose(r["p_max"], 1.0)


def test_max_p_p_values_in_unit_interval(linear_design):
    r = max_p(LinearRegression(), single_features(10), linear_design.X,
              linear_design.y, loss="squared_error", K=3, n_jobs=1,
              rng=np.random.default_rng(0))
    assert ((r["p_max"] >= 0) & (r["p_max"] <= 1)).all()


def test_one_sided_is_no_less_powerful(linear_design):
    common = dict(loss="squared_error", K=3, n_jobs=1)
    two = max_p(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, two_sided=True, rng=np.random.default_rng(0), **common)
    one = max_p(LinearRegression(), single_features(10), linear_design.X,
                linear_design.y, two_sided=False, rng=np.random.default_rng(0), **common)
    # one-sided p is half the two-sided p wherever z > 0
    assert one.n_selected >= two.n_selected


# --- LOCO -------------------------------------------------------------------

def test_loco_ranks_signal_above_noise(linear_design):
    d = linear_design
    split = 300
    model = LinearRegression().fit(d.X[:split], d.y[:split])
    imp = loco(model, single_features(10), d.X[:split], d.X[split:],
               d.y[:split], d.y[split:], loss="squared_error", n_jobs=1)
    assert imp.shape == (10,)
    # x5, x6 (coef 9) must matter far more than the null x2, x4
    assert imp[4] > imp[1] and imp[5] > imp[3]


def test_loco_importance_is_near_zero_for_null_features(linear_design):
    d = linear_design
    split = 300
    model = LinearRegression().fit(d.X[:split], d.y[:split])
    imp = loco(model, single_features(10), d.X[:split], d.X[split:],
               d.y[:split], d.y[split:], loss="squared_error", n_jobs=1)
    signal_scale = imp[[4, 5]].mean()
    for null_j in [1, 3, 7, 8]:
        assert abs(imp[null_j]) < 0.05 * signal_scale


def test_loco_classification_runs():
    d = simulated.logistic_bernoulli(n=400, rng=np.random.default_rng(0))
    model = LogisticRegression(max_iter=200).fit(d.X[:300], d.y[:300])
    imp = loco(model, single_features(10), d.X[:300], d.X[300:],
               d.y[:300], d.y[300:], loss="zero_one", n_jobs=1)
    assert imp.shape == (10,) and np.isfinite(imp).all()


# --- end-to-end recovery ----------------------------------------------------

@pytest.mark.slow
def test_minshap_recovers_linear_support():
    """The headline claim: on a well-specified linear design, find S* and not more."""
    d = simulated.linear_additive(n=1500, noise=0.1, rng=np.random.default_rng(0))
    r = minshap(LinearRegression(), single_features(10), d.X, d.y,
                loss="squared_error", K=10, alpha=0.05, n_jobs=1,
                rng=np.random.default_rng(0))
    scores = recovery_scores(r["rejected"], d.support, 10)
    # strong recall on the large-coefficient features, no false discoveries
    assert scores["false_positives"] == 0, f"selected null features: {r.selected}"
    assert r["rejected"][4] and r["rejected"][5], "must find the coef-9 features"


@pytest.mark.slow
def test_minshap_selects_nothing_under_pure_noise():
    """Type-I error check: y independent of X means nothing should be selected."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 6))
    y = rng.standard_normal(600)
    r = minshap(LinearRegression(), single_features(6), X, y,
                loss="squared_error", K=10, alpha=0.05, n_jobs=1, rng=rng)
    assert r.n_selected == 0, f"false discoveries under the null: {r.selected}"


@pytest.mark.slow
def test_local_mode_runs_and_restricts_evaluation(linear_design):
    d = linear_design
    r = minshap(LinearRegression(), single_features(10), d.X, d.y,
                loss="squared_error", K=4, n_jobs=1, local=True, x_S=d.X[0], k=60,
                rng=np.random.default_rng(0))
    assert r["n_eval"] == 60
    assert len(r["eval_idx"]) == 60
    assert r["phi_min"].shape == (10,)


def test_select_top_k_breaks_ties_fairly():
    """argsort ties by index, which fabricates recall on zero attributions.

    Six features tied at zero, two slots to fill: each tied feature should be
    picked about 2/6 of the time, not deterministically by position.
    """
    from lfs.selection.saliency import select_top_k

    attr = np.array([5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0])
    rng = np.random.default_rng(0)
    counts = np.zeros(10)
    for _ in range(4000):
        counts += select_top_k(attr, 6, rng=rng)
    assert (counts[[0, 1, 8, 9]] == 4000).all(), "the four clear winners always win"
    tied = counts[2:8] / 4000
    assert abs(tied.mean() - 2 / 6) < 0.02
    assert tied.max() - tied.min() < 0.05, "no tied feature is systematically favoured"


def test_select_top_k_is_reproducible_with_a_seed():
    from lfs.selection.saliency import select_top_k

    attr = np.zeros(10)
    a = select_top_k(attr, 4, rng=np.random.default_rng(7))
    b = select_top_k(attr, 4, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
