"""Architecture parameters must survive the adaptive module rebuild.

_AdaptiveMixin reconstructs module_ on every fit so a single estimator can be
handed to minshap, which refits on inputs of changing width. The rebuild has to
carry the module__* kwargs through: if it does not, every estimator silently
gets the module default and a width or depth sweep becomes a no-op that looks
like a real (flat) result.
"""
import numpy as np
import pytest

pytest.importorskip("skorch")

from lfs.models import neural_net, neural_net_regressor  # noqa: E402


def _data(n=32, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    return X, X[:, 0] + rng.normal(0, 0.1, n)


def _n_params(net):
    return sum(q.numel() for q in net.module_.parameters())


@pytest.mark.parametrize("hidden", [4, 16, 64])
def test_module_hidden_is_respected(hidden):
    X, y = _data()
    reg = neural_net_regressor(module__hidden=hidden, module__depth=1, max_epochs=1)
    reg.fit(X, y)
    assert reg.module_.net[0].out_features == hidden


def test_width_changes_parameter_count():
    """The regression test for the sweep: more width must mean more parameters."""
    X, y = _data()
    counts = []
    for hidden in (4, 16, 64):
        reg = neural_net_regressor(module__hidden=hidden, module__depth=1, max_epochs=1)
        reg.fit(X, y)
        counts.append(_n_params(reg))
    assert counts[0] < counts[1] < counts[2]
    assert counts[2] > 5 * counts[0]


def test_depth_changes_parameter_count():
    X, y = _data()
    shallow = neural_net_regressor(module__hidden=8, module__depth=1, max_epochs=1)
    deep = neural_net_regressor(module__hidden=8, module__depth=4, max_epochs=1)
    shallow.fit(X, y)
    deep.fit(X, y)
    assert _n_params(deep) > _n_params(shallow)


def test_defaults_still_build_without_module_kwargs():
    X, y = _data()
    reg = neural_net_regressor(max_epochs=1)
    reg.fit(X, y)
    assert _n_params(reg) > 0


def test_classifier_accepts_module_kwargs():
    X, _ = _data()
    y = (X[:, 0] > 0).astype(int)
    clf = neural_net(module__hidden=6, max_epochs=1)
    clf.fit(X, y)
    assert clf.module_.net[0].out_features == 6


def test_rebuild_on_changing_width_keeps_module_kwargs():
    """minshap refits on masked inputs; hidden must not reset on the rebuild."""
    X, y = _data(p=6)
    reg = neural_net_regressor(module__hidden=5, module__depth=1, max_epochs=1)
    reg.fit(X, y)
    assert reg.module_.net[0].out_features == 5
    reg.fit(X[:, :3], y)                      # narrower input forces a rebuild
    assert reg.module_.net[0].in_features == 3
    assert reg.module_.net[0].out_features == 5
