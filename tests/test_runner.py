"""Tests for the experiment runner's config handling.

The runner is the thing that guarantees provenance, so its config expansion and
metadata stamping need to be correct even though the experiments themselves are
too slow to run here.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.run import _build_data, _build_model, _git_info, _set_in, expand_sweep, run_one


def test_set_in_nested():
    cfg = {"data": {"patch_size": 4}}
    _set_in(cfg, "data.patch_size", 7)
    assert cfg["data"]["patch_size"] == 7


def test_set_in_creates_missing_levels():
    cfg = {}
    _set_in(cfg, "a.b.c", 1)
    assert cfg == {"a": {"b": {"c": 1}}}


def test_set_in_top_level():
    cfg = {"seed": 0}
    _set_in(cfg, "seed", 5)
    assert cfg["seed"] == 5


def test_expand_sweep_absent_is_single_run():
    assert len(expand_sweep({"name": "x"})) == 1


def test_expand_sweep_cartesian_product():
    cfg = {"name": "x", "data": {"patch_size": 4}, "method": {"kind": "minshap"},
           "sweep": {"data.patch_size": [2, 4, 7], "method.kind": ["minshap", "max_p"]}}
    configs = expand_sweep(cfg)
    assert len(configs) == 6
    combos = {(c["data"]["patch_size"], c["method"]["kind"]) for c in configs}
    assert combos == {(p, m) for p in (2, 4, 7) for m in ("minshap", "max_p")}


def test_expand_sweep_labels_are_unique():
    cfg = {"name": "x", "sweep": {"seed": [0, 1, 2]}}
    labels = [c["_sweep_label"] for c in expand_sweep(cfg)]
    assert len(set(labels)) == 3


def test_expand_sweep_does_not_alias_configs():
    """Each expanded config must be an independent deepcopy."""
    cfg = {"name": "x", "data": {"n": 100}, "sweep": {"seed": [0, 1]}}
    a, b = expand_sweep(cfg)
    a["data"]["n"] = 999
    assert b["data"]["n"] == 100


def test_build_model_known_kinds():
    from sklearn.linear_model import LinearRegression, LogisticRegression

    assert isinstance(_build_model({"kind": "linear_regression"}), LinearRegression)
    assert isinstance(
        _build_model({"kind": "logistic_regression", "params": {"max_iter": 50}}),
        LogisticRegression,
    )


def test_build_model_unknown_kind_raises():
    with pytest.raises(KeyError):
        _build_model({"kind": "random_forest_of_dreams"})


def test_build_model_returns_unfitted():
    """Permutation methods refit themselves; a prefitted model would be wasted work."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    model = _build_model({"kind": "linear_regression"})
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_build_data_simulated_splits_and_carries_design():
    spec = {"source": "simulated", "design": "linear_additive", "n": 200,
            "test_frac": 0.25}
    Xtr, Xte, ytr, yte, patches, design = _build_data(spec, np.random.default_rng(0))
    assert len(Xtr) == 150 and len(Xte) == 50
    assert len(patches) == 10
    assert design.name == "linear_additive"
    np.testing.assert_array_equal(design.support, [0, 2, 4, 5, 6, 9])


def test_build_data_unknown_source_raises():
    with pytest.raises(KeyError):
        _build_data({"source": "imagenet"}, np.random.default_rng(0))


def test_git_info_shape():
    info = _git_info()
    assert set(info) == {"sha", "branch", "dirty"}


def test_run_one_writes_full_provenance(tmp_path):
    """A completed run must be reconstructible from its own output directory."""
    cfg = {
        "name": "unit",
        "seed": 3,
        "data": {"source": "simulated", "design": "linear_additive", "n": 300,
                 "noise": 0.1, "test_frac": 0.3},
        "model": {"kind": "linear_regression"},
        "method": {"kind": "minshap",
                   "params": {"K": 2, "alpha": 0.05, "loss": "squared_error",
                              "n_jobs": 1}},
    }
    out = run_one(cfg, out_root=tmp_path)

    for fname in ("config.yaml", "meta.json", "metrics.json", "arrays.npz"):
        assert (out / fname).exists(), f"missing {fname}"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["seed"] == 3
    assert meta["method"] == "minshap"
    assert meta["git"]["sha"] is not None
    assert meta["versions"]["numpy"]

    # the saved config must round-trip back to something runnable
    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["method"]["params"]["K"] == 2
    assert "_sweep_label" not in saved

    arrays = np.load(out / "arrays.npz")
    assert arrays["phi_min"].shape == (10,)
    assert arrays["phi"].shape == (2, 10)

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_true"] == 6
    assert 0.0 <= metrics["fdr"] <= 1.0


def test_run_one_dry_run_writes_nothing(tmp_path):
    cfg = {"name": "unit", "seed": 0,
           "data": {"source": "simulated", "design": "xor", "n": 50},
           "model": {"kind": "linear_regression"},
           "method": {"kind": "minshap", "params": {}}}
    run_one(cfg, out_root=tmp_path, dry_run=True)
    assert list(tmp_path.iterdir()) == []


def test_shipped_configs_are_valid_yaml_with_required_keys():
    """Guard against a config drifting out of the schema the runner expects."""
    config_dir = Path(__file__).resolve().parent.parent / "experiments" / "configs"
    configs = list(config_dir.glob("*.yaml"))
    assert configs, "no shipped configs found"
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        assert {"name", "data", "model", "method"} <= set(cfg), f"{path.name} missing keys"
        assert cfg["method"]["kind"] in {"minshap", "max_p", "loco", "saliency"}, path.name
        assert cfg["data"]["source"] in {"mnist", "simulated"}, path.name
        for c in expand_sweep(dict(cfg)):
            assert c["method"]["kind"] in {"minshap", "max_p", "loco", "saliency"}
