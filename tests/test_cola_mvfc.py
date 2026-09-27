"""Tests: structural guarantees of the paper on a synthetic scene, determinism, and (when the authors' `ds` library and experiment
harness are importable) numerical parity with the implementation used for the paper's experiments.

    pytest tests/ -q
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cola_mvfc import NO_LABEL, PAPER_PARAMS, cola_mvfc, holdout_split, select_params  # noqa: E402
from protocol import cloud_blocks, region_labels, stratified_labels  # noqa: E402
from synthetic import make_scene  # noqa: E402

SEED = 42


@pytest.fixture(scope="module")
def scene():
    sc = make_scene(seed=SEED)
    sc["valid"] = sc["y_full"] >= 0
    sc["lab"] = stratified_labels(sc["y_full"], sc["n_classes"], 0.05, SEED)
    return sc


def _fit(sc, views=None, **kw):
    p = {**PAPER_PARAMS, "theta": 1.0, "theta_g": 0.8, **kw}
    return cola_mvfc(views or sc["views"], sc["lab"], sc["n_classes"], seed=SEED, valid_mask=sc["valid"], **p)


def test_objective_non_increasing_and_labels_fixed(scene):
    for kw in (dict(), dict(theta=0.0), dict(theta_g=0.0), dict(metric="euclid"), dict(view_prior="uniform"), dict(m=1.5)):
        r = _fit(scene, **kw)
        J = np.asarray(r.objective_history)
        assert np.all(np.diff(J) <= 1e-9 * np.maximum(1.0, np.abs(J[:-1]))), kw
        lab = scene["lab"] >= 0
        assert np.array_equal(r.labels[lab], scene["lab"][lab])
        assert np.allclose(r.membership.sum(axis=1), 1.0) and r.membership.min() >= 0.0


def test_label_anchoring_unit(scene):
    """Proposition 1: the guidance cost has mean 1 on the labelled pixels (out-of-fold posteriors) in every view."""
    r = _fit(scene)
    lab = scene["lab"] >= 0
    y = scene["lab"][lab]
    for P, S in zip(r.posteriors, r.S_g):
        assert abs(np.mean(-np.log(P[lab][np.arange(y.size), y])) / S - 1.0) < 1e-12
    assert abs(r.view_weights.sum() - 1.0) < 1e-12


def test_deterministic(scene):
    a, b = _fit(scene), _fit(scene)
    assert np.array_equal(a.membership, b.membership) and a.objective_history == b.objective_history


def test_holdout_and_selection(scene):
    lab_tr, held = holdout_split(scene["lab"], scene["n_classes"], 0.2, SEED)
    assert held.sum() > 0 and np.all(lab_tr[held] == NO_LABEL) and np.all(scene["lab"][held] >= 0)
    th, tg, table = select_params(scene["views"], scene["lab"], scene["n_classes"], thetas=(0.5, 2.0), theta_gs=(0.5, 0.8), seed=SEED,
                                  valid_mask=scene["valid"], **PAPER_PARAMS)
    assert len(table) == 4 and (th, tg) in {(t["theta"], t["theta_g"]) for t in table}


def test_protocol_shapes(scene):
    y, C = scene["y_full"], scene["n_classes"]
    reg = region_labels(y, C, scene["width"], 0.05, SEED)
    assert np.all(y[reg >= 0] == reg[reg >= 0])
    cv, dam = cloud_blocks(scene["views"], 0, scene["valid"], scene["height"], scene["width"], 0.6, SEED)
    assert dam.sum() >= 0.6 * scene["valid"].sum() and np.array_equal(cv[1], scene["views"][1])


# ------------------------------------------------------------------ parity with the authors' implementation (skipped elsewhere)
ds = pytest.importorskip("ds.clustering", reason="authors' ds library not installed")


@pytest.mark.parametrize("kw", [dict(), dict(theta=0.0), dict(theta_g=0.0), dict(metric="euclid"), dict(view_prior="uniform"),
                                dict(theta=4.0, theta_g=0.95), dict(cloud=True)])
def test_parity_with_ds(scene, kw):
    kw = dict(kw)
    views = scene["views"]
    if kw.pop("cloud", False):
        views, _ = cloud_blocks(views, 0, scene["valid"], scene["height"], scene["width"], 0.6, SEED)
    p = {**PAPER_PARAMS, "theta": 1.0, "theta_g": 0.8, **kw}
    ours = cola_mvfc(views, scene["lab"], scene["n_classes"], seed=SEED, valid_mask=scene["valid"], **p)
    ref = ds.CoLA_MVFC(n_clusters=scene["n_classes"], device="CPU", lam=float("inf"),
                       classifier="lda", **p).fit(views, scene["lab"], seed=SEED, valid_mask=scene["valid"]).result
    assert ours.n_iter == ref.n_iter
    assert np.max(np.abs(ours.membership - ref.membership)) < 1e-8
    assert np.allclose(ours.objective_history, ref.objective_history, rtol=1e-10, atol=1e-8)
    assert np.allclose(ours.view_weights, ref.view_prior, rtol=1e-12)


def test_protocol_parity_with_harness(scene):
    harness = os.environ.get("COLA_HARNESS")                      # path to the authors' experiment code (src/), if available
    if not harness or not Path(harness).exists():
        pytest.skip("experiment harness not available (set COLA_HARNESS)")
    sys.path.insert(0, str(harness))
    from gate.degrade import cloud_blocks as h_cloud
    from gate.label_regimes import region as h_region
    from gate.label_regimes import stratified as h_strat
    sc = dict(y_full=scene["y_full"], n_clusters=scene["n_classes"], width=scene["width"], height=scene["height"],
              mask=scene["valid"], views=scene["views"])
    assert np.array_equal(h_strat(sc, 0.05, SEED), stratified_labels(scene["y_full"], scene["n_classes"], 0.05, SEED))
    assert np.array_equal(h_region(sc, 0.05, SEED), region_labels(scene["y_full"], scene["n_classes"], scene["width"], 0.05, SEED))
    h = h_cloud(sc, 0, 0.6, SEED, block=32, level=3.0)
    ours, dam = cloud_blocks(scene["views"], 0, scene["valid"], scene["height"], scene["width"], 0.6, SEED)
    assert np.array_equal(h["damaged"], dam) and all(np.array_equal(a, b) for a, b in zip(h["views"], ours))
