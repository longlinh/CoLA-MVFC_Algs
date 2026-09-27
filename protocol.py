"""Evaluation protocol of the paper (Section 4.1): label layouts and the local-corruption scenario, deterministic in the seed.

    stratified_labels : r % of the reference pixels of every class, drawn at random (paper: r = 5; curves: 1, 3, 5, 10)
    region_labels     : r % of every class collected as (2*radius+1)^2 patches around random seeds (spatially clustered labels)
    cloud_blocks      : random 32 x 32 blocks until >= frac of the valid pixels are covered; the corrupted view gets the
                        value `level` + N(0, 1) noise in every band of the covered pixels, labelled pixels included; no mask
                        is given to the model (paper: hyperspectral view, frac = 0.6, level = 3 on z-scored bands)

Inputs follow one convention: y_full (N,) int with -1 outside the reference map, row-major pixels of an H x W image.
The paper uses seeds 42-51; the same seed drives the label layout and the corruption.
"""
from __future__ import annotations

import numpy as np


def stratified_labels(y_full: np.ndarray, n_classes: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lab = np.full(len(y_full), -1, dtype=np.int32)
    for c in range(n_classes):
        idx = np.flatnonzero(y_full == c)
        if len(idx) == 0:
            continue
        n = max(1, int(round(ratio * len(idx))))
        lab[rng.choice(idx, n, replace=False)] = c
    return lab


def region_labels(y_full: np.ndarray, n_classes: int, width: int, ratio: float, seed: int, radius: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = len(y_full)
    yy, xx = np.divmod(np.arange(N), width)
    lab = np.full(N, -1, dtype=np.int32)
    for c in range(n_classes):
        idx = np.flatnonzero(y_full == c)
        if len(idx) == 0:
            continue
        target = max(1, int(round(ratio * len(idx))))
        got = tries = 0
        while got < target and tries < 10000:
            tries += 1
            s = int(rng.choice(idx))
            if lab[s] == c:
                continue
            sel = (np.abs(yy - yy[s]) <= radius) & (np.abs(xx - xx[s]) <= radius) & (y_full == c) & (lab < 0)
            lab[sel] = c
            got += int(sel.sum())
    return lab


def cloud_blocks(views: list, view_idx: int, valid: np.ndarray, height: int, width: int, frac: float, seed: int,
                 block: int = 32, level: float = 3.0):
    """Returns (corrupted views, damaged mask (N,) bool). Only views[view_idx] is changed."""
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    dam = np.zeros((height, width), bool)
    goal = frac * valid.sum()
    tries = 0
    while (dam.ravel() & valid).sum() < goal and tries < 100000:
        tries += 1
        r0, c0 = rng.integers(0, max(height - block, 1)), rng.integers(0, max(width - block, 1))
        dam[r0:r0 + block, c0:c0 + block] = True
    dam = dam.ravel() & valid
    out = [v.copy() for v in views]
    n, d = int(dam.sum()), out[view_idx].shape[1]
    out[view_idx][dam] = level + rng.normal(size=(n, d))
    return out, dam
