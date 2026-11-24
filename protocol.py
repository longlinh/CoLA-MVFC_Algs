"""Evaluation protocol of the paper (Section 4.1): label layouts and local-corruption scenarios, deterministic in the seed.

    stratified_labels : r % of the reference pixels of every class, drawn at random (paper: r = 5; curves: 1, 3, 5, 10)
    region_labels     : r % of every class collected as (2*radius+1)^2 patches around random seeds (spatially clustered labels)
    cloud_blocks      : random 32 x 32 blocks until >= frac of the valid pixels are covered; the corrupted view gets the
                        value `level` + N(0, 1) noise in every band of the covered pixels, labelled pixels included; no mask
                        is given to the model (paper: hyperspectral view, frac = 0.2, 0.4, 0.6, level = 3)
    zero_blocks       : the same blocks as cloud_blocks for the same seed, values set to 0 (missing data, the band mean)
    haze_blocks       : thin cloud with soft edges: the same blocks blurred by a Gaussian (sigma 8 px) give an opacity
                        alpha in [0, level]; covered pixels (alpha >= 0.1 level) are mixed (1 - alpha) x + alpha (3 + N(0, 0.3^2))

Inputs follow one convention: y_full (N,) int with -1 outside the reference map, row-major pixels of an H x W image; every band is
z-scored with the mean and population standard deviation over the reference-map pixels (the corruption levels are in these units).
The paper uses seeds 42-51; the same seed drives the label layout and the corruption.
"""
from __future__ import annotations

import numpy as np

HAZE_SIGMA, HAZE_BRIGHT, HAZE_MIN = 8.0, 3.0, 0.1


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


def _blocks(valid: np.ndarray, height: int, width: int, frac: float, rng, block: int) -> np.ndarray:
    """Random block x block squares until >= frac of the valid pixels are covered; returns the covered valid pixels (N,) bool."""
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0, 1)")
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    dam = np.zeros((height, width), bool)
    goal = frac * valid.sum()
    tries = 0
    while (dam.ravel() & valid).sum() < goal and tries < 100000:
        tries += 1
        r0, c0 = rng.integers(0, max(height - block, 1)), rng.integers(0, max(width - block, 1))
        dam[r0:r0 + block, c0:c0 + block] = True
    return dam.ravel() & valid


def cloud_blocks(views: list, view_idx: int, valid: np.ndarray, height: int, width: int, frac: float, seed: int,
                 block: int = 32, level: float = 3.0):
    """Returns (corrupted views, covered mask (N,) bool). Only views[view_idx] is changed."""
    rng = np.random.default_rng(seed)
    dam = _blocks(valid, height, width, frac, rng, block)
    out = [v.copy() for v in views]
    n, d = int(dam.sum()), out[view_idx].shape[1]
    out[view_idx][dam] = level + rng.normal(size=(n, d))
    return out, dam


def zero_blocks(views: list, view_idx: int, valid: np.ndarray, height: int, width: int, frac: float, seed: int, block: int = 32):
    """Missing data: the blocks of cloud_blocks for the same seed, values set to 0. Returns (corrupted views, covered mask)."""
    dam = _blocks(valid, height, width, frac, np.random.default_rng(seed), block)
    out = [v.copy() for v in views]
    out[view_idx][dam] = 0.0
    return out, dam


def haze_blocks(views: list, view_idx: int, valid: np.ndarray, height: int, width: int, frac: float, seed: int,
                block: int = 32, level: float = 0.8):
    """Thin cloud with soft edges (maximum opacity `level`). Returns (corrupted views, covered mask (N,) bool)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    blocks = _blocks(valid, height, width, frac, rng, block)
    a = gaussian_filter(blocks.reshape(height, width).astype(float), HAZE_SIGMA).ravel()
    alpha = level * a / max(a.max(), 1e-12)
    dam = (alpha >= HAZE_MIN * level) & valid
    out = [v.copy() for v in views]
    n, d = int(dam.sum()), out[view_idx].shape[1]
    al = alpha[dam][:, None]
    out[view_idx][dam] = (1.0 - al) * out[view_idx][dam] + al * (HAZE_BRIGHT + rng.normal(0.0, 0.3, size=(n, d)))
    return out, dam
