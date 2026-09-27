"""Synthetic co-registered multi-view scene: piecewise-constant class map on an H x W grid, one high-dimensional view and two
low-dimensional auxiliary views that separate only some classes. Deterministic in the seed; z-scored per band like the paper."""
import numpy as np


def make_scene(height=60, width=80, n_classes=5, dims=(60, 1, 2), seed=42):
    rng = np.random.default_rng(seed)
    # class map: Voronoi cells of random seeds, a few pixels outside the reference map (-1)
    pts = rng.uniform(0, 1, size=(4 * n_classes, 2)) * [height, width]
    cls = np.arange(len(pts)) % n_classes
    yy, xx = np.mgrid[0:height, 0:width]
    near = ((yy.ravel()[:, None] - pts[None, :, 0]) ** 2 + (xx.ravel()[:, None] - pts[None, :, 1]) ** 2).argmin(axis=1)
    y = cls[near].astype(np.int32)
    y[rng.random(y.size) < 0.1] = -1
    views = []
    for j, d in enumerate(dims):
        mu = rng.normal(0.0, 1.0, size=(n_classes, d))
        if j > 0:                                           # auxiliary views: only classes 0 and 1 stand out
            mu[2:] = mu[2:].mean(axis=0)
        scale = 3.0 if j == 0 else 0.6
        X = mu[np.maximum(y, 0)] + scale * rng.normal(size=(y.size, d))
        views.append((X - X.mean(axis=0)) / X.std(axis=0))
    return dict(views=views, y_full=y, n_classes=n_classes, height=height, width=width)
