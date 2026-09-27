"""CoLA-MVFC: collaborative label-anchored semi-supervised multi-view fuzzy clustering.

NumPy reference implementation written to be read next to the equations of the paper (Section 3).
One shared fuzzy membership U (N x C) is estimated from the label-anchored assignment costs of all views:

    J(U, V) = sum_j a_j [ sum_i nu_i sum_k u_ik^m e_ik^(j) + sum_k rho_k || v_k^(j) - xbar_k^(j) ||^2_{sigma^(j)_k} ]
    e^(j)   = delta^(j) + alpha g^(j),                         alpha = theta_g / (1 - theta_g)
    delta^(j)_ik = sum_d (x_id - v_kd)^2 sigma^(j)_kd,         sigma^(j)_kd = 1 / (d_j s^(j)_kd)   (per-class variance on labels)
    g^(j)_ik = -ln p^(j)_ik / S_g^(j),                         p^(j): shrinkage LDA of view j, out-of-fold on the labelled pixels
    S_g^(j)  = mean_{i in L} -ln p^(j)_{i, y_i}                 (label anchoring of the guidance cost)
    a_j      proportional to 1 / Lambda_j                       (leave-one-out log-loss of view j at the class means, labels only)
    rho_k    = theta * N_nu / |L| * n_k                         (class-mean constraint weight)

Closed-form updates for every m > 1 (U: Lagrange; V: stationarity); labelled rows of U are fixed to their one-hot labels.
U-step  u_ik  proportional to D_ik^(-1/(m-1)),  D_ik = sum_j a_j e_ik^(j)
V-step  v_k^(j) = (sum_i nu_i u_ik^m x_i^(j) + rho_k xbar_k^(j)) / (sum_i nu_i u_ik^m + rho_k)
Each step minimises J exactly in its block, so J is non-increasing (Theorem 1 of the paper).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

NO_LABEL = -1
_EPS = 1e-12

# Settings used in the paper (Section 4.1); theta and theta_g are selected on held-out labels (select_params).
PAPER_PARAMS = dict(m=2.0, metric="diag", view_prior="labels", epsilon=1e-4, max_iter=300, objective_tol=1e-6)
PAPER_GRID = dict(thetas=(0.5, 1.0, 2.0, 4.0), theta_gs=(0.5, 0.8, 0.95))


@dataclass
class Result:
    membership: np.ndarray                 # (N, C) shared fuzzy membership U
    labels: np.ndarray                     # (N,) argmax_k u_ik (cluster k = class k)
    centroids: list                        # M arrays (C, d_j)
    view_weights: np.ndarray               # (M,) a_j
    log_loss: np.ndarray | None            # (M,) Lambda_j (None when view_prior='uniform')
    posteriors: list                       # M arrays (N, C) p^(j) (empty when theta_g = 0)
    S_g: np.ndarray                        # (M,) guidance unit per view (nan when theta_g = 0)
    Sinv: list                             # M arrays (C, d_j) sigma^(j)
    alpha: float
    objective_history: list = field(default_factory=list)   # [t] = J(U^{t+1}, V^t); last = J at the returned (U, V)
    n_iter: int = 0
    converged: bool = False
    time: float = 0.0


# ----------------------------------------------------------------------------- closed-form blocks
def update_u(D: np.ndarray, m: float) -> np.ndarray:
    """u_ik proportional to D_ik^(-1/(m-1)) along the last axis; D > 0 (normalised by the row minimum to avoid overflow)."""
    Dn = D / D.min(axis=-1, keepdims=True)
    P = Dn ** (-1.0 / (m - 1.0))
    return P / P.sum(axis=-1, keepdims=True)


def update_v(X: np.ndarray, w: np.ndarray, xbar: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """v_k = (sum_i w_ik x_i + rho_k xbar_k) / (sum_i w_ik + rho_k); X (N,d), w (N,C) >= 0, rho (C,) >= 0."""
    s = w.sum(axis=0) + rho
    s = s + (s <= 0) * _EPS
    return (w.T @ X + rho[:, None] * xbar) / s[:, None]


def assignment_distance(X: np.ndarray, V: np.ndarray, Sinv: np.ndarray) -> np.ndarray:
    """delta_ik = sum_d (x_id - v_kd)^2 Sinv_kd, (N, C), clipped at 0."""
    d2 = (X * X) @ Sinv.T - 2.0 * (X @ (V * Sinv).T) + ((V * V) * Sinv).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


# ----------------------------------------------------------------------------- label anchoring
def class_scales(views, lab_mask, y_lab, counts, nu, metric="diag"):
    """Class means xbar^(j) (C, d_j), scatter S_d^(j) and metric Sinv^(j) (C, d_j).

    metric 'diag'  : Sinv_kd = 1 / (d_j s_kd), s_kd = per-dimension variance of class k on the labels (n_k >= 10; otherwise the
                     pooled within-class variance; floor 1e-3 x pooled);
    metric 'euclid': Sinv_kd = 1 / S_d (normalised Euclidean distance; ablation)."""
    C = counts.size
    ok = counts[y_lab] >= 2
    xbar, S_d = [], np.empty(len(views))
    for j, X in enumerate(views):
        XL = X[lab_mask]
        mu = np.stack([XL[y_lab == k].mean(axis=0) for k in range(C)])
        xbar.append(mu)
        s = float(((XL[ok] - mu[y_lab[ok]]) ** 2).sum(axis=1).mean()) if ok.any() else 0.0
        if s <= 0.0:
            Xv = X[nu > 0]
            s = float(np.median(((Xv[:, None, :] - mu[None]) ** 2).sum(-1).min(axis=1)))
            warnings.warn(f"S_d[{j}]: no class with >= 2 labels; using the median distance to the nearest class mean", stacklevel=2)
        S_d[j] = s if s > 0.0 else 1.0
    Sinv = []
    for j, X in enumerate(views):
        d = X.shape[1]
        if metric == "euclid":
            Sinv.append(np.full((C, d), 1.0 / S_d[j]))
            continue
        XL = X[lab_mask]
        pooled = np.maximum(((XL - xbar[j][y_lab]) ** 2).mean(axis=0), 1e-12)
        s_kd = np.tile(pooled[None, :], (C, 1))
        for k in range(C):
            if counts[k] >= 10:
                s_kd[k] = np.maximum(XL[y_lab == k].var(axis=0), 1e-3 * pooled)
        Sinv.append(1.0 / (d * s_kd))
    return xbar, S_d, Sinv


def label_view_weights(views, lab_mask, y_lab, counts, xbar, Sinv, m, clip_eps=1e-12):
    """Lambda_j = mean over labelled pixels of -ln u_hat_{i,y_i}, where u_hat is the FCM membership with the centres at the class
    means (the own class mean leaves pixel i out); a_j proportional to 1 / Lambda_j (Eq. (lambda)-(a) of the paper)."""
    n, nk = y_lab.size, counts[y_lab]
    loo = nk >= 2
    rows = np.arange(n)
    Lam = np.empty(len(views))
    for j, X in enumerate(views):
        XL, mu = X[lab_mask], xbar[j]
        d2 = (((XL[:, None, :] - mu[None]) ** 2) * Sinv[j][None]).sum(-1)
        mu_loo = (nk[:, None] * mu[y_lab] - XL) / np.maximum(nk - 1, 1)[:, None]
        d_own = (((XL - mu_loo) ** 2) * Sinv[j][y_lab]).sum(axis=1)
        d2[rows, y_lab] = np.where(loo, d_own, d2[rows, y_lab])
        u = update_u(np.maximum(d2, clip_eps), m)
        Lam[j] = (-np.log(np.maximum(u[rows, y_lab], clip_eps))).mean()
    Lam = np.maximum(Lam, clip_eps)
    return Lam, (1.0 / Lam) / (1.0 / Lam).sum()


def _proba(clf, X, C, clip_p):
    p = np.asarray(clf.predict_proba(X), dtype=np.float64)
    classes = np.asarray(clf.classes_, dtype=int)
    if p.shape[1] != C or not np.array_equal(classes, np.arange(C)):
        full = np.full((X.shape[0], C), clip_p)
        full[:, classes] = p
        p = full
    p = np.clip(p, clip_p, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def lda_posteriors(views, lab_mask, y_lab, C, folds=5, seed=None, clip_p=1e-6):
    """Per-view shrinkage LDA (Ledoit-Wolf, closed form): posteriors on every pixel; labelled rows are replaced by stratified
    out-of-fold predictions so that S_g is not biased low."""
    counts = np.bincount(y_lab, minlength=C)
    n_splits = int(min(folds, counts.min()))
    out = []
    for X in views:
        XL = X[lab_mask]
        P = _proba(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(XL, y_lab), X, C, clip_p)
        if n_splits >= 2:
            oof = np.empty((XL.shape[0], C))
            for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(XL, y_lab):
                oof[te] = _proba(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(XL[tr], y_lab[tr]), XL[te], C, clip_p)
            P[lab_mask] = oof
        out.append(P)
    return out


# ----------------------------------------------------------------------------- the model
def cola_mvfc(views, labeled, n_clusters, theta=1.0, theta_g=0.8, m=2.0, metric="diag", view_prior="labels",
              epsilon=1e-4, max_iter=300, objective_tol=1e-6, valid_mask=None, seed=None, oof_folds=5,
              clip_eps=1e-12, clip_p=1e-6) -> Result:
    """Fit CoLA-MVFC.

    views     : list of M arrays (N, d_j), rows = the same pixels in every view (co-registered)
    labeled   : (N,) int, class 0..C-1 on labelled pixels, NO_LABEL (-1) elsewhere; every class needs >= 1 label
    valid_mask: (N,) bool, pixels that enter J (default: all); labelled pixels must be valid
    theta     : class-mean constraint ratio (0 = off, large = centres fixed at the class means)
    theta_g   : guidance share in [0, 1) (0 = no classifier guidance)
    """
    t0 = time.time()
    views = [np.asarray(v, dtype=np.float64) for v in views]
    N, M, C = views[0].shape[0], len(views), int(n_clusters)
    if any(v.ndim != 2 or v.shape[0] != N for v in views):
        raise ValueError("views must be 2-D arrays with the same number of rows")
    if m <= 1.0 or theta < 0.0 or not (0.0 <= theta_g < 1.0):
        raise ValueError("need m > 1, theta >= 0 and 0 <= theta_g < 1")
    labeled = np.asarray(labeled).astype(int).reshape(-1)
    lab_mask = labeled != NO_LABEL
    y_lab = labeled[lab_mask]
    counts = np.bincount(y_lab, minlength=C)
    if y_lab.size == 0 or y_lab.min() < 0 or y_lab.max() >= C or counts.min() == 0:
        raise ValueError("labels must be in 0..C-1 and every class needs at least one label")
    nu = np.ones(N) if valid_mask is None else np.asarray(valid_mask).reshape(-1).astype(bool).astype(np.float64)
    if np.any(nu[lab_mask] == 0):
        raise ValueError("labelled pixels must lie inside valid_mask")
    n_valid = float(nu.sum())

    # label anchoring: metric, class means, view weights, guidance unit (computed once, on the labels only)
    xbar, _, Sinv = class_scales(views, lab_mask, y_lab, counts, nu, metric)
    if view_prior == "labels":
        Lam, a = label_view_weights(views, lab_mask, y_lab, counts, xbar, Sinv, m, clip_eps)
    else:
        Lam, a = None, np.full(M, 1.0 / M)
    alpha = theta_g / (1.0 - theta_g)
    P, S_g, g = [], np.full(M, np.nan), None
    if alpha > 0.0:
        P = lda_posteriors(views, lab_mask, y_lab, C, oof_folds, seed, clip_p)
        rows = np.arange(y_lab.size)
        S_g = np.array([max(float(np.mean(-np.log(p[lab_mask][rows, y_lab]))), clip_eps) for p in P])
        g = np.stack([-np.log(p) / S_g[j] for j, p in enumerate(P)])                     # (M, N, C)
    rho = theta * (n_valid / y_lab.size) * counts.astype(np.float64)
    Y = np.eye(C)[y_lab]
    lab_idx = np.flatnonzero(lab_mask)

    def costs(V):                                                                        # e^(j), (M, N, C)
        delta = np.stack([assignment_distance(X, V[j], Sinv[j]) for j, X in enumerate(views)])
        return delta + alpha * g if g is not None else delta

    def objective(U, V, e):
        w = nu[None, :, None] * (U ** m)[None] * a[:, None, None]
        fit = float((w * e).sum(dtype=np.float64))
        cm = sum(a[j] * float(((((V[j] - xbar[j]) ** 2) * Sinv[j]).sum(axis=1) * rho).sum()) for j in range(M))
        return fit + cm

    V = [mu.copy() for mu in xbar]                                                       # centres start at the class means
    history, U_prev, converged, n_iter = [], None, False, 0
    for t in range(max_iter):
        e = costs(V)
        D = np.einsum("j,jnc->nc", a, e)                                                  # fused cost D = sum_j a_j e^(j)
        U = update_u(np.maximum(D, clip_eps), m)
        U[lab_idx] = Y                                                                    # labelled rows fixed to their labels
        history.append(objective(U, V, e))                                                # J(U^{t+1}, V^t)
        Um = nu[:, None] * U ** m
        V = [update_v(views[j], a[j] * Um, xbar[j], a[j] * rho) for j in range(M)]
        n_iter = t + 1
        if U_prev is not None:
            dU = float((np.abs(U - U_prev) * nu[:, None]).max())
            dJ_ok = objective_tol > 0 and abs(history[-2] - history[-1]) <= objective_tol * max(1.0, abs(history[-2]))
            U_prev = U
            if dU < epsilon or dJ_ok:
                converged = True
                break
        else:
            U_prev = U
    history.append(objective(U, V, costs(V)))                                            # J at the returned (U, V)
    return Result(membership=U, labels=U.argmax(axis=1), centroids=V, view_weights=a, log_loss=Lam, posteriors=P, S_g=S_g,
                  Sinv=Sinv, alpha=alpha, objective_history=history, n_iter=n_iter, converged=converged, time=time.time() - t0)


# ----------------------------------------------------------------------------- parameter selection (Section 3.3)
def holdout_split(labeled, n_clusters, frac=0.2, seed=0):
    """Hold out frac of the labels of every class (stratified, rng seed + 1000); returns (labels_for_fitting, held_mask)."""
    rng = np.random.default_rng(seed + 1000)
    lab_train, held = labeled.copy(), np.zeros(len(labeled), bool)
    for k in range(n_clusters):
        idx = np.flatnonzero(labeled == k)
        n_h = min(int(round(frac * len(idx))), len(idx) - 1)
        if n_h <= 0:
            continue
        h = rng.choice(idx, n_h, replace=False)
        held[h] = True
        lab_train[h] = NO_LABEL
    return lab_train, held


def select_params(views, labeled, n_clusters, thetas=PAPER_GRID["thetas"], theta_gs=PAPER_GRID["theta_gs"], frac=0.2, seed=0,
                  **kw):
    """Choose (theta, theta_g) by overall accuracy on 20 % held-out labels, fitting on the other 80 % (per run, as in the paper).
    Returns (theta, theta_g, table)."""
    labeled = np.asarray(labeled).astype(int).reshape(-1)
    lab_tr, held = holdout_split(labeled, n_clusters, frac, seed)
    table = []
    for th in thetas:
        for tg in theta_gs:
            r = cola_mvfc(views, lab_tr, n_clusters, theta=th, theta_g=tg, seed=seed, **kw)
            table.append(dict(theta=th, theta_g=tg, acc_holdout=float((r.labels[held] == labeled[held]).mean())))
    best = max(table, key=lambda d: d["acc_holdout"])
    return best["theta"], best["theta_g"], table
