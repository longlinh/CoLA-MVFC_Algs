"""Self-contained demonstration on a synthetic co-registered scene (no data needed, a few seconds on a CPU).

View 0 has 60 dimensions (a stand-in for a hyperspectral view), views 1 and 2 have 1 and 2 dimensions (auxiliary sources).
The script samples 5 % stratified labels, selects (theta, theta_g) on 20 % held-out labels, fits CoLA-MVFC on the clean scene and
after 60 % block corruption of view 0, and prints the overall accuracy on unlabelled pixels, the view weights and whether the
objective was non-increasing.

    python demo.py
"""
import numpy as np

from cola_mvfc import PAPER_PARAMS, cola_mvfc, select_params
from protocol import cloud_blocks, stratified_labels
from synthetic import make_scene

SEED = 42


def run(views, y, lab, C, valid):
    th, tg, _ = select_params(views, lab, C, seed=SEED, valid_mask=valid, **PAPER_PARAMS)
    r = cola_mvfc(views, lab, C, theta=th, theta_g=tg, seed=SEED, valid_mask=valid, **PAPER_PARAMS)
    unl = valid & (lab < 0)
    J = np.asarray(r.objective_history)
    mono = bool(np.all(np.diff(J) <= 1e-9 * np.maximum(1.0, np.abs(J[:-1]))))
    return 100 * float((r.labels[unl] == y[unl]).mean()), th, tg, r.view_weights, r.n_iter, mono


def main():
    sc = make_scene(seed=SEED)
    views, y, C, H, W = sc["views"], sc["y_full"], sc["n_classes"], sc["height"], sc["width"]
    valid = y >= 0
    lab = stratified_labels(y, C, 0.05, SEED)
    cloudy, dam = cloud_blocks(views, 0, valid, H, W, 0.6, SEED)
    print(f"scene {H}x{W}, C = {C}, views d = {[v.shape[1] for v in views]}, labelled pixels = {int((lab >= 0).sum())}")
    for name, vs in (("clean", views), ("60 % cloud on view 0", cloudy)):
        oa, th, tg, a, it, mono = run(vs, y, lab, C, valid)
        print(f"{name:>22}: OA = {oa:5.2f} %   theta = {th:g}, theta_g = {tg:g}   a = {np.round(a, 3).tolist()}   "
              f"iterations = {it}, J non-increasing = {mono}")


if __name__ == "__main__":
    main()
