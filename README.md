# CoLA-MVFC — reference implementation

Companion source code for the manuscript

> **Collaborative label-anchored semi-supervised multi-view fuzzy clustering for hyperspectral and low-dimensional auxiliary
> remote sensing data.** Xuan Hoang Nguyen, Long Giang Nguyen, Dinh Sinh Mai. Manuscript submitted for publication (2026).

A compact NumPy/scikit-learn implementation written to be read next to the equations of the paper: plain functions, no class
hierarchy, CPU only. The repository publishes the **algorithm**, the **evaluation protocol** (label layouts and the local-corruption
scenario, deterministic in the seed), a self-contained synthetic demonstration and tests, including numerical parity with the
implementation used for the paper's experiments.

**CoLA-MVFC** estimates one shared fuzzy membership `U` for co-registered views of very different dimensionality (for example a
hyperspectral image with LiDAR, SAR or a digital surface model). Every quantity a view contributes is estimated from the fixed
label set and expressed in a unit defined on that label set (*label anchoring*); the views collaborate only through their
assignment-cost matrices and the shared membership:

```
J(U,V)   = Σ_j a_j [ Σ_i ν_i Σ_k u_ik^m e_ik^(j) + Σ_k ρ_k ‖v_k^(j) − x̄_k^(j)‖²_σ ]
e^(j)    = δ^(j) + α g^(j),  α = θ_g/(1−θ_g)
δ^(j)_ik = Σ_d (x_id − v_kd)² σ_kd,   σ_kd = 1/(d_j s_kd)         per-class variances on the labels
g^(j)_ik = −ln p^(j)_ik / S_g^(j)                                 p^(j): shrinkage LDA of view j (out-of-fold on the labels)
a_j ∝ 1/Λ_j                                                       leave-one-out log-loss of view j at the class means
ρ_k = θ N_ν n_k / |L|                                             class-mean constraint weight
U-step  u_ik ∝ (Σ_j a_j e_ik^(j))^(−1/(m−1)),  labelled rows fixed to their labels
V-step  v_k^(j) = (Σ_i ν_i u_ik^m x_i + ρ_k x̄_k) / (Σ_i ν_i u_ik^m + ρ_k)
```

Both steps are closed form for every `m > 1` and `J` is non-increasing. By construction the guidance cost of every view has mean 1
on the labelled pixels, whatever its dimensionality. Early fusion is the one-view case of the same model (concatenated features) and
late fusion is the same model run per view with the memberships averaged by `a_j` afterwards (collaboration off).

## Install and run

Python 3.9 or newer; three dependencies, no GPU:

```bash
pip install -r requirements.txt     # numpy, scipy, scikit-learn
python demo.py                      # synthetic scene, about 2 s
pytest tests/ -q                    # 15 tests (parity tests are skipped without the authors' library)
```

`demo.py` prints, deterministically (5 % stratified labels, parameters selected on 20 % held-out labels):

```
scene 60x80, C = 5, views d = [60, 1, 2], labelled pixels = 216
                 clean: OA = 87.62 %   theta = 0.5, theta_g = 0.5   a = [0.361, 0.313, 0.325]   iterations = 4, J non-increasing = True
  60 % cloud on view 0: OA = 55.10 %   theta = 0.5, theta_g = 0.95   a = [0.342, 0.323, 0.335]   iterations = 3, J non-increasing = True
```

## Use

```python
from cola_mvfc import PAPER_PARAMS, cola_mvfc, select_params
from protocol import stratified_labels

# views: list of (N, d_j) arrays on the same pixel grid, z-scored per band; y_full: (N,) reference map, -1 outside
lab = stratified_labels(y_full, n_classes, 0.05, seed)                       # 5 % labels per class
theta, theta_g, _ = select_params(views, lab, n_classes, seed=seed, valid_mask=y_full >= 0, **PAPER_PARAMS)
res = cola_mvfc(views, lab, n_classes, theta=theta, theta_g=theta_g, seed=seed, valid_mask=y_full >= 0, **PAPER_PARAMS)
res.membership          # (N, C) shared fuzzy membership; cluster k is class k (no label matching)
res.view_weights        # (M,) supervised view weights a_j
res.objective_history   # non-increasing
```

## Paper protocol

| Item | Setting (Section 4.1 of the paper) |
|---|---|
| Data | Trento (HSI + LiDAR), MUUFL Gulfport (HSI + DSM + LiDAR intensity), Augsburg (HSI + SAR + DSM), from their original providers; bands z-scored per view |
| Labels | `stratified_labels` or `region_labels` (radius 3) at 5 % per class; label-fraction curves at 1, 3, 5 and 10 %; seeds 42–51 |
| Corruption | `cloud_blocks(views, 0, valid, H, W, 0.6, seed)`: 32 × 32 blocks covering 60 % of the valid pixels of the hyperspectral view, value 3 + N(0, 1) in every band, labelled pixels included, no mask given to the model |
| Parameters | `PAPER_PARAMS` (m = 2, diagonal metric, label view weights, ε = 1e-4, at most 300 iterations); θ ∈ {0.5, 1, 2, 4}, θ_g ∈ {0.5, 0.8, 0.95} chosen per run by overall accuracy on 20 % held-out labels (`select_params`) |
| Metric | Overall accuracy on the unlabelled valid pixels, cluster k read as class k |

The datasets are not redistributed here; download them from their providers and pass the arrays as above.

## Files

| File | Content |
|---|---|
| `cola_mvfc.py` | the model: label anchoring (`class_scales`, `label_view_weights`, `lda_posteriors`), closed-form updates, `cola_mvfc`, `select_params` |
| `protocol.py` | label layouts and block corruption of the paper |
| `synthetic.py`, `demo.py` | synthetic co-registered scene and demonstration |
| `tests/` | monotonicity, fixed labelled rows, unit of the guidance cost, determinism, parameter selection; parity with the authors' implementation when available |

## Citation and licence

MIT licence (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
