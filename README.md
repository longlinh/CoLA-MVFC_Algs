# CoLA-MVFC — reference implementation

Companion source code for the manuscript

> **Collaborative label-anchored semi-supervised multi-view fuzzy clustering for hyperspectral and low-dimensional auxiliary
> remote sensing data.** Xuan Hoang Nguyen, Long Giang Nguyen, Dinh Sinh Mai. Manuscript submitted for publication (2026).

A compact NumPy/scikit-learn implementation written to be read next to the equations of the paper: plain functions, no class
hierarchy, CPU only. The repository contains the **algorithm** with its parameter selection, the label-sampling and corruption
functions used in the paper (deterministic in the seed), the pixel indices of the label sets, held-out labels and corruption masks
of every run in the paper (`protocol_files/`), a synthetic demonstration and tests, including numerical parity with the
implementation used for the paper's experiments.

**CoLA-MVFC** estimates one shared fuzzy membership `U` for co-registered views of very different dimensionality (for example a
hyperspectral image with LiDAR, SAR or a digital surface model). Every quantity a view contributes is estimated from the fixed
label set `L` and expressed in a unit defined on that label set (*label anchoring*); the views collaborate only through their
assignment-cost matrices and the shared membership:

```
J(U,V)   = Σ_j a_j [ Σ_i ν_i Σ_k u_ik^m e_ik^(j) + Σ_k ρ_k ‖v_k^(j) − x̄_k^(j)‖²_σ ]      ν_i = 1 for valid pixels, N_ν = Σ_i ν_i
e^(j)    = δ^(j) + α g^(j),  α = θ_g/(1−θ_g)
δ^(j)_ik = Σ_d (x^(j)_id − v^(j)_kd)² σ^(j)_kd,  σ^(j)_kd = 1/(d_j s^(j)_kd)  s: per-class variance on the labels
g^(j)_ik = −ln p^(j)_ik / S_g^(j)                               p^(j): shrinkage LDA of view j (out-of-fold on the labels)
S_g^(j)  = mean over i in L of −ln p^(j)_{i,y_i}                guidance unit: mean guidance cost 1 on the labels
a_j ∝ 1/Λ_j                                                     Λ_j: leave-one-out log-loss of view j at the class means
ρ_k = θ N_ν n_k / |L|                                           class-mean constraint weight (n_k labels in class k)
U-step  u_ik ∝ (Σ_j a_j e_ik^(j))^(−1/(m−1)),  labelled rows fixed to their labels
V-step  v_k^(j) = (Σ_i ν_i u_ik^m x_i^(j) + ρ_k x̄_k^(j)) / (Σ_i ν_i u_ik^m + ρ_k)
```

Both steps are closed form for every `m > 1` and `J` is non-increasing. Early fusion is the one-view case of the same model
(concatenated features) and late fusion is the same model run per view with the memberships averaged by `a_j` afterwards
(collaboration off).

## Install and run

Python 3.9 or newer; three dependencies, no GPU:

```bash
pip install -r requirements.txt     # numpy, scipy, scikit-learn (pytest for the tests)
python demo.py                      # synthetic scene, about 2 s
pytest tests/ -q                    # 19 tests; the 10 parity tests need the authors' library and experiment code and are skipped
```

Tested with Python 3.12.9, NumPy 2.3.2, SciPy 1.15.3 and scikit-learn 1.7.0. `demo.py` prints, deterministically (5 % stratified labels,
parameters selected on 20 % held-out labels):

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

`seed` fixes the stratified out-of-fold split of the LDA posteriors, the only random step of the model.

## Paper settings

| Item | Setting (Section 4.1 of the paper) |
|---|---|
| Data | Trento (HSI + LiDAR) [1], MUUFL Gulfport (HSI + DSM + LiDAR intensity) [2, 3], Augsburg (HSI + SAR + DSM) [4], from their providers; every band z-scored with the mean and population standard deviation over the reference-map pixels |
| Labels | `stratified_labels` or `region_labels` (radius 3) at 5 % per class; label-fraction curves at 1, 3, 5 and 10 %; seeds 42–51 |
| Corruption | on the hyperspectral view, labelled pixels included, no mask given to the model: `cloud_blocks` (32 × 32 blocks, value 3 + N(0, 1), 20, 40 and 60 % of the valid pixels), `zero_blocks` (missing data, 40 %), `haze_blocks` (thin cloud with soft edges, 40 and 60 %) |
| Parameters | `PAPER_PARAMS` (m = 2, diagonal metric, label view weights, ε = 1e-4, at most 300 iterations); θ ∈ {0.5, 1, 2, 4}, θ_g ∈ {0.5, 0.8, 0.95} chosen per run by overall accuracy on 20 % held-out labels (`select_params`) |
| Accuracy | Overall accuracy on the unlabelled valid pixels, cluster k read as class k |

The datasets are not redistributed here; download them from their providers. Reference maps used by the paper (row-major, class k =
code k + 1 of the provider's map, other codes outside the map):

| Scene | Reference map | Size | Reference pixels per class |
|---|---|---|---|
| Trento | `allgrd.mat`, variable `mask_test` | 166 × 600 | 4034, 2903, 479, 9123, 10501, 3174 |
| MUUFL Gulfport | `muufl_gulfport_campus_1_hsi_220_label.mat`, `hsi.sceneLabels.labels` (codes 1–11) | 325 × 220 | 23246, 4270, 6882, 1826, 6687, 466, 2233, 6240, 1385, 183, 269 |
| Augsburg | `TrainImage.mat` and `TestImage.mat` merged | 332 × 485 | 13507, 30329, 3851, 26857, 575, 1645, 1530 |

## Label sets and corruption masks of the paper

`protocol_files/<scene>.npz` (`trento`, `muufl`, `augsburg`) store, for seeds 42–51, the pixel indices of every label set used in
the paper (5 % stratified, 5 % region, and 1, 3, 10 % stratified for the label-fraction curves), the 20 % of those labels held out for
selecting θ and θ_g, and the pixels covered by each corruption. They were produced with the functions of this repository and are
identical, element by element, to the sets used in the experiments. Only pixel indices are stored: the class of a labelled pixel is
the value of the reference map at that index.

| Key | Content |
|---|---|
| `height`, `width`, `n_classes`, `seeds`, `regimes`, `corruptions` | image size and the settings stored in the file |
| `reference_counts`, `reference_sha256` | reference pixels per class, and SHA-256 of the reference map as int16 little-endian (−1 outside) |
| `labelled\|<regime>\|<seed>` | indices of the labelled pixels (int32, ascending); `<regime>`: `stratified1`, `stratified3`, `stratified5`, `stratified10`, `region5` |
| `heldout\|<regime>\|<seed>` | indices of the held-out labels used for parameter selection |
| `<corruption>\|<seed>` | covered pixels, `np.packbits` of a boolean vector of length H·W; `<corruption>`: `cloud20`, `cloud40`, `cloud60`, `haze40`, `haze60` (missing data at 40 % uses the blocks of `cloud40`) |

```python
import hashlib
import numpy as np
z = np.load("protocol_files/trento.npz")
H, W = int(z["height"]), int(z["width"])
assert hashlib.sha256(y_full.astype("<i2").tobytes()).hexdigest() == str(z["reference_sha256"])   # same reference map
idx = z["labelled|stratified5|42"]
lab = np.full(H * W, -1)
lab[idx] = y_full[idx]                                                       # labels argument of cola_mvfc / select_params
covered = np.unpackbits(z["cloud60|42"], count=H * W).astype(bool)          # pixels whose hyperspectral bands are replaced
```

The corrupted values are reproduced by the corresponding function with the same seed, e.g. `cloud_blocks(views, 0, y_full >= 0, H,
W, 0.6, seed)`; its mask equals `covered`, and when it does the values match as well (same random stream).

## Files

| File | Content |
|---|---|
| `cola_mvfc.py` | the model: label anchoring (`class_scales`, `label_view_weights`, `lda_posteriors`), closed-form updates, `cola_mvfc`, `select_params` |
| `protocol.py` | label layouts and corruptions of the paper |
| `protocol_files/` | pixel indices of the label sets, held-out labels and corruption masks of the paper's runs (seeds 42–51) |
| `synthetic.py`, `demo.py` | synthetic co-registered scene and demonstration |
| `tests/` | monotonicity, fixed labelled rows, unit of the guidance cost, determinism, parameter selection, input checks, structure of the protocol files; parity with the authors' implementation when available |

## Citation and licence

MIT licence (see `LICENSE`). If you use this code, please cite both the paper and this software (see `CITATION.cff`). If you use the
data, cite the dataset providers:

1. B. Rasti, P. Ghamisi, R. Gloaguen. Hyperspectral and LiDAR fusion using extinction profiles and total variation component
   analysis. IEEE Transactions on Geoscience and Remote Sensing 55(7):3997–4007, 2017. https://doi.org/10.1109/TGRS.2017.2686450
2. P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell. MUUFL Gulfport hyperspectral and LiDAR airborne data set. Technical Report
   REP-2013-570, University of Florida, 2013 (Release 01, https://doi.org/10.5281/zenodo.1186326).
3. X. Du, A. Zare. Technical report: Scene label ground truth map for MUUFL Gulfport data set. Technical Report 20170417,
   University of Florida, Gainesville, FL, 2017. https://ufdc.ufl.edu/IR00009711/00001
4. D. Hong, J. Hu, J. Yao, J. Chanussot, X. X. Zhu. Multimodal remote sensing benchmark datasets for land cover classification with
   a shared and specific feature learning model. ISPRS Journal of Photogrammetry and Remote Sensing 178:68–80, 2021.
   https://doi.org/10.1016/j.isprsjprs.2021.05.011
