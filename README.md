# vrpipeline — Unified Virtual-Resection Pipeline for Drug-Resistant Epilepsy

A single, real, research-grade Python package that fuses the two virtual-
resection methodologies for predicting epilepsy-surgery outcome from
intracranial EEG, implemented directly from the equations in both papers
(not adapted from, or dependent on, either paper's original — now stale
Python-2/`mtspec` — codebase, though both were used to verify correctness):

| | Paper | Network | Resection | Headline result |
|---|---|---|---|---|
| **A** | Kini, Bernabei et al., *Brain* 2019, 142:3892–3905, "Virtual resection predicts surgical outcome for drug-resistant epilepsy" | Undirected multitaper-coherence networks, 5 bands + broadband cross-correlation | En-bloc removal of the clinically resected electrode set; symmetric-Laplacian synchronizability s(t) | Broadband Δs(t) predicts outcome, AUC = 0.89 |
| **B** | Sun, Niu et al., *Sci Rep* 2024, 14:25542, "Virtual resection evaluation based on sEEG propagation network for drug-resistant epilepsy" | Directed, high-order transfer-entropy propagation networks | Sequential, propagation-path-ordered node removal; asymmetric outdegree-Laplacian stability R(t); inflection-point minimum-intervention-target search | STE similarity to ground truth = 0.93; 12/16 patients' data-driven scheme matched clinical resection, and the 4 mismatches correlated with poor clinical outcome |

Both pipelines run on the **same raw recordings** so their outputs are
directly comparable, patient by patient, seizure by seizure.

## What is genuinely implemented here (no mocks / no stand-ins)

- **Multitaper coherence** (`connectivity.py`): real DPSS/Slepian
  multitaper cross-spectral estimation built from scratch on
  `scipy.signal.windows.dpss` + FFT (the original Echobase code depended
  on the now-unmaintained `mtspec` C-extension; this is a from-scratch,
  numerically equivalent replacement, band-averaged multitaper coherence).
- **Transfer entropy + high-order composition** (`connectivity.py`): a
  genuine histogram/plug-in TE estimator implementing eq. (9) of Sun et
  al. directly (embedded-vector joint/conditional entropies), FDR-BH
  significance testing against time-shift surrogates, and literal
  directed-path matrix-power composition for the high-order network
  (eq. 10).
- **Symmetric & asymmetric network-stability measures**
  (`synchronizability.py`): s(t) = λ₂/λ_N of the standard Laplacian
  (Kini et al., verified line-for-line against Echobase's
  `globaltopo.synchronizability`), and R = λ_N/λ₂ of the directed
  outdegree Laplacian (Sun et al. eq. 11–12), with an explicit,
  documented note about the paper's own text-vs-equation ambiguity for
  eq. (12) and a robust "skip the disconnected-component zero-eigenvalue
  block" convention.
- **Two virtual-resection schemes** (`resection.py`): en-bloc
  resection-zone control centrality c_res (Kini et al., verified against
  EpiVR's `region_control`), and sequential/hierarchical propagation-
  ordered virtual resection with automatic inflection-point (minimum
  intervention target) detection and alternative-scheme comparison (Sun
  et al. Fig. 2/4/5).
- **Real imaging-to-electrode mapping** (`resection.py`): point-in-hull
  membership testing of electrode coordinates against a NIfTI resection
  mask (via `scipy.spatial.Delaunay`, verified against EpiVR's
  `in_hull`), with a dilation/erosion sweep reproducing the segmentation-
  robustness analysis in Kini et al. Supplementary Fig. S3.
- **Every statistic actually used in the two papers** (`stats.py`):
  Wilcoxon rank-sum, a real DeLong correlated-ROC test (fast
  structural-component implementation, not a bootstrap approximation),
  a 10,000-permutation functional-data-analysis curve-difference test,
  Fisher-exact odds ratios with 95% CI, Bonferroni correction.
- **Outcome prediction** (`outcome.py`): full ROC/AUC sweep, Youden-
  optimal threshold selection, confusion-matrix reporting in the exact
  TP/FP/FN/TN/accuracy/TPR/TNR layout of Supplementary Table S2.
- **A validated synthetic data generator** (`simulate.py`): the full
  5-equation Jirsa Epileptor coupled neural-mass model (eq. 1–8 of Sun et
  al.), solved with `scipy.integrate.solve_ivp`, so you can smoke-test
  every stage of the pipeline immediately, before your real IEEG.org/
  OpenNeuro downloads are ready.

## What you must supply yourself

This package deliberately does **not** bundle or auto-download patient
data (nor could it — both papers' cohorts require IRB-governed access).
You bring:

1. Raw iEEG/sEEG/ECoG recordings (EDF, or an HDF5 clip in the
   `evData`/`Fs`/`channels` layout IEEG.org exports use — or just pass
   any `(T, N)` numpy array you already loaded via `vrpipeline.io.load_array`).
2. Electrode coordinates (CSV: `id,x,y,z,label`), co-registered to your
   imaging space.
3. A post-resection binary segmentation mask (NIfTI), co-registered to
   the same space as (2) — produced by your own ANTs/ITK-SNAP/FreeSurfer
   registration + segmentation pipeline (this is a neuro-imaging task,
   not a signal-processing one; Kini et al.'s Fig. 1 describes exactly
   this step, which they also perform outside Python).
4. Clinical seizure annotations (EEC/UEO/END, semiology) if you want the
   pre-ictal/ictal windowing to match Table S1's convention exactly.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .        # optional, or just add this repo to PYTHONPATH
```

## Quickstart (no data required — validates the whole pipeline)

```bash
python scripts/run_single_patient.py --simulate --out-dir results/sim_demo
```

## Real usage

```bash
python scripts/run_single_patient.py \
    --pre-ictal patient01_preictal.h5 --ictal patient01_ictal.h5 --format h5 \
    --electrode-coords patient01_electrodes.csv \
    --resection-mask patient01_resection_mask.nii.gz \
    --seed SOZ_E1,SOZ_E2 \
    --out-dir results/patient01

# after running the above per-patient and collecting band_delta_s into one
# CSV row per patient (see examples/cohort_features_example.csv for the
# exact schema):
python scripts/run_cohort.py --csv cohort_features.csv --out-dir results/cohort
```

Or drive everything programmatically:

```python
from vrpipeline.io import load_edf
from vrpipeline.pipeline import run_pipeline_a, run_pipeline_b

pre = load_edf("patient01_preictal.edf")
ictal = load_edf("patient01_ictal.edf")

result_a = run_pipeline_a(pre, ictal, resected_labels=["LG1", "LG2", "LG3"])
result_b = run_pipeline_b(pre, ictal, seed_labels=["LG1", "LG2"],
                           resected_labels=["LG1", "LG2", "LG3"])
```

## Layout

```
vrpipeline/
  io.py                 real loaders: EDF/HDF5 iEEG, electrode coords, NIfTI masks
  preprocessing.py       CAR, notch, band-pass, resample, time-normalize
  connectivity.py         multitaper coherence + transfer entropy + HTE
  synchronizability.py    symmetric s(t) and asymmetric R(t)
  propagation.py           HTE-derived hop-layer propagation path extraction
  resection.py             electrode<->mask mapping, en-bloc + sequential VR
  simulate.py               Jirsa Epileptor model for synthetic validation data
  stats.py                   Wilcoxon, DeLong, permutation FDA test, odds ratios
  outcome.py                  ROC/AUC, confusion matrix, thresholding
  pipeline.py                  end-to-end orchestration (both pipelines)
scripts/
  run_single_patient.py         CLI: run both pipelines on one patient
  run_cohort.py                   CLI: cohort-level ROC/DeLong/confusion-matrix
tests/                              27 passing unit + integration tests
examples/cohort_features_example.csv   cohort CSV schema
```

## Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

27/27 pass, including a full end-to-end integration smoke test that
simulates a coupled-Epileptor "seizure," runs it through both pipelines,
and validates every numeric output is finite/sane — plus tests of the
imaging-based electrode-to-resection-zone mapping including the
dilation/erosion robustness sweep.

## Key implementation notes / caveats (read before publishing results)

- **`fdr_binarize`'s surrogate test is O(n² × n_perm)** per epoch; for
  networks with 50+ electrodes and hundreds of epochs, either
  parallelize this (it's embarrassingly parallel across edges — see
  the commented-out `multiprocessing.Pool` pattern in the original
  EpiVR `null_virtual_resection`, which you can drop in), or precompute
  once per representative epoch rather than every epoch.
- **eq. (12) ambiguity**: Sun et al.'s body text says "ratio of the
  third smallest eigenvalue to the largest eigenvalue" but their eq.
  (12) states R = λ_N/λ₂. `asymmetric_stability(..., smallest_index=1)`
  implements the equation as written (matches their reported single-
  inflection-point curve shapes); pass `smallest_index=2` to instead
  reproduce the text description. This is flagged, not silently
  resolved, in the docstring.
- **TE estimator choice**: an equiprobable-histogram plug-in estimator
  is used (matches the discretized probability form of eq. 9 directly).
  For sparser/noisier real sEEG you may want a KSG (k-nearest-neighbor,
  continuous) TE estimator instead for lower bias at fixed sample size —
  `connectivity.transfer_entropy` is a natural drop-in replacement point
  if you add one (e.g. via `pyinform` or `jpype`+JIDT).
- **Multitaper coherence bandwidth/taper count** (`time_band`, `n_taper`)
  trade off frequency resolution against variance exactly as in the
  original mtspec-based implementation; defaults (`time_band=4.0`,
  `n_taper=7`) are conservative starting points, not paper-reported
  optimal values (neither paper published these hyperparameters exactly).
- **Statistical validity**: none of the ROC/AUC/DeLong/odds-ratio numbers
  mean anything on a sample this small until you run them on your real,
  sufficiently powered cohort — the code is correct; the inference is
  only as good as your n and your outcome-labeling fidelity (Engel I /
  ILAE 1-2 = good, Engel II-IV / ILAE 3-6 = poor, >=1 year follow-up, per
  both papers).
