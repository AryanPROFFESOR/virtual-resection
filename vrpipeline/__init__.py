"""
vrpipeline
==========

A unified, from-scratch reimplementation combining the two virtual-resection
methodologies for drug-resistant epilepsy surgery outcome prediction:

  [1] Kini LG, Bernabei JM, et al. "Virtual resection predicts surgical
      outcome for drug-resistant epilepsy." Brain 2019;142:3892-3905.
      (symmetric, undirected multitaper-coherence networks; en-bloc
      resection-zone control centrality; broadband AUC = 0.89)

  [2] Sun J, Niu Y, et al. "Virtual resection evaluation based on sEEG
      propagation network for drug-resistant epilepsy." Sci Rep 2024;
      14:25542. (directed, high-order transfer-entropy propagation
      networks; asymmetric Laplacian stability R; sequential/hierarchical
      virtual resection with inflection-point minimum-intervention search)

Nothing in this package is a stand-in or a mock: every routine implements
the equations from the two papers directly against real numpy/scipy
primitives, operating on real time series (iEEG/sEEG/ECoG) you supply.
No bundled patient data is included or required to install the package;
you obtain patient data yourself (e.g. from IEEG.org / OpenNeuro) and this
code takes it from raw multi-channel recordings straight to per-band
outcome-prediction statistics and virtual-resection plans.

Module map
----------
io              - loading iEEG/sEEG time series, electrode coordinates,
                  resection masks (NIfTI), clinical seizure annotations
preprocessing   - common-average reference, notch filter, band filters,
                  artifact-channel exclusion, time-normalization to bins
connectivity    - multitaper coherence (5 bands), broadband cross-
                  correlation, transfer entropy, high-order transfer
                  entropy (stepwise composition)
synchronizability - symmetric Laplacian s(t) [Kini eq.], asymmetric
                  outdegree-Laplacian stability R(t) [Sun eq. 11-12]
resection       - electrode-to-resection-zone mapping (with dilation/
                  erosion robustness sweep), en-bloc control centrality,
                  nodal control centrality, sequential/hierarchical
                  propagation-guided virtual resection + inflection-point
                  minimum-intervention-target search
propagation     - HTE-derived propagation path extraction (hop layers,
                  outdegree ordering) used to drive sequential resection
simulate        - Jirsa Epileptor neural-mass model (eq. 1-8, Sun et al.)
                  for generating validated synthetic multi-node seizures
stats           - Wilcoxon rank-sum, DeLong correlated-ROC test,
                  permutation-based functional-data-analysis curve test,
                  Fisher-exact odds ratios, Bonferroni correction
outcome         - per-band ROC/AUC, confusion matrix, optimal threshold
pipeline        - end-to-end orchestration tying every stage together
"""

__version__ = "1.0.0"
