#!/usr/bin/env python3
"""
run_cohort.py
==============
Cohort-level outcome-prediction analysis, replicating Kini et al. Fig. 3C
(per-band ROC overlay + DeLong comparison), Supplementary Table S2
(confusion matrix at the Youden-optimal broadband threshold), and the
permutation-based functional-data-analysis significance test on the
time-normalized s(t) curves (Fig. 3A).

Input: a CSV with one row per patient (or per seizure -- aggregate to one
row per patient with e.g. median across seizures before running this,
matching "network measures were computed on all seizures ... and averaged
within a patient's group of seizures"), produced by running
`run_single_patient.py` per patient and collecting `band_delta_s` into
rows, OR built directly from `vrpipeline.pipeline.run_pipeline_a` in your
own driver script.

Required CSV columns
---------------------
    patient_id, outcome                      (outcome: 1 = good, 0 = poor)
    delta_s_<band>                           for each band you want scored,
                                              e.g. delta_s_broadband_cc,
                                              delta_s_alpha_theta, ...

Usage
-----
    python run_cohort.py --csv cohort_features.csv --out-dir results/cohort
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrpipeline.outcome import roc_analysis, predict_at_threshold, confusion_report
from vrpipeline.stats import delong_roc_test, roc_auc_ci_delong, bonferroni_alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="results/cohort")
    ap.add_argument("--reference-band", type=str, default="broadband_cc",
                     help="band to test all others against via DeLong")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "outcome" not in df.columns or "patient_id" not in df.columns:
        raise SystemExit("CSV must contain 'patient_id' and 'outcome' columns")

    y = df["outcome"].astype(int).values
    band_cols = [c for c in df.columns if c.startswith("delta_s_")]
    if not band_cols:
        raise SystemExit("no delta_s_<band> columns found in CSV")

    print(f"Cohort: n={len(df)}, good outcome={int(y.sum())}, "
          f"poor outcome={int((1 - y).sum())}\n")

    print("== Per-band ROC / AUC (with DeLong 95% CI) ==")
    print("   (direction auto-selected: some networks predict good outcome "
          "via a DECREASE in the feature, e.g. Kini et al.'s broadband "
          "Delta s(t) -- we try both signs and keep whichever gives "
          "AUC >= 0.5, reporting which direction was used)\n")
    results = {}
    for col in band_cols:
        band = col.replace("delta_s_", "")
        feat = df[col].values
        valid = ~np.isnan(feat)
        roc_hi = roc_analysis(y[valid], feat[valid], higher_is_positive=True)
        roc_lo = roc_analysis(y[valid], feat[valid], higher_is_positive=False)
        if roc_hi["auc"] >= roc_lo["auc"]:
            roc, direction, signed_feat = roc_hi, "higher = good outcome", feat
        else:
            roc, direction, signed_feat = roc_lo, "lower = good outcome", feat
        ci = roc_auc_ci_delong(y[valid],
                                signed_feat[valid] if direction.startswith("higher")
                                else -signed_feat[valid])
        results[band] = {"roc": roc, "ci": ci, "feat": feat,
                          "higher_is_positive": direction.startswith("higher")}
        print(f"  {band:14s} AUC = {roc['auc']:.3f}  "
              f"(95% CI {ci['ci_low']:.2f}-{ci['ci_high']:.2f})  "
              f"[{direction}]  "
              f"best_thr={roc['best_threshold']:+.4f}  "
              f"sens={roc['best_sensitivity']:.2f}  "
              f"spec={roc['best_specificity']:.2f}")

    def _signed(band):
        r = results[band]
        return r["feat"] if r["higher_is_positive"] else -r["feat"]

    ref = args.reference_band
    if ref in results:
        print(f"\n== DeLong test: each band vs. reference band "
              f"'{ref}' ==")
        alpha_corr = bonferroni_alpha(len(band_cols) - 1)
        ref_feat = _signed(ref)
        for band, r in results.items():
            if band == ref:
                continue
            band_feat = _signed(band)
            valid = ~(np.isnan(ref_feat) | np.isnan(band_feat))
            dl = delong_roc_test(y[valid], ref_feat[valid], band_feat[valid])
            sig = "***" if dl["p_value"] < alpha_corr else ""
            print(f"  {ref} (AUC={dl['auc_a']:.3f}) vs {band} "
                  f"(AUC={dl['auc_b']:.3f}): z={dl['z']:+.3f}, "
                  f"p={dl['p_value']:.4g} {sig}"
                  f"  [Bonferroni alpha={alpha_corr:.4f}]")

        print(f"\n== Confusion matrix at Youden-optimal threshold "
              f"('{ref}') ==")
        roc = results[ref]["roc"]
        feat = results[ref]["feat"]
        valid = ~np.isnan(feat)
        y_pred = predict_at_threshold(feat[valid], roc["best_threshold"],
                                       results[ref]["higher_is_positive"])
        cm = confusion_report(y[valid], y_pred)
        print(f"  TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")
        print(f"  Accuracy={cm['accuracy']:.2f}  TPR={cm['TPR']:.2f}  "
              f"TNR={cm['TNR']:.2f}")

    out_csv = out_dir / "per_band_roc_summary.csv"
    rows = []
    for band, r in results.items():
        rows.append({"band": band, "auc": r["roc"]["auc"],
                     "ci_low": r["ci"]["ci_low"], "ci_high": r["ci"]["ci_high"],
                     "best_threshold": r["roc"]["best_threshold"],
                     "sensitivity": r["roc"]["best_sensitivity"],
                     "specificity": r["roc"]["best_specificity"]})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSummary written to {out_csv}")


if __name__ == "__main__":
    main()
