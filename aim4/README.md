# Aim 4 — External Validation of Predicted Scores

This folder validates the **Aim 3** machine-learning predictions (`pred_score`, a
Random Forest predicted score) against an **independent ClinVar
ground truth** for BRCA1 missense variants. The question is simple: *do the
predicted functional scores actually separate clinically pathogenic variants from
benign ones?*

- **Lower `pred_score` ⇒ more loss-of-function ⇒ more likely pathogenic.**
- Validation set: **28 BRCA1 missense variants** with non-conflicting ClinVar
  classifications — **14 pathogenic, 14 benign**.

---

## Input

```
clinvar_test_predictions_annotated.csv   (repo root)
```

One row per variant. Columns used by these scripts:

| Column          | Meaning                                                        |
|-----------------|----------------------------------------------------------------|
| `variant`       | Protein-level variant ID (e.g. `M1775R`)                       |
| `clinvar_label` | Ground-truth class, normalised to `pathogenic` / `benign`      |
| `pred_score`    | Aim 3 predicted score (lower = more pathogenic)           |

> Both scripts assert that `clinvar_label` contains **only** `pathogenic` /
> `benign` and abort otherwise — this guards against a silently mis-counted
> validation set if the Aim 3 export format ever changes (e.g. raw ClinVar terms
> like "Likely pathogenic").

---

## Scripts

### `01_distribution_analysis.py`
Tests whether the `pred_score` distributions of the two classes differ.

- **Mann–Whitney U** (two-sided) — difference in central tendency.
- **Kolmogorov–Smirnov** (two-sample) — difference in overall distribution shape.
- Produces an overlaid KDE plot with group medians, the score-based
  classification threshold, p-value annotations, and callouts for any
  threshold-misclassified variants.

**Outputs:** `results/distribution_stats.txt`, `results/kde_distribution.png`

### `02_logistic_regression.py`
Fits a logistic regression with `pred_score` as the sole predictor and the ClinVar
label as the outcome, evaluated by **Leave-One-Out cross-validation (LOO-CV)** —
the appropriate choice for a small N (one unbiased out-of-fold probability per
variant, no leakage).

**Outputs:**
- `results/logistic_regression_metrics.txt`
- `results/confusion_matrix.png` (0.5 cutoff)
- `results/roc_curve.png`, `results/precision_recall_curve.png`

---

## Reading the metrics: threshold-free vs threshold-dependent

This is the most important nuance in the Aim 4 results.

- **AUROC** and **Average Precision** are *threshold-free*. They measure only how
  well the predicted probabilities **rank** pathogenic above benign.
  **AUROC = 1.0 means the scores separate the two classes perfectly by rank** —
  i.e. *some* cutoff classifies all variants correctly.
- **Accuracy / sensitivity / specificity** depend on the **probability cutoff**.
  We report these at the fixed, prespecified **0.5 cutoff only**.

A data-driven "optimal" cutoff (e.g. Youden's J, which maximises
`sensitivity + specificity − 1`) is **deliberately not used**: choosing the
threshold after seeing the labels inflates the apparent accuracy and is not an
honest out-of-sample estimate. The 0.5 cutoff is fixed in advance and reported
as-is.

---

## Key findings

**Distribution analysis** — the two classes are cleanly separated:

| Statistic            | Value             |
|----------------------|-------------------|
| Mann–Whitney U       | 0.0  (p ≈ 7.5e-6) |
| Kolmogorov–Smirnov   | 1.0  (p ≈ 5.0e-8) |
| Median (pathogenic)  | ≈ −0.847          |
| Median (benign)      | ≈ −0.004          |

**Logistic regression (LOO-CV)** at the fixed 0.5 cutoff:

| Metric                          | Value                |
|---------------------------------|----------------------|
| AUROC (threshold-free)          | 1.000                |
| Average Precision (threshold-free) | 1.000             |
| Accuracy @ 0.5 cutoff           | 0.964 (27/28)        |
| Sensitivity @ 0.5 cutoff        | 1.000 (14/14)        |
| Specificity @ 0.5 cutoff        | 0.929 (13/14)        |

> **Note on the two scripts' misclassifications.** The distribution script's
> fixed score threshold flags 2 benign variants (**M1652T**, **I1723T**) just
> past the cutoff. The logistic-regression 0.5 cutoff instead misclassifies 1
> benign variant as pathogenic. These are different operating points on the same
> perfectly-ranked data (AUROC = 1.0) — do not conflate them in the writeup.

---

## Running

From the repository root (the scripts resolve their own paths, so the working
directory does not matter):

```bash
python3 aim4/01_distribution_analysis.py
python3 aim4/02_logistic_regression.py
```

All outputs are (re)written to `aim4/results/`.

### Dependencies
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

---

## Files

```
aim4/
├── 01_distribution_analysis.py
├── 02_logistic_regression.py
├── README.md
└── results/
    ├── distribution_stats.txt
    ├── kde_distribution.png
    ├── logistic_regression_metrics.txt
    ├── confusion_matrix.png            # 0.5 cutoff
    ├── roc_curve.png
    └── precision_recall_curve.png
```
