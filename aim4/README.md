# Aim 4 — External Validation of Predicted MAVE Scores

This folder validates the **Aim 3** machine-learning predictions (`pred_score`, a
Random Forest predicted MAVE functional score) against an **independent ClinVar
ground truth** for BRCA1 missense variants. The question is simple: *do the
predicted functional scores actually separate clinically pathogenic variants from
benign ones?*

- **Lower `pred_score` ⇒ more loss-of-function ⇒ more likely pathogenic.**
- Validation set: **20 BRCA1 missense variants** with non-conflicting ClinVar
  classifications — **10 pathogenic, 10 benign**.

---

## Input

```
aim3/results/clinvar_predictions.csv
```

One row per variant. Columns used by these scripts:

| Column          | Meaning                                                        |
|-----------------|----------------------------------------------------------------|
| `variant`       | Protein-level variant ID (e.g. `M1775R`)                       |
| `clinvar_label` | Ground-truth class, normalised to `pathogenic` / `benign`      |
| `pred_score`    | Aim 3 predicted MAVE score (lower = more pathogenic)           |

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
  classification threshold, rug ticks, p-value annotations, and a callout for the
  one threshold-misclassified variant.

**Outputs:** `results/distribution_stats.txt`, `results/kde_distribution.png`

### `02_logistic_regression.py`
Fits a logistic regression with `pred_score` as the sole predictor and the ClinVar
label as the outcome, evaluated by **Leave-One-Out cross-validation (LOO-CV)** —
the appropriate choice for N=20 (one unbiased out-of-fold probability per variant,
no leakage).

**Outputs:**
- `results/logistic_regression_metrics.txt`
- `results/confusion_matrix.png` (0.5 cutoff)
- `results/confusion_matrix_youden.png` (Youden-optimal cutoff)
- `results/roc_curve.png`, `results/precision_recall_curve.png`

---

## Reading the metrics: threshold-free vs threshold-dependent

This is the most important nuance in the Aim 4 results.

- **AUROC** and **Average Precision** are *threshold-free*. They measure only how
  well the predicted probabilities **rank** pathogenic above benign.
  **AUROC = 1.0 means the scores separate the two classes perfectly by rank** —
  i.e. *some* cutoff classifies all 20 variants correctly.
- **Accuracy / sensitivity / specificity** depend on the **probability cutoff**.
  With near-separable data and LOO on N=20, the fitted logistic probabilities are
  **miscalibrated**, so the naive 0.5 cutoff is *not* the optimal operating point
  and understates performance.

Because of this, `02` reports classification metrics at **two** thresholds so the
difference is explicit:

| Threshold                | What it is                                                        |
|--------------------------|-------------------------------------------------------------------|
| **0.50 (default)**       | Naive probability cutoff. Illustrative only.                      |
| **Youden-optimal**       | Maximises `sensitivity + specificity − 1` on the out-of-fold ROC. |

Since AUROC = 1.0, the Youden cutoff achieves perfect separation (20/20), whereas
the 0.5 cutoff misclassifies several pathogenic variants as benign — a calibration
artifact, **not** a discrimination failure. **Report the threshold-free metrics
(AUROC / AP) as the headline result**; treat the 0.5-cutoff confusion matrix as
illustrative. The exact Youden threshold value is written into
`logistic_regression_metrics.txt` at run time.

---

## Key findings

**Distribution analysis** — the two classes are cleanly separated:

| Statistic            | Value          |
|----------------------|----------------|
| Mann–Whitney U       | 0  (p ≈ 1.8e-4) |
| Kolmogorov–Smirnov   | 1.0 (p ≈ 1.1e-5) |
| Median (pathogenic)  | ≈ −0.91        |
| Median (benign)      | ≈ −0.08        |

**Logistic regression (LOO-CV)** — perfect ranking:

| Metric                          | Value                |
|---------------------------------|----------------------|
| AUROC                           | 1.000                |
| Average Precision               | 1.000                |
| Accuracy @ 0.5 cutoff           | 0.85 (17/20)         |
| Accuracy @ Youden cutoff        | 1.00 (20/20)         |

> **Note on the two scripts' misclassifications.** The distribution script's
> fixed score threshold flags **H1862L** (a benign variant just below the cutoff)
> as the single error. The logistic-regression 0.5 cutoff instead misses a few
> *pathogenic* variants. These are different operating points on the same
> perfectly-ranked data — do not conflate them in the writeup.

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
    ├── confusion_matrix_youden.png     # Youden-optimal cutoff
    ├── roc_curve.png
    └── precision_recall_curve.png
```
