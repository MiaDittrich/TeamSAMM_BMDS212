"""
Aim 4 — Validation  |  2.4a Distribution Analysis
===================================================
Compares the distribution of predicted scores (pred_score) between
ClinVar-labelled pathogenic and benign BRCA1 missense variants.

Tests
-----
  • Mann-Whitney U  — difference in central tendency
  • Kolmogorov-Smirnov — difference in overall distributional shape

Plot
----
  kde_distribution.png — overlaid KDEs with classification threshold,
  p-value annotation, and labels for misclassified variants.

Input
-----
  aim3/results/clinvar_predictions.csv

Outputs  (aim4/results/)
------------------------
  distribution_stats.txt
  kde_distribution.png

Run from repo root
------------------
  python3 aim4/01_distribution_analysis.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_CSV  = ROOT / "clinvar_test_predictions_annotated.csv"
OUT_DIR   = ROOT / "aim4" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
COL_PATH  = "#2aa8fd"   # pathogenic
COL_BEN   = "#57a774"   # benign
THRESHOLD = -0.1726     # training-set median from Spencer's model


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_CSV)

# Normalise label casing ("Pathogenic"/"Benign" → "pathogenic"/"benign") so the
# exact-string matching below is robust to the Aim 3 export's capitalisation.
df["clinvar_label"] = df["clinvar_label"].str.strip().str.lower()

# Validate the upstream export before relying on exact-string label matching.
# (If Aim 3 ever emits raw ClinVar terms, e.g. "Likely pathogenic", a silent
#  miscount here would otherwise pass everything through as benign.)
_expected_labels = {"pathogenic", "benign"}
_unexpected = set(df["clinvar_label"].unique()) - _expected_labels
if _unexpected:
    raise ValueError(
        f"Unexpected clinvar_label value(s) {sorted(_unexpected)}; "
        f"expected only {sorted(_expected_labels)}. "
        "Normalise the Aim 3 export before validation."
    )

# Binary label
df["label_bin"] = (df["clinvar_label"] == "pathogenic").astype(int)

path_scores = df.loc[df["label_bin"] == 1, "pred_score"].values
ben_scores  = df.loc[df["label_bin"] == 0, "pred_score"].values

n_path = len(path_scores)
n_ben  = len(ben_scores)
n_total = len(df)

print(f"Loaded {n_total} variants: {n_path} pathogenic, {n_ben} benign")

# Threshold-based classification and misclassifications (derived, not hardcoded):
# pred_score < THRESHOLD → predicted pathogenic.
df["pred_threshold_class"] = np.where(df["pred_score"] < THRESHOLD,
                                      "pathogenic", "benign")
misclassified = df.loc[df["pred_threshold_class"] != df["clinvar_label"],
                       ["variant", "clinvar_label", "pred_score"]]
print(f"Misclassified at threshold {THRESHOLD}: {len(misclassified)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

mw_stat, mw_p = stats.mannwhitneyu(path_scores, ben_scores, alternative="two-sided")
ks_stat, ks_p = stats.ks_2samp(path_scores, ben_scores)

path_med = float(np.median(path_scores))
ben_med  = float(np.median(ben_scores))

print(f"\nMann-Whitney U = {mw_stat:.1f},  p = {mw_p:.4g}")
print(f"KS statistic  = {ks_stat:.4f},  p = {ks_p:.4g}")
print(f"Median pred_score — pathogenic: {path_med:.4f}  benign: {ben_med:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Save stats
# ══════════════════════════════════════════════════════════════════════════════

if misclassified.empty:
    misclass_text = "None — all variants classified correctly at this threshold."
else:
    misclass_text = "\n    ".join(
        f"{r.variant}  (ClinVar: {r.clinvar_label},  pred_score = {r.pred_score:.4f})"
        for r in misclassified.itertuples(index=False)
    )

stats_path = OUT_DIR / "distribution_stats.txt"
stats_text = textwrap.dedent(f"""
    Aim 4 — Distribution Analysis of Predicted Scores
    =======================================================
    Score used   : pred_score  (Random Forest predicted score;
                   lower = more loss-of-function = more pathogenic)
    Input file   : {DATA_CSV.relative_to(ROOT)}
    N total      : {n_total}
    N pathogenic : {n_path}  (ClinVar: Pathogenic / Likely pathogenic /
                              Pathogenic/Likely pathogenic)
    N benign     : {n_ben}   (ClinVar: Benign / Likely benign /
                              Benign/Likely benign)

    Descriptive statistics
    ----------------------
    Pathogenic — median: {path_med:.4f},  mean: {path_scores.mean():.4f},
                 std: {path_scores.std():.4f},
                 range: [{path_scores.min():.4f}, {path_scores.max():.4f}]
    Benign     — median: {ben_med:.4f},  mean: {ben_scores.mean():.4f},
                 std: {ben_scores.std():.4f},
                 range: [{ben_scores.min():.4f}, {ben_scores.max():.4f}]

    Mann-Whitney U test (two-sided)
    --------------------------------
    Tests whether the central tendency of pred_score differs between groups.
    U statistic : {mw_stat:.1f}
    p-value     : {mw_p:.4g}
    Interpretation: {"Significant" if mw_p < 0.05 else "Not significant"} at α = 0.05
      → {"Pathogenic variants have significantly lower pred_score than benign variants." if mw_p < 0.05 else "No significant difference in central tendency detected."}

    Kolmogorov-Smirnov test (two-sample)
    --------------------------------------
    Tests whether the overall score distributions differ in shape.
    KS statistic : {ks_stat:.4f}
    p-value      : {ks_p:.4g}
    Interpretation: {"Significant" if ks_p < 0.05 else "Not significant"} at α = 0.05
      → {"The two distributions differ significantly in shape." if ks_p < 0.05 else "No significant difference in distributional shape detected."}

    Classification threshold (from Spencer's model)
    -------------------------------------------------
    Threshold : {THRESHOLD}  (training-set median of training scores)
    Variants with pred_score < threshold → classified as Pathogenic

    Misclassified variants (at this threshold)
    ------------------------------------------
    {misclass_text}
""").strip()

stats_path.write_text(stats_text)
print(f"\nStats saved → {stats_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. KDE plot
# ══════════════════════════════════════════════════════════════════════════════

sns.set_style("whitegrid")
plt.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13})
fig, ax = plt.subplots(figsize=(9, 5))

# KDE curves
sns.kdeplot(path_scores, ax=ax, color=COL_PATH, linewidth=2.5,
            fill=True, alpha=0.25, label=f"Pathogenic (n={n_path})")
sns.kdeplot(ben_scores,  ax=ax, color=COL_BEN,  linewidth=2.5,
            fill=True, alpha=0.25, label=f"Benign (n={n_ben})")

y_min = ax.get_ylim()[0]

# Median lines
ax.axvline(path_med, color=COL_PATH, linestyle="--", linewidth=1.4, alpha=0.7)
ax.axvline(ben_med,  color=COL_BEN,  linestyle="--", linewidth=1.4, alpha=0.7)

# Classification threshold
ax.axvline(THRESHOLD, color="#555555", linestyle=":", linewidth=2.0,
           label=f"Classification threshold ({THRESHOLD})")

# Annotate medians
ylim = ax.get_ylim()
ax.text(path_med - 0.01, ylim[1] * 0.92, f"median\n{path_med:.3f}",
        color=COL_PATH, fontsize=11, ha="right", va="top")
ax.text(ben_med + 0.01, ylim[1] * 0.92, f"median\n{ben_med:.3f}",
        color=COL_BEN, fontsize=11, ha="left", va="top")

# Annotate misclassified variants (derived from the threshold, not hardcoded).
# To avoid clutter on the larger test set, label individually only when there
# are a few; otherwise summarise the count.
if len(misclassified) > 0 and len(misclassified) <= 4:
    for i, r in enumerate(misclassified.itertuples(index=False)):
        ax.annotate(
            f"{r.variant}\n({r.clinvar_label}, misclassified)",
            xy=(r.pred_score, y_min + (ylim[1] - y_min) * 0.05),
            xytext=(r.pred_score - 0.35, ylim[1] * (0.45 - 0.12 * i)),
            fontsize=11, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.85),
        )
elif len(misclassified) > 4:
    ax.text(0.03, 0.55, f"{len(misclassified)} misclassified\nat threshold",
            transform=ax.transAxes, fontsize=11, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.85))

# p-value annotation box
pval_text = (
    f"Mann-Whitney U:  p = {mw_p:.4g}\n"
    f"KS test:         p = {ks_p:.4g}"
)
ax.text(0.97, 0.97, pval_text, transform=ax.transAxes,
        fontsize=12, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

ax.set_xlabel("Predicted Score  (lower = more loss-of-function)", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title(
    "Distribution of Predicted Scores by ClinVar Classification\n"
    f"BRCA1 Missense Variants  (N = {n_total})",
    fontsize=16, fontweight="bold"
)

# Legend
legend_handles = [
    mpatches.Patch(color=COL_PATH, alpha=0.7, label=f"Pathogenic (n={n_path})"),
    mpatches.Patch(color=COL_BEN,  alpha=0.7, label=f"Benign (n={n_ben})"),
    plt.Line2D([0], [0], color="#555555", linestyle=":", linewidth=2,
               label=f"Threshold = {THRESHOLD}"),
    plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=1.4,
               label="Group medians"),
]
ax.legend(handles=legend_handles, fontsize=12, loc="upper left")

plt.tight_layout()
plot_path = OUT_DIR / "kde_distribution.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved  → {plot_path}")
