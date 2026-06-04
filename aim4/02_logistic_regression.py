"""
Aim 4 — Validation  |  2.4b Logistic Regression
=================================================
Trains a logistic regression classifier using pred_score as the sole
predictor and ClinVar binary labels (pathogenic / benign) as the outcome.

Leave-One-Out cross-validation is used to obtain unbiased predicted
probabilities for all 20 variants (avoids overfitting on N=20).

Threshold note
--------------
AUROC / Average Precision are threshold-free and measure how well the
out-of-fold probabilities *rank* the two classes. Accuracy / sensitivity /
specificity, by contrast, depend on the probability cutoff. We report these
classification metrics at the fixed, prespecified 0.5 cutoff only. A data-
driven "optimal" cutoff (e.g. Youden's J) is deliberately NOT used, because
choosing the threshold after seeing the labels inflates the apparent
accuracy and is not an honest out-of-sample estimate.

Outputs  (aim4/results/)
------------------------
  confusion_matrix.png
  roc_curve.png
  precision_recall_curve.png
  logistic_regression_metrics.txt

Input
-----
  aim3/results/clinvar_predictions.csv

Run from repo root
------------------
  python3 aim4/02_logistic_regression.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "clinvar_test_predictions_annotated.csv"
OUT_DIR  = ROOT / "aim4" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
COL_PATH = "#2aa8fd"
COL_BEN  = "#57a774"
sns.set_style("whitegrid")


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

df["label_bin"] = (df["clinvar_label"] == "pathogenic").astype(int)

X = df[["pred_score"]].values      # shape (20, 1)
y = df["label_bin"].values          # 1 = pathogenic, 0 = benign

n_total = len(df)
n_path  = y.sum()
n_ben   = n_total - n_path
print(f"Loaded {n_total} variants: {n_path} pathogenic, {n_ben} benign")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Logistic regression with Leave-One-Out CV
# ══════════════════════════════════════════════════════════════════════════════

clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", random_state=42,
                         max_iter=1000)
loo = LeaveOneOut()

# Out-of-fold predicted probabilities (probability of pathogenic = class 1)
oof_proba = cross_val_predict(clf, X, y, cv=loo, method="predict_proba")[:, 1]
oof_pred  = (oof_proba >= 0.5).astype(int)

# Fit on full data for coefficient reporting
clf.fit(X, y)
coef = float(clf.coef_[0][0])
intercept = float(clf.intercept_[0])

print(f"\nLogistic regression coefficient (pred_score): {coef:.4f}")
print(f"Intercept: {intercept:.4f}")
print("(Negative coefficient expected: lower pred_score → higher P(pathogenic))")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metrics from LOO predictions
# ══════════════════════════════════════════════════════════════════════════════

# Threshold-free metrics (depend only on the ranking of oof_proba)
auroc = roc_auc_score(y, oof_proba)
ap    = average_precision_score(y, oof_proba)

# ROC curve (used for the curve plot below; threshold-free)
fpr, tpr, roc_thresholds = roc_curve(y, oof_proba)


def threshold_metrics(threshold: float) -> dict:
    """Confusion-matrix-derived metrics at a given probability cutoff."""
    pred = (oof_proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "threshold":   threshold,
        "pred":        pred,
        "cm":          cm,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":    accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


m_default = threshold_metrics(0.5)


def _print_block(title: str, m: dict) -> None:
    print(f"\n  {title} (threshold = {m['threshold']:.3f}):")
    print(f"    Accuracy     : {m['accuracy']:.3f}  "
          f"({m['tp'] + m['tn']}/{n_total})")
    print(f"    Sensitivity  : {m['sensitivity']:.3f}  "
          f"({m['tp']}/{m['tp'] + m['fn']} pathogenic correct)")
    print(f"    Specificity  : {m['specificity']:.3f}  "
          f"({m['tn']}/{m['tn'] + m['fp']} benign correct)")


print(f"\nLOO-CV threshold-free metrics:")
print(f"  AUROC        : {auroc:.3f}")
print(f"  Avg Precision: {ap:.3f}")
print("\nLOO-CV classification metrics at the 0.5 cutoff:")
_print_block("Default cutoff", m_default)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save metrics
# ══════════════════════════════════════════════════════════════════════════════

metrics_text = textwrap.dedent(f"""
    Aim 4 — Logistic Regression Validation
    =======================================
    Input file     : {DATA_CSV.relative_to(ROOT)}
    Score used     : pred_score  (Random Forest predicted score)
    Predictor (X)  : pred_score  (continuous; lower = more pathogenic)
    Outcome (y)    : ClinVar binary label  (1 = pathogenic, 0 = benign)
    N total        : {n_total}  ({n_path} pathogenic, {n_ben} benign)

    Method
    ------
    Logistic regression (L2 penalty, C=1.0) with Leave-One-Out cross-validation.
    LOO-CV produces one unbiased predicted probability per variant, avoiding
    overfitting on the small N=20 dataset. All metrics below are computed from
    these out-of-fold probabilities.

    Model coefficients (fit on full data)
    --------------------------------------
    pred_score coefficient : {coef:.4f}
    Intercept              : {intercept:.4f}
    Note: negative coefficient confirms that lower pred_score → higher
    probability of being classified pathogenic.

    Threshold-free ranking metrics
    ------------------------------
    AUROC                  : {auroc:.4f}
    Average Precision (AP) : {ap:.4f}
    These depend only on how well the probabilities rank the two classes,
    not on any cutoff. AUROC = 1.0 means the scores separate pathogenic from
    benign perfectly by rank — i.e. SOME threshold classifies all 20 correctly.

    Threshold-dependent classification metrics
    ------------------------------------------
    The accuracy/sensitivity/specificity below are computed at the fixed,
    prespecified 0.5 probability cutoff. A data-driven "optimal" cutoff is
    deliberately not reported, as tuning the threshold on the labels would
    overstate out-of-sample performance.

    Default cutoff (threshold = 0.500)
        Accuracy        : {m_default['accuracy']:.4f}  ({m_default['tp'] + m_default['tn']}/{n_total} correct)
        Sensitivity     : {m_default['sensitivity']:.4f}  ({m_default['tp']}/{m_default['tp'] + m_default['fn']} pathogenic correctly identified)
        Specificity     : {m_default['specificity']:.4f}  ({m_default['tn']}/{m_default['tn'] + m_default['fp']} benign correctly identified)
        TP / TN / FP / FN: {m_default['tp']} / {m_default['tn']} / {m_default['fp']} / {m_default['fn']}

    Confusion matrix rows = actual labels, columns = predicted labels.
    Class order: 0 = benign, 1 = pathogenic.
""").strip()

metrics_path = OUT_DIR / "logistic_regression_metrics.txt"
metrics_path.write_text(metrics_text)
print(f"\nMetrics saved → {metrics_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Confusion matrix plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(m: dict, subtitle: str, out_path: Path) -> None:
    """Render a row-normalised confusion-matrix heatmap for one threshold."""
    cm = m["cm"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    sns.heatmap(
        cm_norm, annot=False, fmt="", cmap="Blues",
        xticklabels=["Benign", "Pathogenic"],
        yticklabels=["Benign", "Pathogenic"],
        linewidths=0.5, linecolor="white",
        ax=ax, cbar=True, vmin=0, vmax=1,
    )

    # Annotate each cell with count and row %
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j + 0.5, i + 0.5, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=color)

    n_correct = m["tp"] + m["tn"]
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True ClinVar label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix — Logistic Regression (LOO-CV)\n"
        f"{subtitle}  |  N = {n_total}  |  "
        f"Accuracy = {n_correct}/{n_total}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved  → {out_path}")


plot_confusion_matrix(
    m_default, "threshold = 0.50 (default)",
    OUT_DIR / "confusion_matrix.png",
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ROC curve
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(6, 5.5))

# Shaded AUC area  (fpr / tpr computed in the metrics section)
ax.fill_between(fpr, tpr, alpha=0.15, color=COL_PATH)
ax.plot(fpr, tpr, color=COL_PATH, linewidth=2.5,
        label=f"Logistic regression (AUROC = {auroc:.3f})")
ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
        linewidth=1.5, label="Random classifier")

# Mark the actual operating point at the fixed 0.5 cutoff. The curve is
# threshold-free (perfect rank → AUROC=1.0), but the classifier in practice
# runs at 0.5, where one benign is mislabelled (specificity < 1).
roc_fpr_op = m_default["fp"] / (m_default["fp"] + m_default["tn"])
roc_tpr_op = m_default["sensitivity"]
ax.scatter(roc_fpr_op, roc_tpr_op, color="#d62728", s=90, zorder=6,
           edgecolor="white", linewidth=1.2,
           label=f"Operating point @ 0.5 cutoff "
                 f"(sens {roc_tpr_op:.2f}, spec {m_default['specificity']:.2f})")

ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title(
    f"ROC Curve — Logistic Regression (LOO-CV)\n"
    f"BRCA1 ClinVar Variants  (N = {n_total})",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)

# AUROC text box (top-left, clear of the lower-right legend)
ax.text(0.03, 0.04, f"AUROC = {auroc:.3f}", transform=ax.transAxes,
        fontsize=11, va="bottom", ha="left", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

plt.tight_layout()
roc_path = OUT_DIR / "roc_curve.png"
fig.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved  → {roc_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Precision-Recall curve
# ══════════════════════════════════════════════════════════════════════════════

precision, recall, pr_thresholds = precision_recall_curve(y, oof_proba)
prevalence = y.mean()

fig, ax = plt.subplots(figsize=(6, 5.5))

ax.fill_between(recall, precision, alpha=0.15, color=COL_PATH)
ax.plot(recall, precision, color=COL_PATH, linewidth=2.5,
        label=f"Logistic regression (AP = {ap:.3f})")
ax.axhline(prevalence, color="#aaaaaa", linestyle="--", linewidth=1.5,
           label=f"Baseline (prevalence = {prevalence:.2f})")

# Mark the actual operating point at the fixed 0.5 cutoff (precision < 1 because
# one benign is predicted pathogenic, even though the ranking is perfect).
pr_recall_op = m_default["sensitivity"]
pr_precision_op = (m_default["tp"] / (m_default["tp"] + m_default["fp"])
                   if (m_default["tp"] + m_default["fp"]) > 0 else 0.0)
ax.scatter(pr_recall_op, pr_precision_op, color="#d62728", s=90, zorder=6,
           edgecolor="white", linewidth=1.2,
           label=f"Operating point @ 0.5 cutoff "
                 f"(recall {pr_recall_op:.2f}, prec {pr_precision_op:.2f})")

ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
ax.set_ylabel("Precision (PPV)", fontsize=11)
ax.set_title(
    f"Precision-Recall Curve — Logistic Regression (LOO-CV)\n"
    f"BRCA1 ClinVar Variants  (N = {n_total})",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, loc="lower left")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)

# AP text box (lower-right, clear of the lower-left legend)
ax.text(0.97, 0.04, f"Average Precision = {ap:.3f}", transform=ax.transAxes,
        fontsize=11, va="bottom", ha="right", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

plt.tight_layout()
pr_path = OUT_DIR / "precision_recall_curve.png"
fig.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved  → {pr_path}")

print(f"\nAll outputs written to: {OUT_DIR}")
