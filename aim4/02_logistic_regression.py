"""
Aim 4 — Validation  |  2.4b Logistic Regression
=================================================
Trains a logistic regression classifier using pred_score as the sole
predictor and ClinVar binary labels (pathogenic / benign) as the outcome.

Leave-One-Out cross-validation is used to obtain unbiased predicted
probabilities for all 20 variants (avoids overfitting on N=20).

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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "aim3" / "results" / "clinvar_predictions.csv"
OUT_DIR  = ROOT / "aim4" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
COL_PATH = "#d62728"
COL_BEN  = "#1f77b4"
sns.set_style("whitegrid")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_CSV)
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

auroc   = roc_auc_score(y, oof_proba)
ap      = average_precision_score(y, oof_proba)
acc     = accuracy_score(y, oof_pred)
cm      = confusion_matrix(y, oof_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"\nLOO-CV metrics:")
print(f"  AUROC        : {auroc:.3f}")
print(f"  Avg Precision: {ap:.3f}")
print(f"  Accuracy     : {acc:.3f}  ({int(acc * n_total)}/{n_total})")
print(f"  Sensitivity  : {sensitivity:.3f}  ({tp}/{tp+fn} pathogenic correct)")
print(f"  Specificity  : {specificity:.3f}  ({tn}/{tn+fp} benign correct)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save metrics
# ══════════════════════════════════════════════════════════════════════════════

metrics_text = textwrap.dedent(f"""
    Aim 4 — Logistic Regression Validation
    =======================================
    Input file     : {DATA_CSV.relative_to(ROOT)}
    Score used     : pred_score  (Random Forest predicted MAVE functional score)
    Predictor (X)  : pred_score  (continuous; lower = more pathogenic)
    Outcome (y)    : ClinVar binary label  (1 = pathogenic, 0 = benign)
    N total        : {n_total}  ({n_path} pathogenic, {n_ben} benign)

    Method
    ------
    Logistic regression (L2 penalty, C=1.0) with Leave-One-Out cross-validation.
    LOO-CV produces one unbiased predicted probability per variant, avoiding
    overfitting on the small N=20 dataset. Metrics are computed from these
    out-of-fold probabilities; the confusion matrix uses threshold = 0.5.

    Model coefficients (fit on full data)
    --------------------------------------
    pred_score coefficient : {coef:.4f}
    Intercept              : {intercept:.4f}
    Note: negative coefficient confirms that lower pred_score → higher
    probability of being classified pathogenic.

    LOO-CV Classification Metrics
    ------------------------------
    AUROC                  : {auroc:.4f}
    Average Precision (AP) : {ap:.4f}
    Accuracy               : {acc:.4f}  ({int(acc * n_total)}/{n_total} correct)
    Sensitivity            : {sensitivity:.4f}  ({tp}/{tp+fn} pathogenic correctly identified)
    Specificity            : {specificity:.4f}  ({tn}/{tn+fp} benign correctly identified)
    True Positives         : {tp}
    True Negatives         : {tn}
    False Positives        : {fp}
    False Negatives        : {fn}

    Confusion matrix rows = actual labels, columns = predicted labels.
    Class order: 0 = benign, 1 = pathogenic.
""").strip()

metrics_path = OUT_DIR / "logistic_regression_metrics.txt"
metrics_path.write_text(metrics_text)
print(f"\nMetrics saved → {metrics_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Confusion matrix plot
# ══════════════════════════════════════════════════════════════════════════════

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

ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True ClinVar label", fontsize=11)
ax.set_title(
    f"Confusion Matrix — Logistic Regression (LOO-CV)\n"
    f"N = {n_total}  |  Accuracy = {int(acc*n_total)}/{n_total}",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved  → {cm_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ROC curve
# ══════════════════════════════════════════════════════════════════════════════

fpr, tpr, thresholds = roc_curve(y, oof_proba)

fig, ax = plt.subplots(figsize=(6, 5.5))

# Shaded AUC area
ax.fill_between(fpr, tpr, alpha=0.15, color=COL_PATH)
ax.plot(fpr, tpr, color=COL_PATH, linewidth=2.5,
        label=f"Logistic regression (AUROC = {auroc:.3f})")
ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--",
        linewidth=1.5, label="Random classifier")

# Mark the operating point closest to (0, 1)
dist_to_ideal = np.sqrt(fpr**2 + (1 - tpr)**2)
best_idx = int(np.argmin(dist_to_ideal))
ax.scatter(fpr[best_idx], tpr[best_idx], color=COL_PATH,
           s=80, zorder=5, label=f"Best threshold = {thresholds[best_idx]:.3f}")

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

# AUROC text box
ax.text(0.97, 0.10, f"AUROC = {auroc:.3f}", transform=ax.transAxes,
        fontsize=11, va="bottom", ha="right", fontweight="bold",
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

# AP text box
ax.text(0.03, 0.10, f"Average Precision = {ap:.3f}", transform=ax.transAxes,
        fontsize=11, va="bottom", ha="left", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

plt.tight_layout()
pr_path = OUT_DIR / "precision_recall_curve.png"
fig.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved  → {pr_path}")

print(f"\nAll outputs written to: {OUT_DIR}")
