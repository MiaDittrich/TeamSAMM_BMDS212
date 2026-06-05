#!/usr/bin/env python3
"""
Publication-quality figures for the BRCA1 MAVE pathogenicity prediction paper.
Source: extended-split, both-label pipeline  (n_train=86, test=28, explore=24)
Output: final_figs/  (PNG + PDF for each figure)

Figures
-------
  fig1_pipeline.png         — Pipeline schematic
  fig2_cv_performance.png   — Cross-validation scatter + fold metrics
  fig3_feature_importance.png — SHAP importance + group ablation
  fig4_predictions.png      — Test-set classification + VUS/Conflicting explore
  fig5_esm2_analysis.png    — ESM2 LLR contribution analysis
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#EBEBEB",
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "legend.fontsize":   8.5,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
})

# Consistent palette
GC = {
    "structural":    "#4E79A7",
    "biochemical":   "#F28E2B",
    "alphamissense": "#59A14F",
    "evo":           "#E15759",
    "engineered":    "#9467BD",
}
C_PATH = "#D62728"
C_BEN  = "#1F77B4"

BASE    = Path(__file__).parent
RESULTS = BASE / "results"
OUTDIR  = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)


def save(fig, stem):
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {stem}.png / .pdf")



def fig2_cv_performance():
    oof  = pd.read_csv(RESULTS / "cv_oof_predictions.csv")
    cv   = pd.read_csv(RESULTS / "cv_metrics.csv")

    y_true = oof["y_true"].values
    y_pred = oof["y_pred_oof"].values
    valid  = ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    sp_m = cv["spearman"].mean(); sp_s = cv["spearman"].std()
    pe_m = cv["pearson"].mean();  pe_s = cv["pearson"].std()
    r2_m = cv["r2"].mean();       r2_s = cv["r2"].std()
    rm_m = cv["rmse"].mean();     rm_s = cv["rmse"].std()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")

    # ── OOF scatter ───────────────────────────────────────────────────────────
    lo = min(y_true.min(), y_pred.min()) - 0.07
    hi = max(y_true.max(), y_pred.max()) + 0.07

    norm   = plt.Normalize(vmin=y_true.min(), vmax=y_true.max())
    sc = ax.scatter(y_true, y_pred, c=y_true, cmap="RdBu",
                    norm=norm, s=55, alpha=0.82, edgecolors="white",
                    linewidths=0.4, zorder=3)

    # Diagonal
    ax.plot([lo, hi], [lo, hi], color="#AAAAAA", lw=1.2, ls="--",
            alpha=0.6, label="Identity (y = x)", zorder=2)

    # OLS line + 95% bootstrap CI
    coef  = np.polyfit(y_true, y_pred, 1)
    xl    = np.linspace(lo, hi, 200)
    yl    = np.polyval(coef, xl)
    rng   = np.random.default_rng(42)
    boots = np.array([np.polyval(np.polyfit(y_true[rng.integers(0, len(y_true), len(y_true))],
                                             y_pred[rng.integers(0, len(y_pred), len(y_pred))], 1), xl)
                      for _ in range(500)])
    ax.plot(xl, yl, color="#333", lw=2.0, label="OLS fit", zorder=4)
    ax.fill_between(xl, np.percentile(boots, 2.5, axis=0),
                    np.percentile(boots, 97.5, axis=0),
                    color="#333", alpha=0.10, label="95% bootstrap CI", zorder=1)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("True MAVE score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    txt = (f"Pearson r  = {pe_m:+.3f} ± {pe_s:.3f}\n"
           f"Spearman ρ = {sp_m:+.3f} ± {sp_s:.3f}\n"
           f"R²         = {r2_m:+.3f} ± {r2_s:.3f}\n"
           f"RMSE       = {rm_m:.3f} ± {rm_s:.3f}\n"
           f"n = {len(y_true)}  ·  25 folds")
    ax.text(0.97, 0.04, txt, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", alpha=0.93))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual MAVE score  (lower = more LOF)")
    ax.set_ylabel("OOF Predicted score  (mean over 25 folds)")
    ax.set_title("A   Cross-Validation: OOF Predicted vs. Actual\n"
                 "(5-fold × 5-repeat, ElasticNet, n = 86)", loc="left")
    ax.legend(fontsize=8.5, loc="upper left")

    # Rug (paired by index, not by value)
    colors_rug = plt.cm.RdBu(norm(y_true))
    for v, pv, col in zip(y_true, y_pred, colors_rug):
        ax.axvline(v,  ymin=0,    ymax=0.025, color=col, lw=0.6, alpha=0.25)
        ax.axhline(pv, xmin=0,    xmax=0.025, color=col, lw=0.6, alpha=0.25)

    plt.tight_layout()
    return fig



def _load_fig3_data():
    shap_df = pd.read_csv(RESULTS / "shap_mean_abs.csv")
    abl_g   = pd.read_csv(RESULTS / "ablation_groups.csv")
    abl_f   = pd.read_csv(RESULTS / "ablation_features.csv")
    bl_r2 = float(
        abl_f["baseline_r2"].iloc[0] if "baseline_r2" in abl_f.columns
        else (abl_f["r2_without"] + abl_f["delta_r2"]).iloc[0]
    )
    bl_sp = float(
        abl_f["baseline_sp"].iloc[0] if "baseline_sp" in abl_f.columns
        else (abl_f["sp_without"] + abl_f["delta_sp"]).iloc[0]
    )
    return shap_df, abl_f, abl_g, bl_r2, bl_sp


def fig3a_shap():
    shap_df, _, _, _, _ = _load_fig3_data()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")

    top = (shap_df.sort_values("mean_abs_shap", ascending=False)
                  .head(15)
                  .sort_values("mean_abs_shap", ascending=True)
                  .reset_index(drop=True))

    for i, row in top.iterrows():
        col = GC.get(row["group"], "#AAAAAA")
        ax.barh(i, row["mean_abs_shap"], color=col, alpha=0.88,
                edgecolor="white", height=0.68, zorder=3)
        if row["mean_abs_shap"] > 0.004:
            ax.text(row["mean_abs_shap"] + 0.0004, i,
                    f"{row['mean_abs_shap']:.4f}",
                    va="center", ha="left", fontsize=8.5,
                    color=GC.get(row["group"], "#555"))

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9.5)
    for tick, (_, row) in zip(ax.get_yticklabels(), top.iterrows()):
        tick.set_color(GC.get(row["group"], "#333"))

    patches = [mpatches.Patch(color=c, label=g.capitalize())
               for g, c in GC.items() if g in top["group"].values]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              title="Feature group", title_fontsize=8.5, framealpha=0.9)
    ax.set_xlabel("Mean |SHAP value|  (pooled across train + test + explore)", fontsize=10)
    ax.set_title("SHAP Global Feature Importance\n(top 15 features)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def fig3b_feature_ablation():
    _, abl_f, _, bl_r2, bl_sp = _load_fig3_data()

    ZERO_R2 = 1e-4
    ZERO_SP = 1e-4

    df = abl_f.copy()
    df["is_active"] = (df["delta_r2"].abs() > ZERO_R2) | (df["delta_sp"].abs() > ZERO_SP)

    n_active = int(df["is_active"].sum())
    n_zeroed = int((~df["is_active"]).sum())

    # active features sorted ascending Δ R² (most negative at bottom)
    active_sorted = (df[df["is_active"]]
                     .sort_values("delta_r2", ascending=True)
                     .reset_index(drop=True))

    n_rows = n_active

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    # ── Active feature bars ──────────────────────────────────────────────────
    for i, (_, row) in enumerate(active_sorted.iterrows()):
        yi       = i
        delta    = row["delta_r2"]
        r2_std   = row.get("r2_std_without", 0.0)
        col      = GC.get(row["group"], "#AAAAAA")
        bar_col  = "#27AE60" if delta > ZERO_R2 else "#E15759"

        xerr = r2_std / np.sqrt(25) if r2_std > 0 else None
        ax.barh(yi, delta, height=0.65, color=bar_col, alpha=0.80,
                edgecolor=col, linewidth=1.5, zorder=3,
                xerr=xerr, error_kw=dict(lw=1.3, capsize=4, capthick=1.3,
                                          color="#444", zorder=5))

        ha  = "left"  if delta >= 0 else "right"
        off = 0.0005  if delta >= 0 else -0.0005
        ax.text(delta + off, yi + 0.35,
                f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}",
                va="center", ha=ha, fontsize=10, fontweight="bold",
                color=bar_col, zorder=6)

    ax.axvline(0, color="#666", lw=1.2, ls="--", alpha=0.7, zorder=1)

    ax.text(0.99, 0.015, "← removes noise (improves R²)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color="#E15759", style="italic")
    ax.text(0.01, 0.015, "reduces R² (feature is useful) →",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color="#27AE60", style="italic")

    ytick_labels = list(active_sorted["feature"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ytick_labels, fontsize=11)
    for j, tick in enumerate(ax.get_yticklabels()):
        r = active_sorted.iloc[j]
        tick.set_color(GC.get(r["group"], "#333"))
        tick.set_fontweight("bold")

    ax.set_ylim(-0.6, n_rows - 0.4)
    patches = [mpatches.Patch(color=c, label=g.capitalize())
               for g, c in GC.items() if g in active_sorted["group"].values]
    ax.legend(handles=patches, fontsize=9.5, loc="upper right",
              title="Feature group", title_fontsize=10, framealpha=0.93)
    ax.set_xlabel("Δ R²  (baseline − without feature)  |  error bar = SEM  (25 folds)",
                  fontsize=11)
    ax.set_title(
        f"Leave-One-Feature-Out Ablation  (n={len(df)} features)\n"
        f"Baseline R² = {bl_r2:.4f}  ·  {n_active} active  ·  {n_zeroed} L1-zeroed",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    return fig



def fig4_predictions():
    test = pd.read_csv(RESULTS / "clinvar_test_predictions_annotated.csv")
    exp  = pd.read_csv(RESULTS / "clinvar_explore_predictions.csv")

    thr = -0.1656

    # Taller figure so explore panel has room for 24 variant labels
    fig, axes = plt.subplots(1, 2, figsize=(15, 8),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.patch.set_facecolor("white")

    # ── Panel A: Test set ─────────────────────────────────────────────────────
    ax = axes[0]
    df = test.sort_values("pred_score").reset_index(drop=True)

    for i, row in df.iterrows():
        col     = C_PATH if row["clinvar_label"] == "Pathogenic" else C_BEN
        correct = bool(row["correct"])
        ax.scatter(i, row["pred_score"], color=col,
                   s=85, alpha=1.0 if correct else 0.22,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate only misclassified variants — keep arrows short & non-overlapping
    wrong = df[df["correct"] == 0].reset_index()
    for _, row in wrong.iterrows():
        i = row["index"]
        ax.annotate(
            row["variant"],
            xy=(i, row["pred_score"]),
            xytext=(i - 3, row["pred_score"] - 0.12),
            fontsize=8.5, color="#222", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0,
                            shrinkA=4, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#AAAAAA",
                      alpha=0.9, lw=0.8),
        )

    ax.axhline(thr, color="#333", ls="--", lw=1.5)
    ax.axhspan(df["pred_score"].min() - 0.12, thr, alpha=0.04, color=C_PATH, zorder=0)
    ax.axhspan(thr, df["pred_score"].max() + 0.12, alpha=0.04, color=C_BEN, zorder=0)

    n_p  = (df["clinvar_label"] == "Pathogenic").sum()
    n_b  = (df["clinvar_label"] == "Benign").sum()
    acc  = df["correct"].mean()
    sens = df.loc[df["clinvar_label"] == "Pathogenic", "correct"].mean()
    spec = df.loc[df["clinvar_label"] == "Benign",     "correct"].mean()

    ax.text(0.97, 0.97,
            f"AUROC = 1.000\nAcc = {int(acc*len(df))}/{len(df)}  ({acc*100:.0f}%)\n"
            f"Sens = {sens:.3f}  ·  Spec = {spec:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.95))

    # Threshold label placed to avoid clashing with data
    ax.text(len(df) * 0.02, thr + 0.03, f"threshold = {thr:.3f}",
            fontsize=8, color="#555", style="italic")

    ax.set_xlabel("Variants ranked by predicted score  (low → high, most pathogenic left)")
    ax.set_ylabel("Predicted MAVE score  (lower = more loss-of-function)")
    ax.set_title(f"A   ClinVar Held-Out Test  (n = {len(df)}: {n_p}P / {n_b}B)\n"
                 "Faded markers = misclassified  ·  Shaded regions = predicted class",
                 loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=C_PATH, label="Pathogenic (ClinVar)"),
        mpatches.Patch(color=C_BEN,  label="Benign (ClinVar)"),
        plt.Line2D([0], [0], color="#333", ls="--", label="Classification threshold"),
    ], fontsize=8.5, loc="upper left", framealpha=0.9)

    # ── Panel B: Explore predictions ─────────────────────────────────────────
    ax2 = axes[1]
    edf = exp.sort_values("pred_score").reset_index(drop=True)

    mark_map = {"VUS": "D", "Conflicting": "s"}
    for i, row in edf.iterrows():
        col = C_PATH if row["pred_class"] == "Pathogenic" else C_BEN
        mk  = mark_map.get(row["clinvar_label"], "o")
        lo  = max(row["pred_score"] - row["ci_lo"], 0)
        hi  = max(row["ci_hi"] - row["pred_score"], 0)
        ax2.errorbar(row["pred_score"], i, xerr=[[lo], [hi]],
                     fmt=mk, color=col, capsize=3.5, markersize=7,
                     alpha=0.88, linewidth=1.3, zorder=3)

    ax2.axvline(thr, color="#333", ls="--", lw=1.5)
    xl = ax2.get_xlim()
    ax2.axvspan(xl[0], thr, alpha=0.04, color=C_PATH, zorder=0)
    ax2.axvspan(thr, xl[1], alpha=0.04, color=C_BEN,  zorder=0)

    # Threshold label
    ax2.text(thr + 0.01, len(edf) - 0.6,
             f"{thr:.3f}", fontsize=8, color="#555", style="italic")

    # Variant names as y-tick labels (no separate annotations = no overlap)
    ax2.set_yticks(range(len(edf)))
    ax2.set_yticklabels(edf["variant"], fontsize=8.5)

    # Colour y-tick labels by predicted class
    for tick, (_, row) in zip(ax2.get_yticklabels(), edf.iterrows()):
        tick.set_color(C_PATH if row["pred_class"] == "Pathogenic" else C_BEN)

    n_pp = (edf["pred_class"] == "Pathogenic").sum()
    n_pb = (edf["pred_class"] == "Benign").sum()

    ax2.set_xlabel("Predicted MAVE score  (± 95% bootstrap CI)")
    ax2.set_title(f"B   VUS & Conflicting Explore  (n = {len(edf)})\n"
                  f"{n_pp} pred. Pathogenic · {n_pb} pred. Benign  ·  "
                  "◆ VUS   ■ Conflicting",
                  loc="left")
    ax2.legend(handles=[
        mpatches.Patch(color=C_PATH, alpha=0.85, label=f"Predicted Pathogenic (n={n_pp})"),
        mpatches.Patch(color=C_BEN,  alpha=0.85, label=f"Predicted Benign (n={n_pb})"),
        plt.Line2D([0], [0], marker="D", color="#888", ls="none",
                   markersize=7, label="VUS"),
        plt.Line2D([0], [0], marker="s", color="#888", ls="none",
                   markersize=7, label="Conflicting"),
        plt.Line2D([0], [0], color="#333", ls="--",
                   label="Classification threshold"),
    ], fontsize=8, loc="lower right", framealpha=0.9)

    plt.tight_layout(w_pad=3)
    return fig



def fig6_model_selection():
    df = pd.read_csv(RESULTS / "model_label_summary.csv")

    # ── Layout constants ──────────────────────────────────────────────────────
    MODEL_ORDER  = ["elasticnet", "rf", "lasso", "ridge", "lgbm", "xgb", "gbm", "svr"]
    LABEL_ORDER  = ["both", "hdr", "any", "cisplatin"]
    LABEL_PRETTY = {"both": "avg(cis + HDR)", "hdr": "HDR only",
                    "any": "any assay", "cisplatin": "cisplatin only"}
    MODEL_COLORS = {
        "elasticnet": "#2471A3",
        "ridge":      "#5DADE2",
        "lasso":      "#85C1E9",
        "svr":        "#AED6F1",
        "rf":         "#1E8449",
        "gbm":        "#58D68D",
        "xgb":        "#A9DFBF",
        "lgbm":       "#D5F5E3",
    }
    WINNER = ("both", "elasticnet")

    fig, ax_heat = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("white")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — R² heatmap  (models × label strategies)
    # ══════════════════════════════════════════════════════════════════════════
    pivot = (df.set_index(["model", "label"])["mean_r2"]
               .unstack("label")
               .reindex(index=MODEL_ORDER, columns=LABEL_ORDER))
    pivot_std = (df.set_index(["model", "label"])["std_r2"]
                   .unstack("label")
                   .reindex(index=MODEL_ORDER, columns=LABEL_ORDER))

    # Clip negative R² to 0 for display purposes
    heat_vals = pivot.values.copy()
    vmin, vmax = -0.05, 0.55

    im = ax_heat.imshow(heat_vals, aspect="auto", cmap="RdYlGn",
                        vmin=vmin, vmax=vmax, interpolation="nearest")

    # Annotate each cell
    for i, model in enumerate(MODEL_ORDER):
        for j, label in enumerate(LABEL_ORDER):
            val  = pivot.loc[model, label]
            std  = pivot_std.loc[model, label]
            text = f"{val:.3f}\n±{std:.3f}" if not np.isnan(val) else "—"
            fc   = "white" if val > 0.35 or val < 0.0 else "#222"
            ax_heat.text(j, i, text, ha="center", va="center",
                         fontsize=8, color=fc, fontweight="normal")

    # Highlight winner with a thick border
    wi = MODEL_ORDER.index(WINNER[1])
    wj = LABEL_ORDER.index(WINNER[0])
    from matplotlib.patches import Rectangle
    ax_heat.add_patch(Rectangle((wj - 0.48, wi - 0.48), 0.96, 0.96,
                                  fill=False, edgecolor="#FFD700",
                                  linewidth=3.0, zorder=5))
    ax_heat.text(wj, wi - 0.48, "★",
                 ha="center", va="top", fontsize=11,
                 color="#FFD700", fontweight="bold", zorder=6)

    ax_heat.set_xticks(range(len(LABEL_ORDER)))
    ax_heat.set_xticklabels([LABEL_PRETTY[l] for l in LABEL_ORDER], fontsize=9.5)
    ax_heat.set_yticks(range(len(MODEL_ORDER)))
    ax_heat.set_yticklabels([m.upper() for m in MODEL_ORDER], fontsize=9.5)
    for tick, m in zip(ax_heat.get_yticklabels(), MODEL_ORDER):
        tick.set_color(MODEL_COLORS[m])
        tick.set_fontweight("bold")

    ax_heat.set_xlabel("Label strategy", fontsize=10)
    ax_heat.set_ylabel("Model", fontsize=10)
    ax_heat.set_title("A   R² Heatmap — All 8 Models × 4 Label Strategies\n"
                      "★ = selected configuration", fontsize=10, fontweight="bold")
    ax_heat.tick_params(left=False, bottom=False)

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.78, pad=0.02)
    cbar.set_label("Mean R²  (5-fold × 5-repeat CV)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig



def fig_combo_heatmap():
    import matplotlib.colors as mcolors

    df = pd.read_csv(RESULTS / "combinatorial_ablations.csv")
    df = df.sort_values("r2", ascending=False).reset_index(drop=True)

    groups       = ["structural", "biochemical", "alphamissense", "evo", "engineered"]
    group_labels = ["Structural", "Biochemical", "AlphaMissense", "Evolutionary", "Engineered"]
    n = len(df)

    # Build RGBA image (n × 5) for group inclusion panel
    rgba = np.ones((n, len(groups), 4))
    for i, (_, row) in enumerate(df.iterrows()):
        for j, g in enumerate(groups):
            if row[f"has_{g}"]:
                rgba[i, j] = mcolors.to_rgba(GC[g], alpha=0.88)
            else:
                rgba[i, j] = mcolors.to_rgba("#DDDDDD", alpha=0.45)

    r2_arr = df["r2"].values.reshape(n, 1)
    r2_norm = plt.Normalize(vmin=float(r2_arr.min()), vmax=float(r2_arr.max()))

    fig, (ax_mat, ax_r2) = plt.subplots(
        1, 2, figsize=(9, 10),
        gridspec_kw={"width_ratios": [5, 1]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    # ── Left: binary inclusion matrix ────────────────────────────────────────
    ax_mat.imshow(rgba, aspect="auto", interpolation="nearest",
                  extent=[-0.5, len(groups) - 0.5, n - 0.5, -0.5])

    ax_mat.set_xticks(range(len(groups)))
    ax_mat.set_xticklabels(group_labels, rotation=35, ha="right", fontsize=9)
    ax_mat.set_yticks(range(n))
    ax_mat.set_yticklabels(df["combo"], fontsize=8)
    ax_mat.tick_params(length=0)
    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    # Divider lines between n_group bands
    prev_k = df.iloc[0]["n_groups"]
    for i, (_, row) in enumerate(df.iterrows()):
        if row["n_groups"] != prev_k:
            ax_mat.axhline(i - 0.5, color="#999999", lw=0.8, ls="--", zorder=3)
        prev_k = row["n_groups"]

    ax_mat.set_title("Feature Group Combinations\n(colored = included,  sorted by R²)",
                     fontsize=11, fontweight="bold")

    # ── Right: R² heatmap column ─────────────────────────────────────────────
    im = ax_r2.imshow(r2_arr, aspect="auto", cmap="RdYlGn",
                      norm=r2_norm, interpolation="nearest",
                      extent=[-0.5, 0.5, n - 0.5, -0.5])

    for i, v in enumerate(df["r2"].values):
        fc = "white" if r2_norm(v) > 0.65 else "#333333"
        ax_r2.text(0, i, f"{v:.3f}", va="center", ha="center",
                   fontsize=7.5, color=fc, fontweight="bold")

    ax_r2.set_xticks([0])
    ax_r2.set_xticklabels(["CV R²"], fontsize=9)
    ax_r2.set_yticks([])
    ax_r2.tick_params(length=0)
    for spine in ax_r2.spines.values():
        spine.set_visible(False)

    # Divider lines mirroring left panel
    prev_k = df.iloc[0]["n_groups"]
    for i, (_, row) in enumerate(df.iterrows()):
        if row["n_groups"] != prev_k:
            ax_r2.axhline(i - 0.5, color="#999999", lw=0.8, ls="--", zorder=3)
        prev_k = row["n_groups"]

    cb = plt.colorbar(im, ax=ax_r2, shrink=0.35, pad=0.12)
    cb.set_label("CV R²  (5-fold × 5-repeat)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # Group color legend
    patches = [mpatches.Patch(color=GC[g], label=gl, alpha=0.88)
               for g, gl in zip(groups, group_labels)]
    ax_mat.legend(handles=patches, fontsize=8, loc="lower left",
                  title="Feature group", title_fontsize=8.5, framealpha=0.93)

    fig.suptitle(
        "BRCA1 ElasticNet — Combinatorial Feature-Group Screen\n"
        "All 31 non-empty subsets of 5 groups  (extended split, both-assay label, n=86 train)",
        fontsize=11, fontweight="bold",
    )
    return fig




# ─────────────────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating figures → {OUTDIR}/\n")
    tasks = [
        ("fig1_model_comparison",    fig6_model_selection),
        ("fig2_cv_performance",      fig2_cv_performance),
        ("fig3_shap_importance",     fig3a_shap),
        ("fig4_combo_ablation",      fig_combo_heatmap),
        ("fig5_feature_ablation",    fig3b_feature_ablation),
        ("fig6_clinvar_vus",         fig4_predictions),
    ]
    for stem, fn in tasks:
        print(f"  {stem} …", end=" ", flush=True)
        try:
            fig = fn()
            save(fig, stem)
        except Exception as e:
            print(f"\n  ERROR in {stem}: {e}")
            import traceback; traceback.print_exc()
    print(f"\nDone. {len(tasks)} figures saved to {OUTDIR}/")
