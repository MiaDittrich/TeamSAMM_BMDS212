#!/usr/bin/env python3
"""
Combinatorial feature-group ablation for the BRCA1 MAVE ElasticNet model.

Tests all 2^5 - 1 = 31 non-empty subsets of the 5 feature groups:
  structural, biochemical, alphamissense, evo, engineered

For each subset, runs 5×5 repeated K-Fold CV (fixed ElasticNet params) and
records R², Spearman ρ, Pearson r, RMSE, and AUROC.

Outputs:
  results_ext_both/combinatorial_ablations.csv   — raw results for all 31 combos
  results_ext_both/fig_combo_ablation_heatmap.png — compound heatmap figure
"""

import warnings
warnings.filterwarnings("ignore")

import re
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline

BASE   = Path(__file__).parent
OUTDIR = BASE / "results"
OUTDIR.mkdir(exist_ok=True)

AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

CLINVAR_ORIGINAL_20 = {
    "S1841R", "L1839S", "V1838E", "M1775R", "Y1703S",
    "W1718L", "G1706R", "G1738E", "S1715R", "I1760S",
    "D1733G", "I1766V", "T1773S", "V1736I", "K1793Q",
    "E1794G", "S1797C", "H1862L", "P1831S", "E1829T",
}

FEATURE_GROUPS = {
    "structural":    ["mutant_plddt", "ca_rmsd", "backbone_rmsd", "mutant_ca_displacement",
                      "shell_rmsd_5A", "shell_rmsd_8A", "shell_rmsd_12A",
                      "ramachandran_violation", "is_disordered_variant"],
    "biochemical":   ["pam250_score", "delta_hydrophobicity", "delta_size",
                      "delta_charge", "delta_aromaticity",
                      "is_charge_reversal", "is_size_increase",
                      "is_hydrophobic_to_polar", "is_polar_to_hydrophobic"],
    "alphamissense": ["am_pathogenicity", "am_class_enc"],
    "evo":           ["evoef2_ddg_Total", "ddg_evoef2", "esm2_llr"],
    "engineered":    ["am_pathogenicity_sq", "am_x_evo", "plddt_rmsd", "esm2_x_am"],
}

GROUP_COLORS = {
    "structural":    "#4E79A7",
    "biochemical":   "#F28E2B",
    "alphamissense": "#59A14F",
    "evo":           "#E15759",
    "engineered":    "#9467BD",
}

SPEARMAN_SCORER = __import__("sklearn.metrics", fromlist=["make_scorer"]).make_scorer(
    lambda y, yhat: float(stats.spearmanr(y, yhat)[0]),
    greater_is_better=True,
)


def parse_variant(v):
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def load_training_data():
    feats      = pd.read_csv(BASE / "data" / "brca1_final_feats.csv")
    mave       = pd.read_csv(BASE / "data" / "brca1_mave.csv")
    cv_test    = pd.read_csv(BASE / "data" / "clinvar_test.csv",    usecols=[0, 1])
    cv_explore = pd.read_csv(BASE / "data" / "clinvar_explore.csv", usecols=[0, 1])

    parsed = feats["variant"].apply(
        lambda v: pd.Series(parse_variant(v), index=["ref_aa", "_pos", "alt_aa"])
    )
    feats = pd.concat([feats, parsed], axis=1)

    mave_cols = ["uniprot_position", "ref_aa", "alt_aa",
                 "am_pathogenicity", "am_class",
                 "cisplatin_score", "hdr_activity_score"]
    merged = feats.merge(
        mave[mave_cols],
        left_on=["mutant_residue", "ref_aa", "alt_aa"],
        right_on=["uniprot_position", "ref_aa", "alt_aa"],
        how="left", validate="m:1",
    ).drop(columns=["_pos", "uniprot_position"])

    merged["am_class_enc"] = merged["am_class"].map(AM_CLASS_MAP)

    base_feats = []
    for grp, cols in FEATURE_GROUPS.items():
        if grp == "engineered":
            continue
        base_feats.extend([c for c in cols if c in merged.columns and c not in base_feats])
    base_feats = [f for f in base_feats if not merged[f].isna().all()]

    eng_feats = []
    if "am_pathogenicity" in base_feats:
        merged["am_pathogenicity_sq"] = merged["am_pathogenicity"] ** 2
        eng_feats.append("am_pathogenicity_sq")
    if "am_pathogenicity" in base_feats and "evoef2_ddg_Total" in base_feats:
        evo_abs = merged["evoef2_ddg_Total"].fillna(0).abs()
        merged["am_x_evo"] = (
            merged["am_pathogenicity"].fillna(0)
            * np.sign(merged["evoef2_ddg_Total"].fillna(0))
            * np.log1p(evo_abs)
        )
        eng_feats.append("am_x_evo")
    if "mutant_plddt" in base_feats and "ca_rmsd" in base_feats:
        merged["plddt_rmsd"] = (merged["mutant_plddt"] / 100.0) * merged["ca_rmsd"]
        eng_feats.append("plddt_rmsd")
    if "esm2_llr" in base_feats and "am_pathogenicity" in base_feats:
        merged["esm2_x_am"] = (
            merged["esm2_llr"].fillna(0) * merged["am_pathogenicity"].fillna(0)
        )
        eng_feats.append("esm2_x_am")

    all_feats = base_feats + eng_feats

    test_variants_all  = set(cv_test["variant"])
    explore_variants   = set(cv_explore["variant"])
    test_variants_held = test_variants_all - CLINVAR_ORIGINAL_20
    train_exclude      = test_variants_held | explore_variants

    train_all = merged[~merged["variant"].isin(train_exclude)].copy()
    mask  = train_all["cisplatin_score"].notna() & train_all["hdr_activity_score"].notna()
    label = train_all.loc[mask, ["cisplatin_score", "hdr_activity_score"]].mean(axis=1)
    train_all["label"] = label.reindex(train_all.index)
    train_df = train_all[train_all["label"].notna()].copy().reset_index(drop=True)

    print(f"Training set: n={len(train_df)}, total features available={len(all_feats)}")
    return train_df, all_feats


def fit_elasticnet(train_df, all_feats, seed=42):
    X = train_df[all_feats]
    y = train_df["label"]
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", GridSearchCV(
            ElasticNet(max_iter=10_000, random_state=seed),
            param_grid={"alpha": [1e-3, 1e-2, 1e-1], "l1_ratio": [0.2, 0.5, 0.8]},
            cv=inner, scoring=SPEARMAN_SCORER, n_jobs=-1, refit=True,
        )),
    ])
    pipe.fit(X, y)
    best = pipe.named_steps["est"].best_estimator_
    best_params = {"alpha": best.alpha, "l1_ratio": best.l1_ratio}
    print(f"Best ElasticNet params: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    return best_params


def ablation_cv(X, y, best_params, n_folds=5, n_repeats=5, seed=42):
    """5×5 repeated K-Fold CV with fixed ElasticNet hyperparameters."""
    rkf  = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", ElasticNet(max_iter=10_000, **best_params)),
    ])
    spearmans, pearsons, r2s, rmses = [], [], [], []
    oof_sum = np.zeros(len(y))
    oof_cnt = np.zeros(len(y))
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for tr, te in rkf.split(X):
        m = clone(pipe)
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        m.fit(X_tr, y_arr[tr])
        yhat = m.predict(X_te)
        oof_sum[te] += yhat
        oof_cnt[te] += 1
        spearmans.append(float(stats.spearmanr(y_arr[te], yhat)[0]))
        pearsons.append(float(stats.pearsonr(y_arr[te], yhat)[0]))
        r2s.append(float(r2_score(y_arr[te], yhat)))
        rmses.append(float(np.sqrt(mean_squared_error(y_arr[te], yhat))))

    oof  = np.where(oof_cnt > 0, oof_sum / oof_cnt, np.nan)
    thr  = float(np.median(y_arr))
    ybin = (y_arr <= thr).astype(int)
    auroc = float(roc_auc_score(ybin, -oof)) if ybin.min() != ybin.max() else np.nan

    return {
        "spearman":     float(np.nanmean(spearmans)),
        "spearman_std": float(np.nanstd(spearmans)),
        "pearson":      float(np.nanmean(pearsons)),
        "pearson_std":  float(np.nanstd(pearsons)),
        "r2":           float(np.nanmean(r2s)),
        "r2_std":       float(np.nanstd(r2s)),
        "rmse":         float(np.nanmean(rmses)),
        "auroc":        auroc,
    }


def run_combinatorial_ablations(train_df, all_feats, best_params,
                                n_folds=5, n_repeats=5, seed=42):
    groups = list(FEATURE_GROUPS.keys())
    feat_to_grp = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}

    # All features available in this dataset (post-engineering)
    avail = {g: [f for f in FEATURE_GROUPS[g] if f in all_feats] for g in groups}

    rows = []
    n_combos = 2 ** len(groups) - 1
    print(f"\nRunning {n_combos} group combinations (all non-empty subsets of {len(groups)} groups):")

    for k in range(1, len(groups) + 1):
        for combo in itertools.combinations(groups, k):
            feats_in_combo = []
            for g in combo:
                feats_in_combo.extend(avail[g])

            if not feats_in_combo:
                continue

            label = "+".join(g[:5] for g in combo)
            print(f"  [{label:<35}] n_feats={len(feats_in_combo):>2} ...", end=" ", flush=True)

            X_sub = train_df[feats_in_combo]
            res   = ablation_cv(X_sub, train_df["label"], best_params, n_folds, n_repeats, seed)

            row = {
                "combo":        label,
                "groups":       list(combo),
                "n_groups":     k,
                "n_feats":      len(feats_in_combo),
                "spearman":     res["spearman"],
                "spearman_std": res["spearman_std"],
                "pearson":      res["pearson"],
                "pearson_std":  res["pearson_std"],
                "r2":           res["r2"],
                "r2_std":       res["r2_std"],
                "rmse":         res["rmse"],
                "auroc":        res["auroc"],
            }
            # Binary inclusion columns for each group
            for g in groups:
                row[f"has_{g}"] = int(g in combo)

            rows.append(row)
            print(f"R²={res['r2']:+.4f} ± {res['r2_std']:.4f}  "
                  f"ρ={res['spearman']:+.4f}")

    df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    out_csv = OUTDIR / "combinatorial_ablations.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}  ({len(df)} combinations)")
    return df


def plot_combo_heatmap(df: pd.DataFrame):
    """
    Compound figure:
      Left panel  — binary inclusion matrix (group present = colored, absent = light gray)
      Right panel — horizontal bars showing CV R² ± 1 std
    Rows are sorted by R² (best at top). Banded by number of groups.
    """
    groups    = list(FEATURE_GROUPS.keys())
    group_abbr = {
        "structural":    "Structural",
        "biochemical":   "Biochemical",
        "alphamissense": "AlphaMissense",
        "evo":           "Evolutionary",
        "engineered":    "Engineered",
    }
    n = len(df)

    fig, (ax_mat, ax_bar) = plt.subplots(
        1, 2, figsize=(13, max(8, n * 0.32)),
        gridspec_kw={"width_ratios": [5, 3]},
    )
    fig.patch.set_facecolor("white")

    # ── Left: inclusion matrix ──────────────────────────────────────────────
    for row_idx, row in df.iterrows():
        # Alternating band shading by n_groups
        band_color = "#F7F7F7" if row["n_groups"] % 2 == 0 else "white"
        ax_mat.axhspan(row_idx - 0.5, row_idx + 0.5, color=band_color, zorder=0)

        for col_idx, g in enumerate(groups):
            if row[f"has_{g}"]:
                c = GROUP_COLORS[g]
                rect = plt.Rectangle(
                    (col_idx - 0.45, row_idx - 0.4), 0.9, 0.8,
                    color=c, alpha=0.85, zorder=2,
                )
                ax_mat.add_patch(rect)
            else:
                rect = plt.Rectangle(
                    (col_idx - 0.45, row_idx - 0.4), 0.9, 0.8,
                    color="#DDDDDD", alpha=0.4, zorder=1,
                )
                ax_mat.add_patch(rect)

    ax_mat.set_xlim(-0.5, len(groups) - 0.5)
    ax_mat.set_ylim(-0.5, n - 0.5)
    ax_mat.set_xticks(range(len(groups)))
    ax_mat.set_xticklabels([group_abbr[g] for g in groups],
                           rotation=35, ha="right", fontsize=9)
    ax_mat.set_yticks(range(n))

    # Row labels: combo name + n_feats
    row_labels = [
        f"{row['combo']}  ({row['n_feats']}f)"
        for _, row in df.iterrows()
    ]
    ax_mat.set_yticklabels(row_labels, fontsize=7.5)
    ax_mat.invert_yaxis()  # best at top
    ax_mat.set_title("Feature Group Combinations\n(colored = included)", fontsize=10, fontweight="bold")
    ax_mat.tick_params(axis="x", length=0)
    ax_mat.tick_params(axis="y", length=0)
    for spine in ax_mat.spines.values():
        spine.set_visible(False)

    # Divider lines between n_group bands
    prev_k = df.iloc[0]["n_groups"]
    for row_idx, row in df.iterrows():
        if row["n_groups"] != prev_k and row_idx > 0:
            ax_mat.axhline(row_idx - 0.5, color="#999999", lw=0.8, ls="--", zorder=3)
        prev_k = row["n_groups"]

    # ── Right: R² bar chart ─────────────────────────────────────────────────
    r2_vals   = df["r2"].values
    r2_std    = df["r2_std"].values
    n_groups_col = df["n_groups"].values

    # Color bars by n_groups (1→5 groups)
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
    bar_colors = [palette[k - 1] for k in n_groups_col]

    for row_idx in range(n):
        ax_bar.barh(row_idx, r2_vals[row_idx], color=bar_colors[row_idx],
                    alpha=0.80, height=0.65, zorder=2)
        # ±1 std error bar
        ax_bar.errorbar(r2_vals[row_idx], row_idx,
                        xerr=r2_std[row_idx],
                        fmt="none", color="#333333", capsize=2, linewidth=0.8, zorder=3)
        # Value annotation
        ax_bar.text(max(r2_vals[row_idx] + r2_std[row_idx] + 0.005, 0.01),
                    row_idx, f"{r2_vals[row_idx]:+.3f}",
                    va="center", fontsize=7, color="#222")

    # Band shading matches left panel
    for row_idx, row in df.iterrows():
        band_color = "#F7F7F7" if row["n_groups"] % 2 == 0 else "white"
        ax_bar.axhspan(row_idx - 0.5, row_idx + 0.5, color=band_color, zorder=0)

    # Baseline full-model vertical line (max R² = first row = all groups)
    full_r2 = df[df["n_groups"] == 5]["r2"].max() if (df["n_groups"] == 5).any() else r2_vals[0]
    ax_bar.axvline(full_r2, color="#333333", lw=1.2, ls=":", alpha=0.6,
                   label=f"All-group baseline R²={full_r2:.3f}")

    ax_bar.axvline(0, color="#888888", lw=0.7, ls="-", alpha=0.5)
    ax_bar.set_xlim(min(r2_vals.min() - 0.05, -0.05),
                    max(r2_vals.max() + 0.12, 0.75))
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("CV R²  (5-fold × 5-repeat, ± 1 std)", fontsize=9)
    ax_bar.set_title("R² by Combination", fontsize=10, fontweight="bold")
    ax_bar.set_yticks([])
    ax_bar.grid(axis="x", color="#EBEBEB", linewidth=0.5, zorder=0)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)

    # Divider lines (mirror left panel)
    prev_k = df.iloc[0]["n_groups"]
    for row_idx, row in df.iterrows():
        if row["n_groups"] != prev_k and row_idx > 0:
            ax_bar.axhline(row_idx - 0.5, color="#999999", lw=0.8, ls="--", zorder=3)
        prev_k = row["n_groups"]

    # Legend for n_groups color
    n_patches = [
        mpatches.Patch(color=palette[k - 1], alpha=0.8, label=f"{k} group{'s' if k > 1 else ''}")
        for k in range(1, 6)
    ]
    ax_bar.legend(handles=n_patches, title="# groups", fontsize=7.5,
                  title_fontsize=8, loc="lower right")

    fig.suptitle(
        "BRCA1 ElasticNet — Combinatorial Feature-Group Ablation\n"
        "All 31 non-empty subsets of 5 groups  (extended split, both-assay label, n=86 train)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_fig = OUTDIR / "fig_combo_ablation_heatmap.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_fig}")


if __name__ == "__main__":
    print("=" * 65)
    print("  BRCA1 Combinatorial Group Ablation (ElasticNet, ext/both)")
    print("=" * 65)

    train_df, all_feats = load_training_data()

    print("\nFitting ElasticNet to determine best hyperparameters …")
    best_params = fit_elasticnet(train_df, all_feats)

    df = run_combinatorial_ablations(train_df, all_feats, best_params)

    print("\nTop 10 combinations by R²:")
    print(df[["combo", "n_groups", "n_feats", "r2", "r2_std", "spearman", "auroc"]]
          .head(10).to_string(index=False))

    print("\nGenerating heatmap …")
    plot_combo_heatmap(df)

    print("\nDone.")
