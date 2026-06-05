#!/usr/bin/env python3
"""
BRCA1 MAVE Score Prediction Pipeline
=====================================
Predicts MAVE functional assay scores (cisplatin_score, hdr_activity_score)
from structural, biochemical, evolutionary, and alphaMissense variant features.

Usage
-----
python brca1_mave_pipeline.py                                  # all models, all label strategies
python brca1_mave_pipeline.py --model xgb                      # single model
python brca1_mave_pipeline.py --labels hdr --model rf          # single model + label
python brca1_mave_pipeline.py --model xgb --no-ablation        # skip ablation studies
python brca1_mave_pipeline.py --find-best                      # report best model+label combo
python brca1_mave_pipeline.py --rank-metric spearman           # rank models by Spearman (default)
python brca1_mave_pipeline.py --n-repeats 10                   # more stable CV (slower)
python brca1_mave_pipeline.py --no-tune                        # disable nested-CV tuning (faster)

Available models : ridge, lasso, elasticnet, rf, gbm, xgb, lgbm, svr, all
Available labels : both, any, cisplatin, hdr, all
Eval metrics     : Spearman ρ, Pearson r, R², RMSE, MAE, AUROC, AUPRC

Notes
-----
• Cross-validation is Repeated K-Fold (n_folds × n_repeats) to stabilise
  estimates on the small (n≈100) labelled set.
• ridge / elasticnet / rf are tuned by an inner CV grid (nested CV) so the
  reported scores stay honest. Use --no-tune to disable.
• AUROC/AUPRC treat the task as damaging-vs-functional by thresholding the
  score (default: per-label median; override with --class-threshold).

Output
------
All plots and CSVs are written to --outdir (default: brca1_results/).
"""

import os
import re
import sys
import warnings
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Ordinal encoding: benign < ambiguous < pathogenic
AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

# Feature group definitions – must match column names after preprocessing
FEATURE_GROUPS = {
    "structural": [
        "mutant_plddt",
        "ca_rmsd", "backbone_rmsd",
        "mutant_ca_displacement",
        "shell_rmsd_5A", "shell_rmsd_8A", "shell_rmsd_12A",
        "ramachandran_violation",
        "rsa", "ss_helix", "ss_strand", "ss_coil",
        "is_interface_residue",
        "is_disordered_variant",
    ],
    "biochemical": [
        "pam250_score",
        "delta_hydrophobicity", "delta_size",
        "delta_charge", "delta_aromaticity",
        "is_charge_reversal", "is_size_increase",
        "is_hydrophobic_to_polar", "is_polar_to_hydrophobic",
    ],
    "alphamissense": [
        "am_pathogenicity",
        "am_class_enc",          # am_class after ordinal encoding
    ],
    "evo": [
        "evoef2_ddg_Total",
        "ddg_evoef2",
    ],
}

# Palette for models
MODEL_COLORS = {
    "ridge":       "#4E79A7",
    "lasso":       "#F28E2B",
    "elasticnet":  "#E15759",
    "rf":          "#76B7B2",
    "gbm":         "#59A14F",
    "xgb":         "#EDC948",
    "lgbm":        "#B07AA1",
    "svr":         "#FF9DA7",
}

# Palette for feature groups
GROUP_COLORS = {
    "structural":    "#4E79A7",
    "biochemical":   "#F28E2B",
    "alphamissense": "#59A14F",
    "evo":           "#E15759",
}

# Palette for label strategies
LABEL_COLORS = {
    "both":      "#5778A4",
    "any":       "#E49444",
    "cisplatin": "#D1615D",
    "hdr":       "#85B6B2",
}

LABEL_DESCRIPTIONS = {
    "both":      "Average (both scores required)",
    "any":       "Average of available scores",
    "cisplatin": "Cisplatin score only",
    "hdr":       "HDR activity score only",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def parse_variant(v: str):
    """Parse 'P1579C' → ('P', 1579, 'C').  Returns (None,None,None) on failure."""
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def _safe_corr(corr_fn, a, b):
    """Correlation that returns NaN instead of erroring on a constant vector."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(corr_fn(a, b)[0])
    except Exception:
        return np.nan


def _scalar(metrics: dict, key: str) -> float:
    """Return a scalar for a metric whether it is stored per-fold or as a scalar."""
    v = metrics[key]
    if hasattr(v, "__len__"):
        return float(np.nanmean(v))
    return float(v)


# Metrics where a higher value is better (the rest are error metrics: lower is better)
HIGHER_IS_BETTER = {"r2", "pearson", "spearman", "auroc", "auprc"}


def feat_to_group(feature_names):
    """Map each feature name to its group string."""
    mapping = {}
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            mapping[f] = grp
    return {f: mapping.get(f, "other") for f in feature_names}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & MERGE
# ──────────────────────────────────────────────────────────────────────────────

def load_and_merge(feats_path: str, mave_path: str) -> pd.DataFrame:
    """
    Merge BRCA1_FEATS and BRCA1_MAVE on exact variant identity:
        (mutant_residue, wt_aa, mut_aa) == (uniprot_position, ref_aa, alt_aa)

    Raises AssertionError if the resulting row count ≠ len(FEATS).
    """
    banner("STEP 1 — Loading and merging datasets")

    feats = pd.read_csv(feats_path)
    mave  = pd.read_csv(mave_path)

    print(f"  BRCA1_FEATS : {len(feats):,} rows × {feats.shape[1]} columns")
    print(f"  BRCA1_MAVE  : {len(mave):,} rows × {mave.shape[1]} columns")
    n_expected = len(feats)

    # Parse wt_aa / mut_aa out of the variant string so we get an exact match
    parsed = feats["variant"].apply(
        lambda v: pd.Series(parse_variant(v), index=["_wt", "_pos", "_mut"])
    )
    feats = pd.concat([feats, parsed], axis=1)

    n_unparsed = feats["_wt"].isna().sum()
    if n_unparsed:
        print(f"  WARNING: {n_unparsed} variants could not be parsed "
              f"(kept with NaN MAVE columns)")

    # Columns to bring in from MAVE
    mave_keep = ["uniprot_position", "ref_aa", "alt_aa",
                 "am_pathogenicity", "am_class",
                 "cisplatin_score", "hdr_activity_score"]

    merged = feats.merge(
        mave[mave_keep],
        left_on=["mutant_residue", "_wt", "_mut"],
        right_on=["uniprot_position", "ref_aa", "alt_aa"],
        how="left",
        validate="m:1",       # each FEATS row matches at most one MAVE row
    )

    # Drop the redundant merge-key columns
    merged = merged.drop(columns=["_wt", "_pos", "_mut",
                                   "uniprot_position", "ref_aa", "alt_aa"])

    # ── Verification ──────────────────────────────────────────────────────────
    if len(merged) != n_expected:
        raise AssertionError(
            f"\nMerge row-count mismatch: got {len(merged)}, expected {n_expected}.\n"
            "One or more FEATS variants matched multiple MAVE rows.\n"
            "Inspect BRCA1_MAVE for duplicate (position, ref_aa, alt_aa) entries."
        )

    n_unmatched = merged["am_pathogenicity"].isna().sum()
    n_both = (merged["cisplatin_score"].notna() & merged["hdr_activity_score"].notna()).sum()

    print(f"\n  Merge result    : {len(merged):,} rows ✓")
    print(f"  Unmatched vars  : {n_unmatched}  (no corresponding MAVE entry)")
    print(f"  cisplatin avail : {merged['cisplatin_score'].notna().sum()}")
    print(f"  HDR avail       : {merged['hdr_activity_score'].notna().sum()}")
    print(f"  Both scores     : {n_both}")

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame):
    """
    • Ordinally encode am_class → am_class_enc  (benign=0, ambiguous=1, pathogenic=2)
    • Drop features that are entirely NaN across the dataset (no information).
    • Build and return the ordered list of usable feature columns.
    """
    df = df.copy()
    df["am_class_enc"] = df["am_class"].map(AM_CLASS_MAP)

    all_features = []
    for grp_feats in FEATURE_GROUPS.values():
        for f in grp_feats:
            if f in df.columns and f not in all_features:
                all_features.append(f)

    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"  WARNING: expected features absent from the data: {missing}")
        all_features = [f for f in all_features if f in df.columns]

    # Drop features that are entirely empty (carry zero information).
    all_nan = [f for f in all_features if df[f].isna().all()]
    if all_nan:
        print(f"  Dropping all-NaN features (no data anywhere): {all_nan}")
        all_features = [f for f in all_features if f not in all_nan]

    print(f"\n  Features used ({len(all_features)}):")
    for grp, gfeats in FEATURE_GROUPS.items():
        present = [f for f in gfeats if f in all_features]
        if present:
            print(f"    [{grp:>14}] {present}")

    return df, all_features


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — LABEL STRATEGIES
# ──────────────────────────────────────────────────────────────────────────────

def build_label_sets(df: pd.DataFrame) -> dict:
    """
    Returns a dict keyed by strategy name. Each value is a DataFrame
    that has a 'label' column and is restricted to rows where that label
    is non-null.
    """
    banner("STEP 2 — Building label sets")

    label_sets = {}

    # ── Strategy 1: raw average, both scores required ─────────────────────────
    mask = df["cisplatin_score"].notna() & df["hdr_activity_score"].notna()
    tmp = df[mask].copy()
    tmp["label"] = (tmp["cisplatin_score"] + tmp["hdr_activity_score"]) / 2
    label_sets["both"] = tmp
    print(f"  [both]      Variants with BOTH scores  : {len(tmp):>4}")

    # ── Strategy 2: NaN-aware average (use whatever score is available) ────────
    def nanmean_row(row):
        vals = [row["cisplatin_score"], row["hdr_activity_score"]]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.mean(vals)) if vals else np.nan

    tmp2 = df.copy()
    tmp2["label"] = tmp2.apply(nanmean_row, axis=1)
    tmp2 = tmp2[tmp2["label"].notna()]
    label_sets["any"] = tmp2
    print(f"  [any]       Variants with ANY score    : {len(tmp2):>4}")

    # ── Strategy 3: cisplatin only ────────────────────────────────────────────
    mask3 = df["cisplatin_score"].notna()
    tmp3 = df[mask3].copy()
    tmp3["label"] = tmp3["cisplatin_score"]
    label_sets["cisplatin"] = tmp3
    print(f"  [cisplatin] Variants with cisplatin    : {len(tmp3):>4}")

    # ── Strategy 4: HDR only ──────────────────────────────────────────────────
    mask4 = df["hdr_activity_score"].notna()
    tmp4 = df[mask4].copy()
    tmp4["label"] = tmp4["hdr_activity_score"]
    label_sets["hdr"] = tmp4
    print(f"  [hdr]       Variants with HDR          : {len(tmp4):>4}")

    return label_sets


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_models(seed: int = 42, tune: bool = True) -> dict:
    """
    Returns an ordered dict of named sklearn Pipelines.

    Linear models use Imputer → StandardScaler → Estimator.
    Tree / boosting models use Imputer → Estimator  (no scaling needed).

    When `tune=True`, ridge / elasticnet / rf wrap their estimator in a
    GridSearchCV with an INNER 3-fold CV. Combined with the outer
    cross-validation in `evaluate_cv`, this gives proper *nested* CV:
    hyper-parameters are selected only on each outer training fold, so the
    reported scores are not optimistic. The remaining models keep fixed
    defaults (the boosting models in particular should be tuned with care on
    n≈100; fixed light settings here avoid over-fitting the tiny folds).
    """
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)

    def linear_pipe(est):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scl", StandardScaler()),
            ("est", est),
        ])

    def tree_pipe(est):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("est", est),
        ])

    def tuned(est, grid):
        return GridSearchCV(est, grid, cv=inner, scoring="r2",
                            n_jobs=-1, refit=True)

    # ── Estimators (tuned or fixed) ───────────────────────────────────────────
    if tune:
        ridge_est = tuned(Ridge(),
                          {"alpha": [0.1, 1.0, 10.0, 100.0]})
        enet_est  = tuned(ElasticNet(max_iter=10_000),
                          {"alpha":    [1e-3, 1e-2, 1e-1],
                           "l1_ratio": [0.2, 0.5, 0.8]})
        rf_est    = tuned(RandomForestRegressor(
                              n_estimators=300, random_state=seed, n_jobs=-1),
                          {"max_features":     ["sqrt", 0.5],
                           "min_samples_leaf": [1, 2]})
    else:
        ridge_est = Ridge(alpha=1.0)
        enet_est  = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000)
        rf_est    = RandomForestRegressor(
                        n_estimators=300, max_features="sqrt",
                        min_samples_leaf=2, random_state=seed, n_jobs=-1)

    return {
        "ridge":      linear_pipe(ridge_est),
        "lasso":      linear_pipe(Lasso(alpha=0.01, max_iter=10_000)),
        "elasticnet": linear_pipe(enet_est),
        "svr":        linear_pipe(SVR(C=10.0, epsilon=0.05, kernel="rbf")),
        "rf":  tree_pipe(rf_est),
        "gbm": tree_pipe(GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=3, subsample=0.8,
                    random_state=seed)),
        "xgb": tree_pipe(xgb.XGBRegressor(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=3, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbosity=0, n_jobs=-1)),
        "lgbm": tree_pipe(lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=3, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbosity=-1, n_jobs=-1)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — CROSS-VALIDATED EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cv(model_pipe, X: pd.DataFrame, y: pd.Series,
                n_folds: int = 5, n_repeats: int = 5, seed: int = 42,
                clf_threshold: float = None) -> dict:
    """
    Repeated K-Fold CV on a regression task.

    With n_repeats > 1 the K-Fold split is repeated with different shuffles,
    which stabilises the (otherwise noisy) estimates on a small dataset. The
    per-fold metric arrays then have length n_folds * n_repeats, so their
    std reflects both fold-to-fold and repeat-to-repeat variability.

    Returns a dict with:
        r2, rmse, mae, pearson, spearman   → per-fold arrays, shape (n_folds*n_repeats,)
        auroc, auprc                       → scalars, computed on pooled OOF predictions
                                             after binarising y at `clf_threshold`
                                             (default = median; class 1 = "damaging",
                                             i.e. low score)
        y_pred_oof                         → OOF predictions averaged over repeats, shape (n,)
        clf_threshold                      → threshold actually used
    """
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    r2s, rmses, maes, pearsons, spearmans = [], [], [], [], []

    n = len(y)
    y_arr = y.values
    oof_sum = np.zeros(n)
    oof_cnt = np.zeros(n)

    for tr_idx, te_idx in rkf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

        m = clone(model_pipe)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)

        oof_sum[te_idx] += y_hat
        oof_cnt[te_idx] += 1

        r2s.append(r2_score(y_te, y_hat))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_hat)))
        maes.append(mean_absolute_error(y_te, y_hat))
        pearsons.append(_safe_corr(stats.pearsonr, y_te, y_hat))
        spearmans.append(_safe_corr(stats.spearmanr, y_te, y_hat))

    # Average OOF predictions across the repeats (each sample is held out n_repeats times)
    y_pred_oof = np.where(oof_cnt > 0, oof_sum / np.maximum(oof_cnt, 1), np.nan)

    # ── Classification view: can the model separate damaging vs functional? ────
    thr = float(np.median(y_arr)) if clf_threshold is None else float(clf_threshold)
    y_bin = (y_arr <= thr).astype(int)        # 1 = damaging (low functional score)
    if y_bin.min() != y_bin.max():            # need both classes present
        damage_score = -y_pred_oof            # higher → predicted more damaging
        auroc = float(roc_auc_score(y_bin, damage_score))
        auprc = float(average_precision_score(y_bin, damage_score))
    else:
        auroc, auprc = np.nan, np.nan

    return {
        "r2":          np.array(r2s),
        "rmse":        np.array(rmses),
        "mae":         np.array(maes),
        "pearson":     np.array(pearsons),
        "spearman":    np.array(spearmans),
        "auroc":       auroc,
        "auprc":       auprc,
        "y_pred_oof":  y_pred_oof,
        "clf_threshold": thr,
    }


def evaluate_all(models: dict, X: pd.DataFrame, y: pd.Series,
                 n_folds: int = 5, n_repeats: int = 5, seed: int = 42,
                 clf_threshold: float = None) -> dict:
    results = {}
    for name, pipe in models.items():
        res = evaluate_cv(pipe, X, y, n_folds, n_repeats, seed, clf_threshold)
        results[name] = res
        print(
            f"    {name:<12} "
            f"Spearman={np.nanmean(res['spearman']):+.3f}  "
            f"R²={np.nanmean(res['r2']):+.3f}±{np.nanstd(res['r2']):.3f}  "
            f"RMSE={res['rmse'].mean():.3f}  MAE={res['mae'].mean():.3f}  "
            f"AUROC={res['auroc']:.3f}"
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — ABLATION STUDIES
# ──────────────────────────────────────────────────────────────────────────────

def ablation_leave_one_feature_out(model_pipe, X: pd.DataFrame, y: pd.Series,
                                    n_folds: int = 5, n_repeats: int = 5,
                                    seed: int = 42) -> pd.DataFrame:
    """
    Leave-one-feature-out ablation.
    Trains the model once with all features (baseline), then once per feature
    with that feature dropped.  Returns a DataFrame sorted by ΔR² (descending),
    where ΔR² = baseline_R² − R²_without_feature.  Positive ΔR² means the
    feature was helpful.
    """
    print("      Computing baseline …", end=" ", flush=True)
    base = evaluate_cv(clone(model_pipe), X, y, n_folds, n_repeats, seed)
    base_r2 = np.nanmean(base["r2"])
    base_pearson = np.nanmean(base["pearson"])
    print(f"R²={base_r2:.3f}")

    rows = []
    for feat in X.columns:
        X_drop = X.drop(columns=[feat])
        res = evaluate_cv(clone(model_pipe), X_drop, y, n_folds, n_repeats, seed)
        rows.append({
            "feature":        feat,
            "group":          feat_to_group(X.columns.tolist())[feat],
            "r2_without":     np.nanmean(res["r2"]),
            "r2_std_without": np.nanstd(res["r2"]),
            "delta_r2":       base_r2 - np.nanmean(res["r2"]),
            "delta_pearson":  base_pearson - np.nanmean(res["pearson"]),
        })
        print(f"      − {feat:<30}  R²={np.nanmean(res['r2']):+.3f}  "
              f"ΔR²={base_r2 - np.nanmean(res['r2']):+.3f}")

    df = pd.DataFrame(rows).sort_values("delta_r2", ascending=False)
    df.insert(0, "baseline_r2", base_r2)
    return df


def ablation_leave_one_group_out(model_pipe, X: pd.DataFrame, y: pd.Series,
                                  n_folds: int = 5, n_repeats: int = 5,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Leave-one-feature-group-out ablation.
    Groups: structural, biochemical, alphamissense, evo.
    """
    print("      Computing baseline …", end=" ", flush=True)
    base = evaluate_cv(clone(model_pipe), X, y, n_folds, n_repeats, seed)
    base_r2 = np.nanmean(base["r2"])
    base_pearson = np.nanmean(base["pearson"])
    print(f"R²={base_r2:.3f}")

    rows = []
    for grp_name, grp_feats in FEATURE_GROUPS.items():
        to_remove = [f for f in grp_feats if f in X.columns]
        remaining  = [f for f in X.columns if f not in to_remove]
        if not remaining:
            print(f"      WARNING: removing '{grp_name}' leaves no features; skipped.")
            continue
        res = evaluate_cv(clone(model_pipe), X[remaining], y, n_folds, n_repeats, seed)
        rows.append({
            "group":           grp_name,
            "n_features":      len(to_remove),
            "features_removed": ", ".join(to_remove),
            "r2_without":      np.nanmean(res["r2"]),
            "r2_std_without":  np.nanstd(res["r2"]),
            "delta_r2":        base_r2 - np.nanmean(res["r2"]),
            "pearson_without": np.nanmean(res["pearson"]),
            "delta_pearson":   base_pearson - np.nanmean(res["pearson"]),
        })
        print(f"      − {grp_name:<15}  R²={np.nanmean(res['r2']):+.3f}  "
              f"ΔR²={base_r2 - np.nanmean(res['r2']):+.3f}")

    df = pd.DataFrame(rows).sort_values("delta_r2", ascending=False)
    df.insert(0, "baseline_r2", base_r2)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":     True,
    "grid.color":    "#EBEBEB",
    "grid.linewidth": 0.6,
    "figure.dpi":    120,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")


def plot_label_distributions(label_sets: dict, outdir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, df) in zip(axes, label_sets.items()):
        y = df["label"]
        ax.hist(y, bins=30, color=LABEL_COLORS[name], alpha=0.80,
                edgecolor="white", linewidth=0.4)
        ax.axvline(y.median(), color="#111", linestyle="--", linewidth=1.4,
                   label=f"median={y.median():.2f}")
        ax.axvline(y.mean(),   color="#555", linestyle=":",  linewidth=1.2,
                   label=f"mean={y.mean():.2f}")
        ax.set_title(f"[{name}]  n={len(y)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Label Distribution by Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "01_label_distributions.png")


def plot_score_correlation(df: pd.DataFrame, outdir: Path):
    """Scatter of cisplatin_score vs hdr_activity_score for rows with both."""
    mask = df["cisplatin_score"].notna() & df["hdr_activity_score"].notna()
    sub = df[mask]
    r, p = stats.pearsonr(sub["cisplatin_score"], sub["hdr_activity_score"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub["cisplatin_score"], sub["hdr_activity_score"],
               c=LABEL_COLORS["both"], alpha=0.7, s=35, edgecolors="white", linewidths=0.4)
    lo = min(sub["cisplatin_score"].min(), sub["hdr_activity_score"].min())
    hi = max(sub["cisplatin_score"].max(), sub["hdr_activity_score"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
    ax.set_xlabel("Cisplatin score", fontsize=11)
    ax.set_ylabel("HDR activity score", fontsize=11)
    ax.set_title(f"Score correlation (n={len(sub)})\nPearson r={r:.3f}  p={p:.2e}",
                 fontsize=11, fontweight="bold")
    _save(fig, outdir / "02_score_correlation.png")


def plot_model_comparison(results_all: dict, outdir: Path):
    """4 metrics × N label strategies → grid of bar charts."""
    label_names  = list(results_all.keys())
    model_names  = list(next(iter(results_all.values())).keys())
    metrics      = ["spearman", "r2", "rmse", "pearson"]
    metric_labels = ["Spearman ρ", "R²", "RMSE", "Pearson r"]
    n_l = len(label_names)
    n_m = len(metrics)

    fig, axes = plt.subplots(n_l, n_m, figsize=(5 * n_m, 4 * n_l), squeeze=False)

    for row, lname in enumerate(label_names):
        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row][col]
            means  = [np.nanmean(results_all[lname][m][metric]) for m in model_names]
            stds   = [np.nanstd(results_all[lname][m][metric])  for m in model_names]
            colors = [MODEL_COLORS.get(m, "#999") for m in model_names]

            ax.bar(model_names, means, yerr=stds, color=colors, capsize=4,
                   alpha=0.85, edgecolor="white", linewidth=0.5, error_kw={"linewidth": 1.2})
            ax.set_title(f"[{lname}] — {mlabel}", fontsize=10, fontweight="bold")
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
            if metric in ("r2", "spearman", "pearson"):
                ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)

    fig.suptitle("Model Comparison Across Label Strategies  (Repeated K-Fold CV, mean ± std)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "03_model_comparison.png")


def plot_cv_distributions(results_all: dict, outdir: Path):
    """Violin/box of per-fold R² for each model."""
    label_names = list(results_all.keys())
    model_names = list(next(iter(results_all.values())).keys())
    n_l = len(label_names)

    fig, axes = plt.subplots(1, n_l, figsize=(5 * n_l, 5), squeeze=False)
    for col, lname in enumerate(label_names):
        ax = axes[0][col]
        data   = [results_all[lname][m]["r2"] for m in model_names]
        colors = [MODEL_COLORS.get(m, "#999") for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "#111", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("R² (per fold)", fontsize=10)
        ax.set_title(f"[{lname}] fold R² distribution", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)

    fig.suptitle("Cross-Validation R² Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "04_cv_distributions.png")


def plot_predictions_scatter(results_all: dict, label_sets: dict,
                              best_model: str, outdir: Path):
    """OOF predicted vs actual for the best model across label strategies."""
    label_names = list(results_all.keys())
    n = len(label_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for col, lname in enumerate(label_names):
        ax = axes[0][col]
        y_true = label_sets[lname]["label"].values
        y_pred = results_all[lname][best_model]["y_pred_oof"]
        r2     = np.nanmean(results_all[lname][best_model]["r2"])
        pearson= np.nanmean(results_all[lname][best_model]["pearson"])
        spear  = np.nanmean(results_all[lname][best_model]["spearman"])
        rmse   = results_all[lname][best_model]["rmse"].mean()

        ax.scatter(y_true, y_pred, c=LABEL_COLORS[lname], alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.4)
        lo = min(y_true.min(), np.nanmin(y_pred)) - 0.05
        hi = max(y_true.max(), np.nanmax(y_pred)) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4, label="y=x")
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted (OOF, mean over repeats)", fontsize=11)
        ax.set_title(
            f"[{lname}]  n={len(y_true)}\n"
            f"Spearman={spear:.3f}  R²={r2:.3f}  Pearson={pearson:.3f}  RMSE={rmse:.3f}",
            fontsize=10, fontweight="bold",
        )

    fig.suptitle(f"OOF Predicted vs Actual — {best_model.upper()}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / f"05_predictions_{best_model}.png")


def plot_feature_importance(model_pipe, feature_names: list,
                             label_name: str, model_name: str, outdir: Path):
    """Fit on full data, then plot horizontal importance bars coloured by group."""
    est = model_pipe.named_steps["est"]
    # If the estimator was tuned, the pipeline step is a GridSearchCV; unwrap it.
    if isinstance(est, GridSearchCV):
        est = est.best_estimator_

    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_)
    else:
        return None

    g_map = feat_to_group(feature_names)
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": imp,
        "group":      [g_map[f] for f in feature_names],
    }).sort_values("importance", ascending=True)

    bar_colors = [GROUP_COLORS.get(g, "#999") for g in imp_df["group"]]
    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.38)))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance\n{model_name.upper()}  |  label: {label_name}",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, outdir / f"06_feature_importance_{label_name}_{model_name}.png")
    return imp_df


def plot_feature_ablation(abl_df: pd.DataFrame, label_name: str,
                           model_name: str, outdir: Path):
    df = abl_df.sort_values("delta_r2", ascending=True)
    bar_colors = [GROUP_COLORS.get(g, "#999") for g in df["group"]]
    shade = ["#E15759" if d < 0 else "#59A14F" for d in df["delta_r2"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.38)))

    # ΔR²
    ax = axes[0]
    ax.barh(df["feature"], df["delta_r2"], color=shade, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.9)
    ax.set_xlabel("ΔR²  (positive = feature contributes)", fontsize=11)
    ax.set_title(
        f"Feature Ablation (LOO)\n{model_name.upper()}  |  [{label_name}]  "
        f"baseline R²={abl_df['baseline_r2'].iloc[0]:.3f}",
        fontsize=11, fontweight="bold",
    )

    # R² without that feature (coloured by group)
    ax2 = axes[1]
    ax2.barh(df["feature"], df["r2_without"],
             xerr=df["r2_std_without"], color=bar_colors,
             alpha=0.85, edgecolor="white", capsize=3)
    ax2.axvline(abl_df["baseline_r2"].iloc[0], color="#333",
                linestyle="--", linewidth=1.2, label=f"baseline R²={abl_df['baseline_r2'].iloc[0]:.3f}")
    ax2.set_xlabel("R² without feature  (mean ± std)", fontsize=11)
    ax2.set_title("R² When Feature Is Removed", fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax2.legend(handles=legend_patches + [plt.Line2D([0], [0], linestyle="--",
                color="#333", label=f"baseline R²={abl_df['baseline_r2'].iloc[0]:.3f}")],
               loc="lower right", fontsize=8)

    plt.tight_layout()
    _save(fig, outdir / f"07_ablation_features_{label_name}_{model_name}.png")


def plot_group_ablation(abl_df: pd.DataFrame, label_name: str,
                         model_name: str, outdir: Path):
    df = abl_df.sort_values("delta_r2", ascending=False)
    colors = [GROUP_COLORS.get(g, "#999") for g in df["group"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(
        axes, ["delta_r2", "delta_pearson"], ["ΔR²", "ΔPearson r"]
    ):
        bars = ax.bar(df["group"], df[metric], color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Feature Group", fontsize=11)
        ax.set_ylabel(ylabel + "  (positive = group helps)", fontsize=11)
        ax.set_title(
            f"Group Ablation  |  {ylabel}\n"
            f"{model_name.upper()}  |  [{label_name}]",
            fontsize=11, fontweight="bold",
        )
        for bar, val in zip(bars, df[metric]):
            ypos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.006
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Feature Group Ablation  |  baseline R²={abl_df['baseline_r2'].iloc[0]:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, outdir / f"08_ablation_groups_{label_name}_{model_name}.png")


def plot_best_model_heatmap(summary_df: pd.DataFrame, outdir: Path):
    panels = [
        ("mean_spearman", "Mean Spearman ρ", "RdYlGn", 0),
        ("mean_r2",       "Mean R²",         "RdYlGn", 0),
        ("mean_pearson",  "Mean Pearson r",  "RdYlGn", 0),
        ("mean_auroc",    "Mean AUROC",      "RdYlGn", 0.5),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5))

    for ax, (col, title, cmap, center) in zip(axes, panels):
        data = summary_df.pivot(index="model", columns="label", values=col)
        sns.heatmap(data, annot=True, fmt=".3f", cmap=cmap, center=center,
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(title + "  (Repeated K-Fold CV)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Label Strategy", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)

    fig.suptitle("Model × Label Strategy Performance Overview",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "09_best_model_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — BEST MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def find_best_combo(results_all: dict, rank_metric: str = "spearman"):
    """Return (best_label, best_model) and a summary DataFrame ranked by rank_metric.

    rank_metric is one of: spearman, pearson, r2, rmse, mae, auroc, auprc.
    Higher-is-better metrics sort descending; error metrics sort ascending.
    """
    rows = []
    for lname, model_results in results_all.items():
        for mname, metrics in model_results.items():
            rows.append({
                "label":         lname,
                "model":         mname,
                "mean_spearman": np.nanmean(metrics["spearman"]),
                "std_spearman":  np.nanstd(metrics["spearman"]),
                "mean_r2":       np.nanmean(metrics["r2"]),
                "std_r2":        np.nanstd(metrics["r2"]),
                "mean_pearson":  np.nanmean(metrics["pearson"]),
                "mean_rmse":     metrics["rmse"].mean(),
                "mean_mae":      metrics["mae"].mean(),
                "mean_auroc":    metrics["auroc"],
                "mean_auprc":    metrics["auprc"],
            })
    summary_df = pd.DataFrame(rows)
    sort_col = f"mean_{rank_metric}"
    ascending = rank_metric not in HIGHER_IS_BETTER
    summary_df = summary_df.sort_values(sort_col, ascending=ascending,
                                        na_position="last")
    best = summary_df.iloc[0]
    return (best["label"], best["model"]), summary_df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _here = Path(__file__).parent
    p.add_argument("--feats",   default=str(_here / "data" / "brca1_final_feats.csv"),
                   help="Path to brca1_final_feats.csv")
    p.add_argument("--mave",    default=str(_here / "data" / "brca1_mave.csv"),
                   help="Path to brca1_mave.csv")
    p.add_argument("--outdir",  default=str(_here / "results"),
                   help="Output directory (created if absent)")
    p.add_argument("--model",   default="all",
                   choices=["ridge","lasso","elasticnet","svr",
                            "rf","gbm","xgb","lgbm","all"],
                   help="Model(s) to evaluate  [default: all]")
    p.add_argument("--labels",  default="all",
                   choices=["both","any","cisplatin","hdr","all"],
                   help="Label strategy  [default: all]")
    p.add_argument("--n-folds", type=int, default=5, dest="n_folds",
                   help="Number of CV folds  [default: 5]")
    p.add_argument("--n-repeats", type=int, default=5, dest="n_repeats",
                   help="Number of times the K-fold split is repeated  [default: 5]")
    p.add_argument("--rank-metric", default="spearman", dest="rank_metric",
                   choices=["spearman", "pearson", "r2", "rmse", "mae", "auroc", "auprc"],
                   help="Metric used to pick the best model/label  [default: spearman]")
    p.add_argument("--class-threshold", type=float, default=None, dest="class_threshold",
                   help="Score threshold for the AUROC/AUPRC binarisation "
                        "(<= threshold = damaging). Default: per-label median.")
    p.add_argument("--tune", dest="tune", action="store_true", default=True,
                   help="Nested-CV tuning for ridge/elasticnet/rf  [default: on]")
    p.add_argument("--no-tune", dest="tune", action="store_false",
                   help="Disable tuning (faster; uses fixed defaults)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--ablation",    dest="ablation", action="store_true",  default=True)
    p.add_argument("--no-ablation", dest="ablation", action="store_false",
                   help="Skip ablation studies")
    p.add_argument("--find-best",   dest="find_best", action="store_true", default=True,
                   help="Report best model+label combination  [default: True]")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 62}")
    print(f"  BRCA1 MAVE Score Prediction Pipeline")
    print(f"{'═' * 62}")
    print(f"  feats   : {args.feats}")
    print(f"  mave    : {args.mave}")
    print(f"  outdir  : {outdir}")
    print(f"  model   : {args.model}")
    print(f"  labels  : {args.labels}")
    print(f"  n_folds : {args.n_folds}")
    print(f"  n_repeats: {args.n_repeats}")
    print(f"  tune    : {args.tune}  (ridge/elasticnet/rf via nested CV)")
    print(f"  rank by : {args.rank_metric}")
    print(f"  ablation: {args.ablation}")

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    merged = load_and_merge(args.feats, args.mave)
    merged.to_csv(outdir / "merged_data.csv", index=False)
    print(f"\n  Saved merged data → {outdir / 'merged_data.csv'}")

    # ── 2. Preprocess features ────────────────────────────────────────────────
    banner("STEP 2 — Feature preprocessing")
    merged, all_features = preprocess_features(merged)

    # ── 3. Build label sets ───────────────────────────────────────────────────
    label_sets = build_label_sets(merged)
    plot_label_distributions(label_sets, outdir)
    plot_score_correlation(merged, outdir)

    # Filter to user-requested labels
    if args.labels != "all":
        label_sets = {args.labels: label_sets[args.labels]}

    # ── 4. Models ─────────────────────────────────────────────────────────────
    all_models = get_models(seed=args.seed, tune=args.tune)
    if args.model != "all":
        all_models = {args.model: all_models[args.model]}

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    banner("STEP 3 — Cross-validated model evaluation")
    results_all = {}
    for lname, ldf in label_sets.items():
        print(f"\n  Label strategy: [{lname}]  n={len(ldf)}")
        X = ldf[all_features]
        y = ldf["label"]
        results_all[lname] = evaluate_all(
            all_models, X, y, args.n_folds, args.n_repeats, args.seed,
            clf_threshold=args.class_threshold,
        )

    # ── 6. Visualisations ─────────────────────────────────────────────────────
    banner("STEP 4 — Visualisations")
    plot_model_comparison(results_all, outdir)
    plot_cv_distributions(results_all, outdir)

    # Overall best model by the chosen ranking metric (averaged across label strategies)
    rm = args.rank_metric
    higher_better = rm in HIGHER_IS_BETTER
    model_avg = {
        m: np.nanmean([_scalar(results_all[l][m], rm) for l in results_all])
        for m in all_models
    }
    best_model_name = (max if higher_better else min)(model_avg, key=model_avg.get)
    print(f"\n  Best model (avg {rm} across labels): "
          f"{best_model_name.upper()} = {model_avg[best_model_name]:.3f}")

    plot_predictions_scatter(results_all, label_sets, best_model_name, outdir)

    # Feature importance — fit on full data per label strategy
    for lname, ldf in label_sets.items():
        X = ldf[all_features]
        y = ldf["label"]
        imp_pipe = clone(all_models[best_model_name])
        imp_pipe.fit(X, y)
        imp_df = plot_feature_importance(imp_pipe, all_features, lname, best_model_name, outdir)
        if imp_df is not None:
            imp_df.to_csv(outdir / f"feature_importance_{lname}_{best_model_name}.csv", index=False)

    # ── 7. Ablation ───────────────────────────────────────────────────────────
    if args.ablation:
        banner("STEP 5 — Ablation studies")

        # Choose the label strategy with the best ranking-metric for the best model
        best_label_name = (max if higher_better else min)(
            results_all,
            key=lambda l: _scalar(results_all[l][best_model_name], rm),
        )
        best_ldf = label_sets[best_label_name]
        X_best = best_ldf[all_features]
        y_best = best_ldf["label"]

        print(f"\n  Ablation target : [{best_label_name}]  n={len(best_ldf)}")
        print(f"  Ablation model  : {best_model_name.upper()}")

        # 7a. Feature-level LOO ablation
        print("\n  7a. Leave-one-feature-out ablation")
        abl_feat = ablation_leave_one_feature_out(
            all_models[best_model_name], X_best, y_best,
            args.n_folds, args.n_repeats, args.seed
        )
        abl_feat.to_csv(
            outdir / f"ablation_features_{best_label_name}_{best_model_name}.csv",
            index=False,
        )
        plot_feature_ablation(abl_feat, best_label_name, best_model_name, outdir)

        print("\n  Top features by ΔR² (most helpful → least):")
        print(
            abl_feat[["feature", "group", "delta_r2", "r2_without"]]
            .head(10)
            .to_string(index=False)
        )

        # 7b. Group-level ablation
        print("\n  7b. Leave-one-group-out ablation")
        abl_grp = ablation_leave_one_group_out(
            all_models[best_model_name], X_best, y_best,
            args.n_folds, args.n_repeats, args.seed
        )
        abl_grp.to_csv(
            outdir / f"ablation_groups_{best_label_name}_{best_model_name}.csv",
            index=False,
        )
        plot_group_ablation(abl_grp, best_label_name, best_model_name, outdir)

        print("\n  Group ablation results:")
        print(
            abl_grp[["group", "n_features", "delta_r2", "delta_pearson", "r2_without"]]
            .to_string(index=False)
        )

    # ── 8. Best combo summary ─────────────────────────────────────────────────
    if args.find_best:
        banner("STEP 6 — Best model × label selection")
        (best_label, best_model), summary_df = find_best_combo(results_all, rm)
        summary_df.to_csv(outdir / "model_label_summary.csv", index=False)
        plot_best_model_heatmap(summary_df, outdir)

        top = summary_df.iloc[0]
        print(f"\n  ★ BEST COMBINATION  (ranked by {rm})")
        print(f"    Model   : {best_model.upper()}")
        print(f"    Label   : {best_label}  ({LABEL_DESCRIPTIONS[best_label]})")
        print(f"    Spearman: {top['mean_spearman']:.3f} ± {top['std_spearman']:.3f}")
        print(f"    R²      : {top['mean_r2']:.3f} ± {top['std_r2']:.3f}")
        print(f"    Pearson : {top['mean_pearson']:.3f}")
        print(f"    RMSE    : {top['mean_rmse']:.3f}")
        print(f"    AUROC   : {top['mean_auroc']:.3f}   AUPRC: {top['mean_auprc']:.3f}")
        print(f"\n  Full ranking (top 10):")
        print(
            summary_df.head(10)[
                ["label", "model", "mean_spearman", "mean_r2",
                 "mean_pearson", "mean_rmse", "mean_auroc"]
            ].to_string(index=False)
        )

    print(f"\n{'═' * 62}")
    print(f"  Pipeline complete.  Results in: {outdir}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
