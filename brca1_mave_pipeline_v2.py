#!/usr/bin/env python3
"""
BRCA1 MAVE Score Prediction Pipeline v2 — Ablation-Pruned Features
====================================================================
Predicts MAVE functional assay scores (cisplatin_score, hdr_activity_score)
from structural, biochemical, evolutionary, and AlphaMissense features.

Changes from aim3/brca1_mave_research_pipeline_v2.py
------------------------------------------------------
Feature pruning based on leave-one-out ablation results in aim3/results/:

  Features REMOVED (negative delta_sp — removing them improves Spearman):
    structural  : ca_rmsd, shell_rmsd_5A/8A/12A, ramachandran_violation,
                  is_disordered_variant (all ≤ −0.001 delta_sp)
    biochemical : delta_hydrophobicity (−0.037), is_charge_reversal (−0.028),
                  pam250_score (−0.027), is_polar_to_hydrophobic (−0.027),
                  delta_charge (−0.024), is_hydrophobic_to_polar (−0.019)
    evo         : entire group removed (evoef2_ddg_Total −0.011, ddg_evoef2 −0.015)
    engineered  : am_x_evo (−0.021) auto-excluded (evoef2_ddg_Total removed),
                  plddt_rmsd (−0.004) auto-excluded (ca_rmsd removed)

  Features KEPT (positive delta_sp — their removal hurts Spearman):
    am_pathogenicity (+0.057), am_pathogenicity_sq (+0.049),
    mutant_ca_displacement (+0.020), am_class_enc (+0.011),
    delta_aromaticity (+0.009), mutant_plddt (+0.005),
    delta_size (+0.004), backbone_rmsd (+0.002), is_size_increase (+0.001)

All other pipeline logic (KNN imputation, ClinVar hold-out, bootstrap CIs,
stacking ensemble) is identical to aim3/brca1_mave_research_pipeline_v2.py.

Usage
-----
python brca1_mave_pipeline_v2.py
python brca1_mave_pipeline_v2.py --model rf --labels both
python brca1_mave_pipeline_v2.py --no-ablation  # skip ablation (faster)
python brca1_mave_pipeline_v2.py --no-baseline  # skip v1 baseline run
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
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CLINVAR_PATHOGENIC = [
    "S1841R", "L1839S", "V1838E", "M1775R", "Y1703S",
    "W1718L", "G1706R", "G1738E", "S1715R", "I1760S",
]
CLINVAR_BENIGN = [
    "D1733G", "I1766V", "T1773S", "V1736I", "K1793Q",
    "E1794G", "S1797C", "H1862L", "P1831S", "E1829T",
]
CLINVAR_ALL = CLINVAR_PATHOGENIC + CLINVAR_BENIGN

AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

# Ablation-pruned feature groups (9 base + 1 engineered = 10 features).
# Removed features consistently showed negative delta_sp in aim3 LOO ablation.
FEATURE_GROUPS = {
    "structural": [
        "mutant_plddt",
        "backbone_rmsd",
        "mutant_ca_displacement",
    ],
    "biochemical": [
        "delta_size",
        "delta_aromaticity",
        "is_size_increase",
    ],
    "alphamissense": [
        "am_pathogenicity",
        "am_class_enc",
    ],
    "engineered": [
        "am_pathogenicity_sq",
        # am_x_evo and plddt_rmsd auto-excluded: their dependencies
        # (evoef2_ddg_Total, ca_rmsd) are not in base features.
    ],
}

MODEL_COLORS = {
    "ridge":      "#4E79A7",
    "lasso":      "#F28E2B",
    "elasticnet": "#E15759",
    "rf":         "#76B7B2",
    "gbm":        "#59A14F",
    "xgb":        "#EDC948",
    "lgbm":       "#B07AA1",
    "svr":        "#FF9DA7",
    "stack":      "#9467BD",
}
GROUP_COLORS = {
    "structural":    "#4E79A7",
    "biochemical":   "#F28E2B",
    "alphamissense": "#59A14F",
    "engineered":    "#9467BD",
}
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
HIGHER_IS_BETTER = {"r2", "pearson", "spearman", "auroc", "auprc"}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def banner(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def parse_variant(v: str):
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def _safe_corr(corr_fn, a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(corr_fn(a, b)[0])
    except Exception:
        return np.nan


def _scalar(metrics: dict, key: str) -> float:
    v = metrics[key]
    if hasattr(v, "__len__"):
        return float(np.nanmean(v))
    return float(v)


def feat_to_group(feature_names):
    mapping = {}
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            mapping[f] = grp
    return {f: mapping.get(f, "other") for f in feature_names}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & MERGE
# ──────────────────────────────────────────────────────────────────────────────

def load_and_merge(feats_path: str, mave_path: str) -> pd.DataFrame:
    banner("STEP 1 — Loading and merging datasets")
    feats = pd.read_csv(feats_path)
    mave  = pd.read_csv(mave_path)
    print(f"  BRCA1_FEATS : {len(feats):,} rows × {feats.shape[1]} columns")
    print(f"  BRCA1_MAVE  : {len(mave):,} rows × {mave.shape[1]} columns")
    n_expected = len(feats)

    parsed = feats["variant"].apply(
        lambda v: pd.Series(parse_variant(v), index=["_wt", "_pos", "_mut"])
    )
    feats = pd.concat([feats, parsed], axis=1)
    n_unparsed = feats["_wt"].isna().sum()
    if n_unparsed:
        print(f"  WARNING: {n_unparsed} variants could not be parsed")

    mave_keep = ["uniprot_position", "ref_aa", "alt_aa",
                 "am_pathogenicity", "am_class", "cisplatin_score", "hdr_activity_score"]
    merged = feats.merge(
        mave[mave_keep],
        left_on=["mutant_residue", "_wt", "_mut"],
        right_on=["uniprot_position", "ref_aa", "alt_aa"],
        how="left", validate="m:1",
    )
    merged = merged.drop(columns=["_wt", "_pos", "_mut",
                                    "uniprot_position", "ref_aa", "alt_aa"])
    if len(merged) != n_expected:
        raise AssertionError(
            f"Merge row-count mismatch: got {len(merged)}, expected {n_expected}."
        )

    n_both = (merged["cisplatin_score"].notna() & merged["hdr_activity_score"].notna()).sum()
    print(f"  Merge result    : {len(merged):,} rows ✓")
    print(f"  Unmatched vars  : {merged['am_pathogenicity'].isna().sum()}")
    print(f"  Both scores     : {n_both}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESSING + FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_features(df: pd.DataFrame, engineer: bool = True):
    """
    Encodes am_class, drops all-NaN features, optionally adds engineered
    features.  With the pruned feature set only am_pathogenicity_sq is
    created (am_x_evo and plddt_rmsd are auto-excluded because their
    dependencies are absent from FEATURE_GROUPS).
    """
    df = df.copy()
    df["am_class_enc"] = df["am_class"].map(AM_CLASS_MAP)

    base_features = []
    for grp, gfeats in FEATURE_GROUPS.items():
        if grp == "engineered":
            continue
        for f in gfeats:
            if f in df.columns and f not in base_features:
                base_features.append(f)

    missing = [f for f in base_features if f not in df.columns]
    if missing:
        print(f"  WARNING: absent features: {missing}")
        base_features = [f for f in base_features if f in df.columns]

    all_nan = [f for f in base_features if df[f].isna().all()]
    if all_nan:
        print(f"  Dropping all-NaN features: {all_nan}")
        base_features = [f for f in base_features if f not in all_nan]

    eng_features: list = []
    if engineer:
        # am_pathogenicity² — non-linear AlphaMissense signal (delta_sp +0.049)
        if "am_pathogenicity" in base_features:
            df["am_pathogenicity_sq"] = df["am_pathogenicity"] ** 2
            eng_features.append("am_pathogenicity_sq")

        # am_x_evo intentionally skipped: evoef2_ddg_Total not in base features
        # plddt_rmsd intentionally skipped: ca_rmsd not in base features

    all_features = base_features + eng_features

    print(f"\n  Features used ({len(all_features)}):")
    for grp, gfeats in FEATURE_GROUPS.items():
        present = [f for f in gfeats if f in all_features]
        if present:
            print(f"    [{grp:>14}] {present}")

    return df, all_features


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — LABEL SETS
# ──────────────────────────────────────────────────────────────────────────────

def build_label_sets(df: pd.DataFrame) -> dict:
    banner("STEP 2 — Building label sets (full dataset)")

    label_sets = {}

    mask = df["cisplatin_score"].notna() & df["hdr_activity_score"].notna()
    tmp = df[mask].copy()
    tmp["label"] = (tmp["cisplatin_score"] + tmp["hdr_activity_score"]) / 2
    label_sets["both"] = tmp
    print(f"  [both]      Variants with BOTH scores  : {len(tmp):>4}")

    def _nanmean_row(row):
        vals = [v for v in [row["cisplatin_score"], row["hdr_activity_score"]] if pd.notna(v)]
        return float(np.mean(vals)) if vals else np.nan

    tmp2 = df.copy()
    tmp2["label"] = tmp2.apply(_nanmean_row, axis=1)
    tmp2 = tmp2[tmp2["label"].notna()]
    label_sets["any"] = tmp2
    print(f"  [any]       Variants with ANY score    : {len(tmp2):>4}")

    tmp3 = df[df["cisplatin_score"].notna()].copy()
    tmp3["label"] = tmp3["cisplatin_score"]
    label_sets["cisplatin"] = tmp3
    print(f"  [cisplatin] Variants with cisplatin    : {len(tmp3):>4}")

    tmp4 = df[df["hdr_activity_score"].notna()].copy()
    tmp4["label"] = tmp4["hdr_activity_score"]
    label_sets["hdr"] = tmp4
    print(f"  [hdr]       Variants with HDR          : {len(tmp4):>4}")

    return label_sets


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLINVAR HOLD-OUT SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def clinvar_split(label_sets: dict, all_features: list, merged_df: pd.DataFrame):
    banner("STEP 3 — ClinVar held-out split")

    clinvar_df = merged_df[merged_df["variant"].isin(CLINVAR_ALL)].copy()
    clinvar_df["clinvar_label"] = clinvar_df["variant"].apply(
        lambda v: "pathogenic" if v in CLINVAR_PATHOGENIC else "benign"
    )

    missing = [v for v in CLINVAR_ALL if v not in merged_df["variant"].values]
    if missing:
        print(f"  WARNING: ClinVar variants not found in FEATS: {missing}")

    print(f"  ClinVar test set : {len(clinvar_df)} variants")
    print(f"    Pathogenic : {(clinvar_df['clinvar_label']=='pathogenic').sum()}")
    print(f"    Benign     : {(clinvar_df['clinvar_label']=='benign').sum()}")

    train_label_sets = {}
    for name, df in label_sets.items():
        train = df[~df["variant"].isin(CLINVAR_ALL)].copy()
        train_label_sets[name] = train
        removed = len(df) - len(train)
        print(f"  [{name:<10}]  train={len(train):>4}  (held out {removed} ClinVar rows)")

    return train_label_sets, clinvar_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_models_v1(seed: int = 42, tune: bool = True) -> dict:
    """Original pipeline (SimpleImputer + original grids) — used as baseline."""
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
        return GridSearchCV(est, grid, cv=inner, scoring="r2", n_jobs=-1, refit=True)

    if tune:
        ridge_est = tuned(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]})
        rf_est    = tuned(
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
        )
    else:
        ridge_est = Ridge(alpha=1.0)
        rf_est    = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                           min_samples_leaf=2, random_state=seed, n_jobs=-1)

    return {
        "ridge": linear_pipe(ridge_est),
        "rf":    tree_pipe(rf_est),
        "xgb":   tree_pipe(xgb.XGBRegressor(
                     n_estimators=300, learning_rate=0.05, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=seed, verbosity=0, n_jobs=-1)),
        "lgbm":  tree_pipe(lgb.LGBMRegressor(
                     n_estimators=300, learning_rate=0.05, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=seed, verbosity=-1, n_jobs=-1)),
    }


def get_models_v2(seed: int = 42, tune: bool = True) -> dict:
    """
    Improved pipeline: KNNImputer, wider tuning grids, stacking ensemble.
    """
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)

    def linear_pipe(est):
        return Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("scl", StandardScaler()),
            ("est", est),
        ])

    def tree_pipe(est):
        return Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("est", est),
        ])

    def tuned(est, grid):
        return GridSearchCV(est, grid, cv=inner, scoring="r2", n_jobs=-1, refit=True)

    if tune:
        ridge_est = tuned(Ridge(),
                          {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]})
        rf_est    = tuned(
            RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1),
            {"max_features":     ["sqrt", 0.5, 0.7],
             "min_samples_leaf": [1, 2, 4]},
        )
    else:
        ridge_est = Ridge(alpha=1.0)
        rf_est    = RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                           min_samples_leaf=2, random_state=seed, n_jobs=-1)

    stack_base = [
        ("rf",   RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                        min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ("xgb",  xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, verbosity=0, n_jobs=-1)),
        ("lgbm", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, verbosity=-1, n_jobs=-1)),
    ]
    stacking = StackingRegressor(
        estimators=stack_base,
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1,
    )

    return {
        "ridge": linear_pipe(ridge_est),
        "rf":    tree_pipe(rf_est),
        "xgb":   tree_pipe(xgb.XGBRegressor(
                     n_estimators=300, learning_rate=0.05, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=seed, verbosity=0, n_jobs=-1)),
        "lgbm":  tree_pipe(lgb.LGBMRegressor(
                     n_estimators=300, learning_rate=0.05, max_depth=3,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=seed, verbosity=-1, n_jobs=-1)),
        "stack": Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("scl", StandardScaler()),
            ("est", stacking),
        ]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — CROSS-VALIDATED EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cv(model_pipe, X: pd.DataFrame, y: pd.Series,
                n_folds: int = 5, n_repeats: int = 5, seed: int = 42,
                clf_threshold: float = None) -> dict:
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

    y_pred_oof = np.where(oof_cnt > 0, oof_sum / np.maximum(oof_cnt, 1), np.nan)

    thr = float(np.median(y_arr)) if clf_threshold is None else float(clf_threshold)
    y_bin = (y_arr <= thr).astype(int)
    if y_bin.min() != y_bin.max():
        damage_score = -y_pred_oof
        auroc = float(roc_auc_score(y_bin, damage_score))
        auprc = float(average_precision_score(y_bin, damage_score))
    else:
        auroc, auprc = np.nan, np.nan

    return {
        "r2":            np.array(r2s),
        "rmse":          np.array(rmses),
        "mae":           np.array(maes),
        "pearson":       np.array(pearsons),
        "spearman":      np.array(spearmans),
        "auroc":         auroc,
        "auprc":         auprc,
        "y_pred_oof":    y_pred_oof,
        "clf_threshold": thr,
    }


def evaluate_all(models: dict, X: pd.DataFrame, y: pd.Series,
                 n_folds=5, n_repeats=5, seed=42, clf_threshold=None) -> dict:
    results = {}
    for name, pipe in models.items():
        res = evaluate_cv(pipe, X, y, n_folds, n_repeats, seed, clf_threshold)
        results[name] = res
        print(
            f"    {name:<12}  "
            f"Spearman={np.nanmean(res['spearman']):+.3f}  "
            f"R²={np.nanmean(res['r2']):+.3f}±{np.nanstd(res['r2']):.3f}  "
            f"RMSE={res['rmse'].mean():.3f}  "
            f"AUROC={res['auroc']:.3f}"
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — ABLATION STUDIES
# ──────────────────────────────────────────────────────────────────────────────

def ablation_leave_one_feature_out(model_pipe, X, y, n_folds=5, n_repeats=5, seed=42):
    print("      Computing baseline …", end=" ", flush=True)
    base = evaluate_cv(clone(model_pipe), X, y, n_folds, n_repeats, seed)
    base_sp = np.nanmean(base["spearman"])
    base_r2 = np.nanmean(base["r2"])
    base_auroc = base["auroc"]
    print(f"Spearman={base_sp:.3f}  R²={base_r2:.3f}")

    rows = []
    for feat in X.columns:
        X_drop = X.drop(columns=[feat])
        res = evaluate_cv(clone(model_pipe), X_drop, y, n_folds, n_repeats, seed)
        sp_without = np.nanmean(res["spearman"])
        rows.append({
            "feature":        feat,
            "group":          feat_to_group(X.columns.tolist())[feat],
            "sp_without":     sp_without,
            "sp_std_without": np.nanstd(res["spearman"]),
            "r2_without":     np.nanmean(res["r2"]),
            "r2_std_without": np.nanstd(res["r2"]),
            "delta_sp":       base_sp - sp_without,
            "delta_r2":       base_r2 - np.nanmean(res["r2"]),
            "auroc_without":  res["auroc"],
            "delta_auroc":    base_auroc - res["auroc"],
        })
        print(f"      − {feat:<30}  Spearman={sp_without:+.3f}  "
              f"Δsp={base_sp - sp_without:+.3f}")

    df = pd.DataFrame(rows).sort_values("delta_sp", ascending=False)
    df.insert(0, "baseline_sp", base_sp)
    df.insert(1, "baseline_r2", base_r2)
    return df


def ablation_leave_one_group_out(model_pipe, X, y, n_folds=5, n_repeats=5, seed=42):
    print("      Computing baseline …", end=" ", flush=True)
    base = evaluate_cv(clone(model_pipe), X, y, n_folds, n_repeats, seed)
    base_sp = np.nanmean(base["spearman"])
    base_r2 = np.nanmean(base["r2"])
    print(f"Spearman={base_sp:.3f}  R²={base_r2:.3f}")

    rows = []
    for grp_name, grp_feats in FEATURE_GROUPS.items():
        to_remove = [f for f in grp_feats if f in X.columns]
        if not to_remove:
            continue
        remaining = [f for f in X.columns if f not in to_remove]
        if not remaining:
            print(f"      WARNING: removing '{grp_name}' leaves no features; skipped.")
            continue
        res = evaluate_cv(clone(model_pipe), X[remaining], y, n_folds, n_repeats, seed)
        sp_without = np.nanmean(res["spearman"])
        rows.append({
            "group":            grp_name,
            "n_removed":        len(to_remove),
            "features_removed": ", ".join(to_remove),
            "sp_without":       sp_without,
            "sp_std_without":   np.nanstd(res["spearman"]),
            "r2_without":       np.nanmean(res["r2"]),
            "r2_std_without":   np.nanstd(res["r2"]),
            "delta_sp":         base_sp - sp_without,
            "delta_r2":         base_r2 - np.nanmean(res["r2"]),
            "auroc_without":    res["auroc"],
            "delta_auroc":      base["auroc"] - res["auroc"],
        })
        print(f"      − {grp_name:<15}  Spearman={sp_without:+.3f}  "
              f"Δsp={base_sp - sp_without:+.3f}")

    df = pd.DataFrame(rows).sort_values("delta_sp", ascending=False)
    df.insert(0, "baseline_sp", base_sp)
    df.insert(1, "baseline_r2", base_r2)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — BOOTSTRAP PREDICTION FOR CLINVAR
# ──────────────────────────────────────────────────────────────────────────────

def make_bootstrap_pipe(fitted_model_pipe) -> Pipeline:
    new_steps = []
    for name, step in fitted_model_pipe.steps:
        if isinstance(step, GridSearchCV):
            new_steps.append((name, clone(step.best_estimator_)))
        else:
            new_steps.append((name, clone(step)))
    return Pipeline(new_steps)


def bootstrap_predict(model_pipe, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, n_bootstrap: int = 200,
                       seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_tr = len(X_train)
    preds = np.zeros((n_bootstrap, len(X_test)))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_tr, size=n_tr)
        Xb = X_train.iloc[idx].reset_index(drop=True)
        yb = y_train.values[idx]
        m = clone(model_pipe)
        m.fit(Xb, yb)
        preds[i] = m.predict(X_test)

    return {
        "mean":   preds.mean(axis=0),
        "median": np.median(preds, axis=0),
        "ci_lo":  np.percentile(preds, 2.5,  axis=0),
        "ci_hi":  np.percentile(preds, 97.5, axis=0),
        "std":    preds.std(axis=0),
        "all":    preds,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — CLINVAR CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_clinvar_classification(clinvar_df: pd.DataFrame,
                                     pred_bs: dict,
                                     train_median: float,
                                     label_strategy: str,
                                     model_name: str) -> dict:
    y_true = (clinvar_df["clinvar_label"] == "pathogenic").astype(int).values
    y_mean = pred_bs["mean"]
    ci_lo  = pred_bs["ci_lo"]
    ci_hi  = pred_bs["ci_hi"]

    damage_score = -y_mean
    auroc = roc_auc_score(y_true, damage_score)
    auprc = average_precision_score(y_true, damage_score)
    fpr, tpr, _ = roc_curve(y_true, damage_score)

    y_pred  = (y_mean < train_median).astype(int)
    acc     = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  ─── ClinVar Classification  [{label_strategy}]  [{model_name}] ───")
    print(f"    AUROC             : {auroc:.3f}")
    print(f"    AUPRC             : {auprc:.3f}")
    print(f"    Accuracy          : {acc:.3f}  ({int(acc*20)}/20 correct)")
    print(f"    Balanced Accuracy : {bal_acc:.3f}")
    print(f"    Sensitivity (path): {sens:.3f}  ({tp}/{tp+fn} pathogenic correct)")
    print(f"    Specificity (ben) : {spec:.3f}  ({tn}/{tn+fp} benign correct)")
    print(f"    Threshold used    : {train_median:.4f}  (training-set median)")

    detail_df = pd.DataFrame({
        "variant":       clinvar_df["variant"].values,
        "clinvar_label": clinvar_df["clinvar_label"].values,
        "pred_score":    y_mean,
        "ci_lo":         ci_lo,
        "ci_hi":         ci_hi,
        "ci_width":      ci_hi - ci_lo,
        "pred_class":    np.where(y_pred == 1, "pathogenic", "benign"),
        "correct":       (y_pred == y_true),
    }).sort_values("pred_score")

    print("\n  Per-variant predictions (sorted by predicted score):")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(detail_df[["variant", "clinvar_label", "pred_score",
                      "ci_lo", "ci_hi", "pred_class", "correct"]].to_string(index=False))
    pd.reset_option("display.float_format")

    return {
        "auroc": auroc, "auprc": auprc, "accuracy": acc,
        "balanced_accuracy": bal_acc, "sensitivity": sens, "specificity": spec,
        "fpr": fpr, "tpr": tpr, "detail_df": detail_df, "threshold": train_median,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 10 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#EBEBEB",
    "grid.linewidth":     0.6,
    "figure.dpi":         120,
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
        ax.axvline(y.mean(), color="#555", linestyle=":", linewidth=1.2,
                   label=f"mean={y.mean():.2f}")
        ax.set_title(f"[{name}]  n={len(y)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Label Distribution by Strategy (Training Set — ClinVar excluded)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "01_label_distributions.png")


def plot_score_correlation(df: pd.DataFrame, outdir: Path):
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


def plot_improvement_comparison(results_v1: dict, results_v2: dict, outdir: Path):
    lname = "both" if "both" in results_v1 else list(results_v1.keys())[0]
    shared = [m for m in results_v1[lname] if m in results_v2[lname]]

    metrics      = ["spearman", "r2", "pearson"]
    metric_labels = ["Spearman ρ", "R²", "Pearson r"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(shared))
    w = 0.35

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]
        v1_m = [np.nanmean(results_v1[lname][m][metric]) for m in shared]
        v2_m = [np.nanmean(results_v2[lname][m][metric]) for m in shared]
        v1_s = [np.nanstd(results_v1[lname][m][metric])  for m in shared]
        v2_s = [np.nanstd(results_v2[lname][m][metric])  for m in shared]

        ax.bar(x - w/2, v1_m, w, yerr=v1_s, label="v1 — original",
               color="#AEC7E8", capsize=3, alpha=0.9)
        ax.bar(x + w/2, v2_m, w, yerr=v2_s, label="v2 — pruned features",
               color="#4E79A7", capsize=3, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(shared, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{mlabel}  [{lname}]", fontsize=10, fontweight="bold")
        ax.set_ylabel(mlabel, fontsize=10)
        if metric in ("r2", "spearman", "pearson"):
            ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)
        if col == 0:
            ax.legend(fontsize=9)

        for xi, (m1, m2) in enumerate(zip(v1_m, v2_m)):
            delta = m2 - m1
            col_ann = "#2ca02c" if delta > 0 else "#d62728"
            ax.text(xi + w/2, m2 + max(v2_s[xi], 0.01) + 0.01,
                    f"{delta:+.3f}", ha="center", va="bottom",
                    fontsize=7, color=col_ann, fontweight="bold")

    fig.suptitle(
        "v1 (All Features) vs v2 (Ablation-Pruned)  —  Repeated 5-Fold CV on Training Set\n"
        "10 features kept (9 base + am_pathogenicity²)  |  Green = improvement",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, outdir / "03_improvement_comparison.png")


def plot_model_comparison(results_all: dict, outdir: Path, suffix: str = "v2"):
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
                   alpha=0.85, edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
            ax.set_title(f"[{lname}] — {mlabel}", fontsize=10, fontweight="bold")
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
            if metric in ("r2", "spearman", "pearson"):
                ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Model Comparison ({suffix} — Pruned Features)  —  Repeated K-Fold CV",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / f"04_model_comparison_{suffix}.png")


def plot_cv_distributions(results_all: dict, outdir: Path, suffix: str = "v2"):
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
    fig.suptitle(f"Cross-Validation R² Distributions ({suffix} — Pruned Features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / f"05_cv_distributions_{suffix}.png")


def plot_predictions_scatter(results_all: dict, label_sets: dict,
                              best_model: str, outdir: Path):
    label_names = list(results_all.keys())
    n = len(label_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for col, lname in enumerate(label_names):
        ax = axes[0][col]
        if best_model not in results_all[lname]:
            ax.set_visible(False)
            continue
        y_true = label_sets[lname]["label"].values
        y_pred = results_all[lname][best_model]["y_pred_oof"]
        r2     = np.nanmean(results_all[lname][best_model]["r2"])
        spear  = np.nanmean(results_all[lname][best_model]["spearman"])
        rmse   = results_all[lname][best_model]["rmse"].mean()
        ax.scatter(y_true, y_pred, c=LABEL_COLORS[lname], alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.4)
        lo = min(y_true.min(), np.nanmin(y_pred)) - 0.05
        hi = max(y_true.max(), np.nanmax(y_pred)) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
        ax.set_xlabel("Actual", fontsize=11)
        ax.set_ylabel("Predicted (OOF)", fontsize=11)
        ax.set_title(
            f"[{lname}]  n={len(y_true)}\n"
            f"Spearman={spear:.3f}  R²={r2:.3f}  RMSE={rmse:.3f}",
            fontsize=10, fontweight="bold",
        )
    fig.suptitle(f"OOF Predicted vs Actual — {best_model.upper()} (v2 pruned)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / f"06_predictions_{best_model}.png")


def plot_feature_importance(model_pipe, feature_names: list,
                             label_name: str, model_name: str, outdir: Path):
    est = model_pipe.named_steps["est"]
    if isinstance(est, GridSearchCV):
        est = est.best_estimator_
    if isinstance(est, StackingRegressor):
        imps = []
        for _, base_est in est.estimators_:
            if hasattr(base_est, "feature_importances_"):
                imps.append(base_est.feature_importances_)
        if not imps:
            return None
        imp = np.mean(imps, axis=0)
    elif hasattr(est, "feature_importances_"):
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
    ax.set_title(f"Feature Importance (Pruned)\n{model_name.upper()}  |  label: {label_name}",
                 fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                      if any(g_map[f] == g for f in feature_names)]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, outdir / f"07_feature_importance_{label_name}_{model_name}.png")
    return imp_df


def plot_feature_ablation(abl_df: pd.DataFrame, label_name: str,
                           model_name: str, outdir: Path):
    df = abl_df.sort_values("delta_sp", ascending=True)
    shade = ["#E15759" if d < 0 else "#59A14F" for d in df["delta_sp"]]
    g_map = feat_to_group(df["feature"].tolist())
    bar_colors = [GROUP_COLORS.get(g_map[f], "#999") for f in df["feature"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df) * 0.38)))
    ax = axes[0]
    ax.barh(df["feature"], df["delta_sp"], color=shade, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.9)
    ax.set_xlabel("Δ Spearman ρ  (positive = feature contributes)", fontsize=11)
    ax.set_title(
        f"Feature Ablation (LOO)\n{model_name.upper()}  |  [{label_name}]  "
        f"baseline Spearman={abl_df['baseline_sp'].iloc[0]:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax2 = axes[1]
    ax2.barh(df["feature"], df["r2_without"],
             xerr=df["r2_std_without"], color=bar_colors,
             alpha=0.85, edgecolor="white", capsize=3)
    ax2.axvline(abl_df["baseline_r2"].iloc[0], color="#333", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("R² without feature  (mean ± std)", fontsize=11)
    ax2.set_title("R² When Feature Is Removed", fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    lp = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax2.legend(handles=lp, loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, outdir / f"08_ablation_features_{label_name}_{model_name}.png")


def plot_group_ablation(abl_df: pd.DataFrame, label_name: str,
                         model_name: str, outdir: Path):
    df = abl_df.sort_values("delta_sp", ascending=False)
    colors = [GROUP_COLORS.get(g, "#999") for g in df["group"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel in zip(axes, ["delta_sp", "delta_r2"],
                                  ["Δ Spearman ρ", "ΔR²"]):
        bars = ax.bar(df["group"], df[metric], color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Feature Group", fontsize=11)
        ax.set_ylabel(ylabel + "  (positive = group helps)", fontsize=11)
        ax.set_title(f"Group Ablation  |  {ylabel}\n{model_name.upper()}  |  [{label_name}]",
                     fontsize=11, fontweight="bold")
        for bar, val in zip(bars, df[metric]):
            ypos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.006
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(
        f"Feature Group Ablation  |  baseline Spearman={abl_df['baseline_sp'].iloc[0]:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, outdir / f"09_ablation_groups_{label_name}_{model_name}.png")


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
        ax.set_title(title + "  (Repeated K-Fold CV — Pruned)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Label Strategy", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)
    fig.suptitle("Model × Label Strategy Performance Overview (v2 — Ablation-Pruned Features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "10_best_model_heatmap.png")


def plot_clinvar_predictions(clf_result: dict, label_strategy: str,
                              model_name: str, outdir: Path):
    detail = clf_result["detail_df"].sort_values(["clinvar_label", "pred_score"])

    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = {"pathogenic": "#E15759", "benign": "#4E79A7"}
    markers = {"pathogenic": "^", "benign": "o"}

    for _, row in detail.iterrows():
        c = colors[row["clinvar_label"]]
        m = markers[row["clinvar_label"]]
        err_lo = row["pred_score"] - row["ci_lo"]
        err_hi = row["ci_hi"] - row["pred_score"]
        ax.errorbar(
            row["variant"], row["pred_score"],
            yerr=[[max(err_lo, 0)], [max(err_hi, 0)]],
            fmt=m, color=c, capsize=4, markersize=9,
            alpha=1.0 if row["correct"] else 0.35,
            linewidth=1.5,
        )
        if not row["correct"]:
            ax.annotate("✗", (row["variant"], row["pred_score"]),
                        ha="center", va="bottom", fontsize=10,
                        color="#333", fontweight="bold")

    ax.axhline(clf_result["threshold"], color="#333", linestyle="--", linewidth=1.4,
               label=f"threshold = {clf_result['threshold']:.3f}  (training median)")
    ax.set_xlabel("Variant", fontsize=10)
    ax.set_ylabel("Predicted MAVE score  (± 95% bootstrap CI)", fontsize=10)
    ax.set_title(
        f"ClinVar Held-Out Predictions  [{label_strategy}]  [{model_name}]\n"
        f"AUROC = {clf_result['auroc']:.3f}   "
        f"Acc = {clf_result['accuracy']:.3f}  ({int(clf_result['accuracy']*20)}/20)   "
        f"Sens = {clf_result['sensitivity']:.3f}   "
        f"Spec = {clf_result['specificity']:.3f}   "
        f"Bal-Acc = {clf_result['balanced_accuracy']:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(color="#E15759", label="Pathogenic (ClinVar)"),
        Patch(color="#4E79A7", label="Benign (ClinVar)"),
        plt.Line2D([0], [0], color="#333", linestyle="--", label="Classification threshold"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, outdir / f"11_clinvar_predictions_{label_strategy}_{model_name}.png")


def plot_clinvar_roc(clf_results_all: dict, outdir: Path, model_name: str = ""):
    fig, ax = plt.subplots(figsize=(6, 6))
    for lname, res in clf_results_all.items():
        ax.plot(res["fpr"], res["tpr"],
                color=LABEL_COLORS.get(lname, "#999"), linewidth=2,
                label=f"{lname}  AUROC={res['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        f"ROC Curve — ClinVar Classification\n"
        f"Pathogenic (n=10) vs Benign (n=10)  |  {model_name}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, outdir / f"12_clinvar_roc_{model_name}.png")


def plot_clinvar_summary_heatmap(clf_results_all: dict, outdir: Path, model_name: str = ""):
    rows = []
    for lname, res in clf_results_all.items():
        rows.append({
            "label":              lname,
            "AUROC":              res["auroc"],
            "Accuracy":           res["accuracy"],
            "Balanced Accuracy":  res["balanced_accuracy"],
            "Sensitivity":        res["sensitivity"],
            "Specificity":        res["specificity"],
            "AUPRC":              res["auprc"],
        })
    df = pd.DataFrame(rows).set_index("label")

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.8)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7},
                vmin=0.0, vmax=1.0)
    ax.set_title(
        f"ClinVar Classification Metrics  |  {model_name}\n"
        "Pathogenic (n=10) vs Benign (n=10)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("Label Strategy", fontsize=10)
    plt.tight_layout()
    _save(fig, outdir / f"13_clinvar_classification_heatmap_{model_name}.png")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 11 — BEST MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def find_best_combo(results_all: dict, rank_metric: str = "spearman"):
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
    sort_col   = f"mean_{rank_metric}"
    ascending  = rank_metric not in HIGHER_IS_BETTER
    summary_df = summary_df.sort_values(sort_col, ascending=ascending, na_position="last")
    best = summary_df.iloc[0]
    return (best["label"], best["model"]), summary_df


# ──────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--feats",   default="aim3/BRCA1_FEATS.csv")
    p.add_argument("--mave",    default="aim3/BRCA1_MAVE.csv")
    p.add_argument("--outdir",  default="brca1_results_v2")
    p.add_argument("--model",   default="all",
                   choices=["ridge", "rf", "xgb", "lgbm", "stack", "all"])
    p.add_argument("--labels",  default="all",
                   choices=["both", "any", "cisplatin", "hdr", "all"])
    p.add_argument("--n-folds",    type=int, default=5,   dest="n_folds")
    p.add_argument("--n-repeats",  type=int, default=5,   dest="n_repeats")
    p.add_argument("--n-bootstrap",type=int, default=200, dest="n_bootstrap",
                   help="Bootstrap resamples for ClinVar CIs  [default: 200]")
    p.add_argument("--rank-metric", default="spearman", dest="rank_metric",
                   choices=["spearman", "pearson", "r2", "rmse", "mae", "auroc", "auprc"])
    p.add_argument("--class-threshold", type=float, default=None, dest="class_threshold")
    p.add_argument("--tune",    dest="tune", action="store_true",  default=True)
    p.add_argument("--no-tune", dest="tune", action="store_false")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--ablation",    dest="ablation", action="store_true",  default=True)
    p.add_argument("--no-ablation", dest="ablation", action="store_false")
    p.add_argument("--find-best",   dest="find_best", action="store_true", default=True)
    p.add_argument("--no-baseline", dest="run_baseline", action="store_false", default=True,
                   help="Skip v1 baseline run (faster)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 62}")
    print(f"  BRCA1 MAVE Prediction Pipeline  v2 — Ablation-Pruned Features")
    print(f"  10 features: 3 structural + 3 biochemical + 2 AlphaMissense + am²")
    print(f"{'═' * 62}")
    print(f"  feats       : {args.feats}")
    print(f"  mave        : {args.mave}")
    print(f"  outdir      : {outdir}")
    print(f"  model       : {args.model}")
    print(f"  labels      : {args.labels}")
    print(f"  n_folds     : {args.n_folds}   n_repeats: {args.n_repeats}")
    print(f"  n_bootstrap : {args.n_bootstrap}")
    print(f"  tune        : {args.tune}")
    print(f"  seed        : {args.seed}")

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    merged = load_and_merge(args.feats, args.mave)
    merged.to_csv(outdir / "merged_data.csv", index=False)

    # ── 2. Preprocess (applies feature engineering to all rows) ───────────────
    banner("STEP 2 — Feature preprocessing + engineering")
    merged, all_features = preprocess_features(merged, engineer=True)

    # ── 3. Build full label sets ──────────────────────────────────────────────
    label_sets_full = build_label_sets(merged)
    plot_score_correlation(merged, outdir)

    # ── 4. ClinVar hold-out split ─────────────────────────────────────────────
    train_label_sets, clinvar_df = clinvar_split(
        label_sets_full, all_features, merged
    )
    plot_label_distributions(train_label_sets, outdir)

    if args.labels != "all":
        train_label_sets = {args.labels: train_label_sets[args.labels]}

    # ── 5. Build model dicts ──────────────────────────────────────────────────
    all_models_v1 = get_models_v1(seed=args.seed, tune=args.tune)
    all_models_v2 = get_models_v2(seed=args.seed, tune=args.tune)

    if args.model != "all":
        all_models_v1 = {k: v for k, v in all_models_v1.items() if k == args.model}
        all_models_v2 = {k: v for k, v in all_models_v2.items()
                         if k == args.model or k == "stack"}

    # ── 6. Baseline (v1 imputer, same pruned features) evaluation ────────────
    results_v1 = {}
    if args.run_baseline:
        banner("STEP 4 — Baseline (v1 imputer) cross-validated evaluation")
        for lname, ldf in train_label_sets.items():
            print(f"\n  Label strategy: [{lname}]  n={len(ldf)}")
            results_v1[lname] = evaluate_all(
                all_models_v1, ldf[all_features], ldf["label"],
                args.n_folds, args.n_repeats, args.seed, args.class_threshold,
            )

    # ── 7. Improved (v2 KNN + stacking) evaluation ───────────────────────────
    banner("STEP 5 — Improved (v2) cross-validated evaluation")
    results_v2 = {}
    for lname, ldf in train_label_sets.items():
        print(f"\n  Label strategy: [{lname}]  n={len(ldf)}")
        results_v2[lname] = evaluate_all(
            all_models_v2, ldf[all_features], ldf["label"],
            args.n_folds, args.n_repeats, args.seed, args.class_threshold,
        )

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    banner("STEP 6 — Visualisations")
    if args.run_baseline and results_v1:
        plot_improvement_comparison(results_v1, results_v2, outdir)
    plot_model_comparison(results_v2, outdir, suffix="v2")
    plot_cv_distributions(results_v2, outdir, suffix="v2")

    # ── 9. Best v2 model ──────────────────────────────────────────────────────
    rm = args.rank_metric
    higher_better = rm in HIGHER_IS_BETTER

    model_avg_v2 = {}
    for m in all_models_v2:
        vals = [_scalar(results_v2[l][m], rm) for l in results_v2 if m in results_v2[l]]
        model_avg_v2[m] = np.nanmean(vals) if vals else np.nan

    best_model_name = (max if higher_better else min)(
        {k: v for k, v in model_avg_v2.items() if not np.isnan(v)},
        key=lambda k: model_avg_v2[k],
    )
    print(f"\n  Best v2 model (avg {rm}): "
          f"{best_model_name.upper()} = {model_avg_v2[best_model_name]:.3f}")

    plot_predictions_scatter(results_v2, train_label_sets, best_model_name, outdir)

    for lname, ldf in train_label_sets.items():
        X, y = ldf[all_features], ldf["label"]
        imp_pipe = clone(all_models_v2[best_model_name])
        imp_pipe.fit(X, y)
        imp_df = plot_feature_importance(imp_pipe, all_features, lname, best_model_name, outdir)
        if imp_df is not None:
            imp_df.to_csv(
                outdir / f"feature_importance_{lname}_{best_model_name}.csv", index=False
            )

    # ── 10. ClinVar hold-out prediction ───────────────────────────────────────
    banner("STEP 7 — ClinVar held-out prediction & classification")
    X_cv = clinvar_df[all_features]
    clf_results: dict = {}

    for lname, ldf in train_label_sets.items():
        X_tr = ldf[all_features]
        y_tr = ldf["label"]
        train_median = float(y_tr.median())

        print(f"\n  [{lname}]  n_train={len(ldf)}  "
              f"train_median={train_median:.3f}")

        final_model = clone(all_models_v2[best_model_name])
        final_model.fit(X_tr, y_tr)
        bootstrap_pipe = make_bootstrap_pipe(final_model)

        print(f"  Computing {args.n_bootstrap} bootstrap samples …",
              end=" ", flush=True)
        bs = bootstrap_predict(
            bootstrap_pipe, X_tr, y_tr, X_cv,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        print("done")

        clf_res = evaluate_clinvar_classification(
            clinvar_df, bs, train_median, lname, best_model_name
        )
        clf_results[lname] = clf_res

        clf_res["detail_df"].to_csv(
            outdir / f"clinvar_predictions_{lname}_{best_model_name}.csv", index=False
        )
        plot_clinvar_predictions(clf_res, lname, best_model_name, outdir)

    plot_clinvar_roc(clf_results, outdir, model_name=best_model_name)
    plot_clinvar_summary_heatmap(clf_results, outdir, model_name=best_model_name)

    clf_summary = pd.DataFrame([
        {"label": l, "auroc": r["auroc"], "accuracy": r["accuracy"],
         "balanced_accuracy": r["balanced_accuracy"],
         "sensitivity": r["sensitivity"], "specificity": r["specificity"],
         "auprc": r["auprc"]}
        for l, r in clf_results.items()
    ])
    clf_summary.to_csv(outdir / "clinvar_classification_summary.csv", index=False)
    print("\n  ClinVar classification summary:")
    print(clf_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── 11. Ablation ──────────────────────────────────────────────────────────
    if args.ablation:
        banner("STEP 8 — Ablation studies (pruned feature set)")
        best_label_name = (max if higher_better else min)(
            results_v2,
            key=lambda l: _scalar(results_v2[l][best_model_name], rm)
            if best_model_name in results_v2[l]
            else (float("-inf") if higher_better else float("inf")),
        )
        best_ldf = train_label_sets[best_label_name]
        X_best = best_ldf[all_features]
        y_best = best_ldf["label"]

        print(f"\n  Ablation: [{best_label_name}]  n={len(best_ldf)}  "
              f"model: {best_model_name.upper()}")

        print("\n  Leave-one-feature-out:")
        abl_feat = ablation_leave_one_feature_out(
            all_models_v2[best_model_name], X_best, y_best,
            args.n_folds, args.n_repeats, args.seed,
        )
        abl_feat.to_csv(
            outdir / f"ablation_features_{best_label_name}_{best_model_name}.csv", index=False
        )
        plot_feature_ablation(abl_feat, best_label_name, best_model_name, outdir)

        print("\n  Leave-one-group-out:")
        abl_grp = ablation_leave_one_group_out(
            all_models_v2[best_model_name], X_best, y_best,
            args.n_folds, args.n_repeats, args.seed,
        )
        abl_grp.to_csv(
            outdir / f"ablation_groups_{best_label_name}_{best_model_name}.csv", index=False
        )
        plot_group_ablation(abl_grp, best_label_name, best_model_name, outdir)

        print("\n  Top features by Δ Spearman:")
        print(abl_feat[["feature", "group", "delta_sp", "delta_r2", "r2_without"]]
              .head(10).to_string(index=False))
        print("\n  Group ablation:")
        print(abl_grp[["group", "n_removed", "delta_sp", "delta_r2", "r2_without"]]
              .to_string(index=False))

    # ── 12. Best combo summary ────────────────────────────────────────────────
    if args.find_best:
        banner("STEP 9 — Best model × label selection (v2 pruned)")
        (best_label, best_model), summary_df = find_best_combo(results_v2, rm)
        summary_df.to_csv(outdir / "model_label_summary_v2.csv", index=False)
        plot_best_model_heatmap(summary_df, outdir)

        top = summary_df.iloc[0]
        print(f"\n  ★ BEST v2 COMBINATION  (ranked by {rm})")
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
    print(f"  Pipeline v2 complete.  Results → {outdir}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
