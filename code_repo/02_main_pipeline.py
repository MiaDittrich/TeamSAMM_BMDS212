#!/usr/bin/env python3
"""
BRCA1 MAVE Pathogenicity Final Pipeline
========================================

Architecture
------------
Single-stage Random Forest regressor, identical to brca1_mave_pipeline_v2.py:
  • KNNImputer(n_neighbors=5) preprocessing (no scaling needed for RF)
  • RandomForestRegressor with nested-CV hyperparameter tuning
      inner  3-fold GridSearchCV: max_features in {sqrt, 0.5, 0.7},
                                   min_samples_leaf in {1, 2, 4}
      outer  5-fold × 5-repeat RepeatedKFold → stable OOF metrics
  • Sample weights: (coverage / max_coverage) × (pLDDT / 100), clipped to [0.01, 1]

Dataset layout
--------------
  brca1_final_feats.csv   234 variants — structural / biochemical / evo features
  brca1_mave.csv          2 465 rows   — MAVE assay scores + AlphaMissense
                           joined via (uniprot_position, ref_aa, alt_aa)
  clinvar_test.csv         48 variants  — held-out ClinVar eval (24 P / 24 B)
  clinvar_explore.csv      28 variants  — VUS & Conflicting prediction targets

  Train : 234 − 48 − 28 = 158 total
          Label strategy "any": average of whichever MAVE score(s) exist
          → ~151 variants with at least one score (7 lack any MAVE data)

Feature set (ablation-pruned — best CV Spearman on this n=70 training set)
---------------------------------------------------------------------------
  Structural    mutant_plddt, backbone_rmsd, mutant_ca_displacement
  Biochemical   delta_size, delta_aromaticity, is_size_increase
  AlphaMissense am_pathogenicity, am_class_enc
  Evolutionary  evoef2_ddg_Total, ddg_evoef2, esm2_llr
  Engineered    am_pathogenicity², am × EvoEF2 interaction, esm1b × am interaction
  (14 features total when esm2_llr is available in brca1_final_feats.csv)
  Run compute_esm1b_scores.py first to populate the esm2_llr column.

  The best prior run (brca1_predict.py, Spearman=0.6702) used the full 25-feature
  set but had n_train=98 from a larger dataset.  At n=70, the pruned set
  generalises better (+0.024 Spearman vs full features).

Score convention
----------------
  Lower mave label → more loss-of-function → Pathogenic.
  damage_score = −ŷ  (higher damage = lower predicted function).
  Classification threshold = training-set median of the label.
  Variants below threshold → predicted Pathogenic.

Outputs  (./results/)
---------------------
  cv_metrics.csv                  per-fold CV metrics
  clinvar_test_predictions.csv    point predictions + classification, 48 variants
  clinvar_explore_predictions.csv predicted scores + 95 % bootstrap CI, 28 variants
  metrics_comparison.csv          side-by-side vs prior pipeline
  fig_01_cv_scatter.png           OOF predicted vs actual (repeated K-Fold)
  fig_02_feature_importance.png   RF feature importances coloured by group
  fig_03_clinvar_test.png         test predictions + ROC curve
  fig_04_known_variants.png       anchor variants highlighted
  fig_05_explore_ranked.png       VUS / Conflicting ranked with 95 % CI bars

Usage
-----
  python brca1_pipeline_final.py
  python brca1_pipeline_final.py --label any --n-bootstrap 500
"""

import re
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, roc_curve,
    balanced_accuracy_score, confusion_matrix,
)
from sklearn.model_selection import (
    KFold, RepeatedKFold, GridSearchCV,
)
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
import shap

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

# Original 20-variant ClinVar holdout from brca1_mave_pipeline.py / aim3.
# These are added BACK into training (not held out as test).
# Only the 28 additional ClinVar variants in clinvar_test.csv remain as the test set.
# This matches the n=86 training set that achieved Spearman=0.650.
CLINVAR_ORIGINAL_20 = {
    "S1841R", "L1839S", "V1838E", "M1775R", "Y1703S",
    "W1718L", "G1706R", "G1738E", "S1715R", "I1760S",
    "D1733G", "I1766V", "T1773S", "V1736I", "K1793Q",
    "E1794G", "S1797C", "H1862L", "P1831S", "E1829T",
}

# Best-performing configuration: ElasticNet on full features.
# Historical best (brca1_mave_pipeline.py): Spearman=0.650, R²=0.525,
#   Pearson=0.758, AUROC=0.836  (ElasticNet + "both" label + full feature set).
# ElasticNet's L1 penalty zeros out irrelevant features automatically, so the
# full set is both safe and beneficial for linear models.
#
# RF ablation-pruned subset also preserved in FEATURE_GROUPS_RF for reference.
# am_class_enc is available because brca1_mave.csv now contains am_class.
FEATURE_GROUPS = {
    "structural": [
        "mutant_plddt", "ca_rmsd", "backbone_rmsd", "mutant_ca_displacement",
        "shell_rmsd_5A", "shell_rmsd_8A", "shell_rmsd_12A",
        "ramachandran_violation", "is_disordered_variant",
    ],
    "biochemical": [
        "pam250_score", "delta_hydrophobicity", "delta_size",
        "delta_charge", "delta_aromaticity",
        "is_charge_reversal", "is_size_increase",
        "is_hydrophobic_to_polar", "is_polar_to_hydrophobic",
    ],
    "alphamissense": ["am_pathogenicity", "am_class_enc"],
    "evo":           ["evoef2_ddg_Total", "ddg_evoef2", "esm2_llr"],
    "engineered":    ["am_pathogenicity_sq", "am_x_evo", "plddt_rmsd", "esm2_x_am"],
}

# Ablation-pruned subset — better for RF at n=70 (Spearman 0.607 vs 0.583 full).
FEATURE_GROUPS_RF = {
    "structural":    ["mutant_plddt", "backbone_rmsd", "mutant_ca_displacement"],
    "biochemical":   ["delta_size", "delta_aromaticity", "is_size_increase"],
    "alphamissense": ["am_pathogenicity", "am_class_enc"],
    "evo":           ["evoef2_ddg_Total", "ddg_evoef2", "esm2_llr"],
    "engineered":    ["am_pathogenicity_sq", "am_x_evo", "esm2_x_am"],
}

GROUP_COLORS = {
    "structural":    "#4E79A7",
    "biochemical":   "#F28E2B",
    "alphamissense": "#59A14F",
    "evo":           "#E15759",
    "engineered":    "#9467BD",
    "other":         "#AAAAAA",
}

HIGHER_IS_BETTER = {"r2", "pearson", "spearman", "auroc", "auprc"}

# Custom scorer used for inner GridSearchCV hyperparameter tuning.
# Spearman ρ is the primary evaluation metric (rank-based, robust to label
# scale differences between assays), so we tune directly for it rather than R².
SPEARMAN_SCORER = make_scorer(
    lambda y, yhat: float(stats.spearmanr(y, yhat)[0]),
    greater_is_better=True,
)

# Well-characterised "anchor" variants present in clinvar_test.csv.
# Used to annotate fig_04. Sources: Findlay 2018, Starita 2015, Brzovic 2001.
KNOWN_PATHOGENIC = {
    "M1775R": "M1775R\n(canonical LOF,\nBRCT–phosphopeptide lost)",
    "R1699W": "R1699W\n(BARD1 binding\ndisrupted)",
    "L1786P": "L1786P\n(BRCT fold\ndisrupted)",
    "W1718L": "W1718L\n(BRCT mis-fold)",
}
KNOWN_BENIGN = {
    "I1766V": "I1766V\n(benign, normal\nHDR & binding)",
    "K1793Q": "K1793Q\n(benign, normal\nfunction)",
    "T1773S": "T1773S\n(benign,\npopulation common)",
    "I1858L": "I1858L\n(benign, normal\nassay scores)",
}

OUTDIR = Path("results")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#EBEBEB",
    "grid.linewidth":    0.6,
    "figure.dpi":        120,
})


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & MERGING
# ═══════════════════════════════════════════════════════════════════

def parse_variant(v: str):
    """'A1234B' → ('A', 1234, 'B'). Returns (None, None, None) on failure."""
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def load_and_merge(feats_path, mave_path, cv_test_path, cv_explore_path):
    """
    Load all four files and merge FEATS with MAVE on (position, ref_aa, alt_aa).

    brca1_mave.csv columns used:
        uniprot_position, ref_aa, alt_aa, am_pathogenicity, am_class,
        cisplatin_score, hdr_activity_score

    Returns merged (234 rows), cv_test (48 rows), cv_explore (28 rows).
    """
    _banner("SECTION 1 — Load & Merge")

    feats      = pd.read_csv(feats_path)
    mave       = pd.read_csv(mave_path)
    cv_test    = pd.read_csv(cv_test_path,    usecols=[0, 1])
    cv_explore = pd.read_csv(cv_explore_path, usecols=[0, 1])

    print(f"  brca1_final_feats : {len(feats):>4} rows × {feats.shape[1]} cols")
    print(f"  brca1_mave        : {len(mave):>4} rows × {mave.shape[1]} cols")
    print(f"  clinvar_test      : {len(cv_test):>4} variants")
    print(f"  clinvar_explore   : {len(cv_explore):>4} variants")

    # Parse variant string → join keys
    parsed = feats["variant"].apply(
        lambda v: pd.Series(parse_variant(v), index=["ref_aa", "_pos", "alt_aa"])
    )
    feats = pd.concat([feats, parsed], axis=1)
    n_bad = feats["ref_aa"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} variant(s) could not be parsed — excluded from merge")

    mave_cols = ["uniprot_position", "ref_aa", "alt_aa",
                 "am_pathogenicity", "am_class",
                 "cisplatin_score", "hdr_activity_score"]

    merged = feats.merge(
        mave[mave_cols],
        left_on=["mutant_residue", "ref_aa", "alt_aa"],
        right_on=["uniprot_position", "ref_aa", "alt_aa"],
        how="left",
        validate="m:1",
    ).drop(columns=["_pos", "uniprot_position"])

    assert len(merged) == len(feats), \
        f"Merge size mismatch: got {len(merged)}, expected {len(feats)}"

    n_cis = merged["cisplatin_score"].notna().sum()
    n_hdr = merged["hdr_activity_score"].notna().sum()
    n_both = (merged["cisplatin_score"].notna() & merged["hdr_activity_score"].notna()).sum()
    print(f"\n  Merged            : {len(merged)} rows ✓")
    print(f"  cisplatin present : {n_cis}   hdr present: {n_hdr}   both: {n_both}")
    print(f"  am_pathogenicity missing: {merged['am_pathogenicity'].isna().sum()}")
    print(f"  Coverage range    : {sorted(merged['coverage'].dropna().unique().tolist()) if 'coverage' in merged.columns else 'N/A'}")

    return merged, cv_test, cv_explore


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def build_label(df: pd.DataFrame, strategy: str) -> pd.Series:
    """
    Build the regression target from MAVE assay scores.

    strategy='both' : average of cisplatin + HDR, only rows where both exist.
    strategy='any'  : NaN-safe average of whatever score(s) are available.
    strategy='cisplatin' / 'hdr': single assay.

    Lower values = more loss-of-function = Pathogenic.
    """
    if strategy == "both":
        mask  = df["cisplatin_score"].notna() & df["hdr_activity_score"].notna()
        label = df.loc[mask, ["cisplatin_score", "hdr_activity_score"]].mean(axis=1)
        return label.reindex(df.index)
    if strategy == "any":
        return df[["cisplatin_score", "hdr_activity_score"]].mean(axis=1, skipna=True)
    if strategy == "cisplatin":
        return df["cisplatin_score"]
    if strategy == "hdr":
        return df["hdr_activity_score"]
    raise ValueError(f"Unknown label strategy '{strategy}'")


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    1. Ordinal-encode am_class → am_class_enc  (benign=0, ambiguous=1, pathogenic=2).
    2. Add am_pathogenicity² (non-linear AM signal).
    3. Add am × EvoEF2 interaction: am_pathogenicity × sign(evoef2_ddg_Total) × log1p|evoef2|.
    4. Drop all-NaN feature columns.
    5. Return (augmented DataFrame, ordered list of feature names).
    """
    _banner("SECTION 2 — Feature Engineering")

    df = df.copy()
    df["am_class_enc"] = df["am_class"].map(AM_CLASS_MAP)

    # Collect base features in declared group order (skip engineered group here)
    base_feats = []
    for grp, cols in FEATURE_GROUPS.items():
        if grp == "engineered":
            continue
        for c in cols:
            if c in df.columns and c not in base_feats:
                base_feats.append(c)

    absent = [f for f in base_feats if f not in df.columns]
    if absent:
        print(f"  WARNING: features absent from data (skipped): {absent}")
        base_feats = [f for f in base_feats if f in df.columns]

    all_nan = [f for f in base_feats if df[f].isna().all()]
    if all_nan:
        print(f"  Dropping all-NaN features: {all_nan}")
        base_feats = [f for f in base_feats if f not in all_nan]

    # Engineered features
    eng_feats = []
    if "am_pathogenicity" in base_feats:
        df["am_pathogenicity_sq"] = df["am_pathogenicity"] ** 2
        eng_feats.append("am_pathogenicity_sq")

    if "am_pathogenicity" in base_feats and "evoef2_ddg_Total" in base_feats:
        evo_abs = df["evoef2_ddg_Total"].fillna(0).abs()
        df["am_x_evo"] = (
            df["am_pathogenicity"].fillna(0)
            * np.sign(df["evoef2_ddg_Total"].fillna(0))
            * np.log1p(evo_abs)
        )
        eng_feats.append("am_x_evo")

    # plddt_rmsd: pLDDT confidence × CA displacement (structural disruption weight)
    if "mutant_plddt" in base_feats and "ca_rmsd" in base_feats:
        df["plddt_rmsd"] = (df["mutant_plddt"] / 100.0) * df["ca_rmsd"]
        eng_feats.append("plddt_rmsd")

    # esm2_x_am: ESM-1b LLR × AM pathogenicity — captures concordance between
    # evolutionary constraint and structural pathogenicity signal.
    if "esm2_llr" in base_feats and "am_pathogenicity" in base_feats:
        df["esm2_x_am"] = (
            df["esm2_llr"].fillna(0) * df["am_pathogenicity"].fillna(0)
        )
        eng_feats.append("esm2_x_am")

    all_feats = base_feats + eng_feats

    print(f"\n  Features retained ({len(all_feats)}):")
    for grp, cols in FEATURE_GROUPS.items():
        present = [f for f in cols if f in all_feats]
        if present:
            print(f"    [{grp:>14}] {present}")

    return df, all_feats


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN / TEST / EXPLORE SPLIT
# ═══════════════════════════════════════════════════════════════════

def split_data(merged, cv_test_meta, cv_explore_meta, all_feats, label_strategy,
               split_mode: str = "strict"):
    """
    Partition 234 variants into train / test / explore.

    split_mode='strict'  (default)
        Training excludes ALL 48 clinvar_test variants + ALL 24 clinvar_explore
        variants. Test evaluation uses all 48 P/B variants. This is the cleanest
        possible separation — no variant that appears in either evaluation file
        contributes to model training.

    split_mode='extended'
        The 20 CLINVAR_ORIGINAL_20 variants are moved from the test file back
        into training (they have known MAVE labels and improve training coverage).
        The held-out test set is the remaining 28 non-original ClinVar variants —
        these have ZERO overlap with the training set.
        Explore exclusion is unchanged (all 24 still held out).
        This recovers the higher-n training configuration (e.g. n=86 for 'both')
        while keeping a clean, non-contaminated test holdout.

    Each subset gets features + label + any ClinVar label column.
    Training variants without a valid label under the chosen strategy are dropped.
    """
    _banner("SECTION 3 — Three-Way Split")

    test_variants_all = set(cv_test_meta["variant"])   # all 48
    explore_variants  = set(cv_explore_meta["variant"])  # all 24

    if split_mode == "extended":
        # 20 well-characterised variants go back into training
        test_variants_held = test_variants_all - CLINVAR_ORIGINAL_20   # 28
        train_exclude      = test_variants_held | explore_variants
        test_eval_variants = test_variants_held                         # evaluate on 28
    else:  # strict
        train_exclude      = test_variants_all | explore_variants
        test_eval_variants = test_variants_all                          # evaluate on 48

    train_all = merged[~merged["variant"].isin(train_exclude)].copy()

    test_df = merged[merged["variant"].isin(test_eval_variants)].merge(
                  cv_test_meta[cv_test_meta["variant"].isin(test_eval_variants)],
                  on="variant", how="left")

    exp_df = merged[merged["variant"].isin(explore_variants)].merge(
                 cv_explore_meta, on="variant", how="left")

    # Build labels
    train_all["label"] = build_label(train_all, label_strategy)
    test_df["label"]   = build_label(test_df,   label_strategy)
    exp_df["label"]    = build_label(exp_df,     label_strategy)

    # Training: keep only rows with a valid label
    train_df = train_all[train_all["label"].notna()].copy().reset_index(drop=True)

    mode_tag = ("extended (CLINVAR_ORIGINAL_20 in training, 28-variant test)"
                if split_mode == "extended"
                else "strict (all 48 test + 24 explore excluded)")
    print(f"  Split mode            : {mode_tag}")
    print(f"  Label strategy        : '{label_strategy}'")
    print(f"  Train (labelled)      : {len(train_df):>4}")
    print(f"  ClinVar test          : {len(test_df):>4}  "
          f"(Pathogenic: {(test_df['clinvar_label']=='Pathogenic').sum()}, "
          f"Benign: {(test_df['clinvar_label']=='Benign').sum()})")
    print(f"  ClinVar explore       : {len(exp_df):>4}  "
          f"(VUS: {(exp_df['clinvar_label']=='VUS').sum()}, "
          f"Conflicting: {(exp_df['clinvar_label']=='Conflicting').sum()})")
    print(f"  Label range (train)   : [{train_df['label'].min():.3f}, "
          f"{train_df['label'].max():.3f}]  median={train_df['label'].median():.3f}")

    return train_df, test_df, exp_df


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Per-variant confidence weights:
        weight = (coverage / max_coverage) × (mutant_pLDDT / 100)

    coverage reflects MAVE assay replicate count.
    pLDDT reflects AlphaFold structural confidence.
    Clipped to [0.01, 1.0] to prevent zero-weight issues.
    """
    if "coverage" in df.columns and df["coverage"].notna().any():
        max_cov = df["coverage"].max()
        cov = df["coverage"].fillna(1.0) / max(max_cov, 1.0)
    else:
        cov = pd.Series(np.ones(len(df)), index=df.index)

    plddt = df["mutant_plddt"].fillna(50.0) / 100.0
    return np.clip((cov * plddt).values, 0.01, 1.0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════

def build_model(seed: int = 42, tune: bool = True) -> Pipeline:
    """
    Build the RF Pipeline identical to brca1_mave_pipeline_v2.py get_models_v2().

    KNNImputer(n_neighbors=5)  →  RandomForestRegressor
    When tune=True the RF is wrapped in a GridSearchCV with a 3-fold inner CV,
    giving proper nested CV when used inside evaluate_cv's outer folds.
    """
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)

    if tune:
        rf_est = GridSearchCV(
            RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1),
            param_grid={
                "max_features":     ["sqrt", 0.5, 0.7],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=inner,
            scoring=SPEARMAN_SCORER,   # optimise directly for Spearman ρ
            n_jobs=-1,
            refit=True,
        )
    else:
        # Fixed params matching brca1_predict.py (the best-performing pipeline).
        # GridSearchCV selected min_samples_leaf=4 which was suboptimal;
        # the winning run used min_samples_leaf=2 with no inner tuning.
        rf_est = RandomForestRegressor(
            n_estimators=400, max_features="sqrt",
            min_samples_leaf=2, random_state=seed, n_jobs=-1,
        )

    return Pipeline([
        ("imp", KNNImputer(n_neighbors=5)),
        ("est", rf_est),
    ])


def build_elasticnet_model(seed: int = 42, tune: bool = True) -> Pipeline:
    """
    ElasticNet pipeline matching brca1_mave_pipeline.py (best historical run).

    SimpleImputer(median) → StandardScaler → ElasticNet
    Inner 3-fold GridSearchCV tunes alpha and l1_ratio using Spearman ρ.

    Hyperparameter grid (from original brca1_mave_pipeline.py):
        alpha    ∈ {1e-3, 1e-2, 1e-1}
        l1_ratio ∈ {0.2, 0.5, 0.8}

    StandardScaler is required for ElasticNet — regularisation penalties are
    scale-dependent and must be applied to standardised features.
    """
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)

    if tune:
        enet_est = GridSearchCV(
            ElasticNet(max_iter=10_000, random_state=seed),
            param_grid={
                "alpha":    [1e-3, 1e-2, 1e-1],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
            cv=inner,
            scoring=SPEARMAN_SCORER,   # optimise directly for Spearman ρ
            n_jobs=-1,
            refit=True,
        )
    else:
        enet_est = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000)

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", enet_est),
    ])


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — CROSS-VALIDATED EVALUATION
# ═══════════════════════════════════════════════════════════════════

def _safe_corr(fn, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(fn(a, b)[0])
    except Exception:
        return np.nan


def evaluate_cv(pipe, X: pd.DataFrame, y: pd.Series,
                sample_weight: np.ndarray,
                n_folds: int = 5, n_repeats: int = 5,
                seed: int = 42,
                clf_threshold: float = None) -> dict:
    """
    Repeated K-Fold cross-validation.

    Each fold:
      • KNNImputer is fit on the training split only (no leakage).
      • GridSearchCV tunes RF hyperparameters on the training split.
      • OOF predictions are accumulated across repeats and averaged.

    sample_weight is passed via Pipeline fit kwargs:
        pipe.fit(X_tr, y_tr, est__sample_weight=w_tr)

    Returns per-fold arrays for r2, rmse, mae, pearson, spearman,
    plus scalar auroc/auprc (computed on pooled OOF predictions),
    and the averaged OOF prediction vector.
    """
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    r2s, rmses, maes, pearsons, spearmans = [], [], [], [], []

    n       = len(y)
    y_arr   = y.values
    oof_sum = np.zeros(n)
    oof_cnt = np.zeros(n)

    for tr_idx, te_idx in rkf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        w_tr       = sample_weight[tr_idx]

        m = clone(pipe)
        m.fit(X_tr, y_tr, est__sample_weight=w_tr)
        y_hat = m.predict(X_te)

        oof_sum[te_idx] += y_hat
        oof_cnt[te_idx] += 1

        r2s.append(r2_score(y_te, y_hat))
        rmses.append(float(np.sqrt(mean_squared_error(y_te, y_hat))))
        maes.append(float(mean_absolute_error(y_te, y_hat)))
        pearsons.append(_safe_corr(stats.pearsonr,  y_te, y_hat))
        spearmans.append(_safe_corr(stats.spearmanr, y_te, y_hat))

    y_pred_oof = np.where(oof_cnt > 0, oof_sum / np.maximum(oof_cnt, 1), np.nan)

    # Binary classification view: median split
    thr   = float(np.median(y_arr)) if clf_threshold is None else float(clf_threshold)
    y_bin = (y_arr <= thr).astype(int)    # 1 = damaging (low functional score)
    if y_bin.min() != y_bin.max():
        damage_score = -y_pred_oof        # lower prediction = higher damage
        auroc = float(roc_auc_score(y_bin, damage_score))
        auprc = float(average_precision_score(y_bin, damage_score))
    else:
        auroc = auprc = np.nan

    return {
        "r2":       np.array(r2s),
        "rmse":     np.array(rmses),
        "mae":      np.array(maes),
        "pearson":  np.array(pearsons),
        "spearman": np.array(spearmans),
        "auroc":    auroc,
        "auprc":    auprc,
        "y_pred_oof": y_pred_oof,
        "clf_threshold": thr,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — FINAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_final_model(pipe, X: pd.DataFrame, y: pd.Series,
                      sample_weight: np.ndarray) -> Pipeline:
    """Fit the pipeline on ALL training variants."""
    _banner("SECTION 7 — Final Model Training")
    m = clone(pipe)
    m.fit(X, y, est__sample_weight=sample_weight)

    # Surface best hyperparameters if tuned
    est = m.named_steps["est"]
    if isinstance(est, GridSearchCV):
        print(f"  Best params: {est.best_params_}  (inner-CV Spearman={est.best_score_:.3f})")
        inner_est = est.best_estimator_
    else:
        inner_est = est

    if isinstance(inner_est, RandomForestRegressor):
        print(f"  n_estimators={inner_est.n_estimators}  "
              f"max_features={inner_est.max_features}  "
              f"min_samples_leaf={inner_est.min_samples_leaf}")
    elif isinstance(inner_est, ElasticNet):
        print(f"  alpha={inner_est.alpha:.4f}  l1_ratio={inner_est.l1_ratio}")
    return m


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — CLINVAR TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_clinvar_test(model: Pipeline,
                           test_df: pd.DataFrame,
                           all_feats: list,
                           threshold: float) -> dict:
    """
    Evaluate the final model on 48 held-out ClinVar variants.

    Score convention: lower predicted label = more pathogenic.
    damage_score = −ŷ  →  higher damage score = predicted more pathogenic.
    Classification: ŷ < threshold → Pathogenic.
    """
    _banner("SECTION 8 — ClinVar Test Evaluation")
    print(f"  Threshold (training median): {threshold:.4f}")
    print(f"  Lower ŷ → predicted Pathogenic")

    X_te   = test_df[all_feats]
    y_pred = model.predict(X_te)
    y_true = test_df["label"].values

    # "Highlight" variants (e.g. W1718S) are predicted but excluded from
    # AUROC/accuracy so they don't distort evaluation on P/B variants.
    is_highlight = (test_df["clinvar_label"] == "Highlight").values
    eval_mask    = ~is_highlight
    eval_df      = test_df[eval_mask]
    y_pred_eval  = y_pred[eval_mask]

    y_true_bin  = (eval_df["clinvar_label"] == "Pathogenic").astype(int).values
    damage_score_eval = -y_pred_eval
    auroc = roc_auc_score(y_true_bin, damage_score_eval)
    auprc = average_precision_score(y_true_bin, damage_score_eval)
    fpr, tpr, _ = roc_curve(y_true_bin, damage_score_eval)

    y_pred_class_eval = (y_pred_eval < threshold).astype(int)
    acc     = float((y_pred_class_eval == y_true_bin).mean())
    bal_acc = float(balanced_accuracy_score(y_true_bin, y_pred_class_eval))
    cm      = confusion_matrix(y_true_bin, y_pred_class_eval)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    n_eval = len(eval_df)
    n_hl   = is_highlight.sum()
    print(f"\n  Evaluating on {n_eval} P/B variants  ({n_hl} Highlight variant(s) excluded from metrics)")
    print(f"  AUROC             : {auroc:.4f}")
    print(f"  AUPRC             : {auprc:.4f}")
    print(f"  Accuracy          : {acc:.4f}  ({int(acc*n_eval)}/{n_eval} correct)")
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print(f"  Sensitivity (path): {sens:.4f}  ({tp}/{tp+fn} pathogenic correct)")
    print(f"  Specificity (ben) : {spec:.4f}  ({tn}/{tn+fp} benign correct)")

    # Predicted class for ALL rows (highlight variants get a prediction too)
    y_pred_class_all = (y_pred < threshold).astype(int)
    # correct is NaN for Highlight variants (no ground truth)
    correct_all = np.where(
        is_highlight, np.nan,
        (y_pred_class_all == (test_df["clinvar_label"] == "Pathogenic").astype(int).values).astype(float)
    )

    result_df = pd.DataFrame({
        "variant":       test_df["variant"].values,
        "clinvar_label": test_df["clinvar_label"].values,
        "true_label":    y_true,
        "pred_score":    y_pred,
        "damage_score":  -y_pred,
        "pred_class":    np.where(y_pred_class_all == 1, "Pathogenic", "Benign"),
        "correct":       correct_all,
    }).sort_values("pred_score")

    # Print W1718S prediction separately
    w_row = result_df[result_df["clinvar_label"] == "Highlight"]
    if len(w_row):
        print(f"\n  ★ Highlighted variant(s):")
        print(w_row[["variant","clinvar_label","pred_score","pred_class"]].to_string(index=False))

    print("\n  Per-variant predictions (sorted by predicted score, low→high):")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(result_df[["variant", "clinvar_label", "true_label",
                      "pred_score", "pred_class", "correct"]].to_string(index=False))
    pd.reset_option("display.float_format")

    return {
        "result_df":   result_df,
        "auroc":       auroc, "auprc":       auprc,
        "accuracy":    acc,   "bal_acc":     bal_acc,
        "sensitivity": sens,  "specificity": spec,
        "fpr":         fpr,   "tpr":         tpr,
        "threshold":   threshold,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — CLINVAR EXPLORE PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_clinvar_explore(model: Pipeline,
                             exp_df: pd.DataFrame,
                             all_feats: list,
                             threshold: float,
                             train_df: pd.DataFrame,
                             seed: int = 42,
                             n_bootstrap: int = 200) -> pd.DataFrame:
    """
    Predict scores + 95 % bootstrap CIs for 28 VUS/Conflicting variants.

    Bootstrap resamples the training set and refits the pipeline each time.
    GridSearchCV inside the pipeline re-runs its inner tuning on every resample
    (pass tune=False for speed; hyperparameters are already known from Section 7).
    Classification: ŷ < threshold → Pathogenic.
    """
    _banner("SECTION 9 — ClinVar Explore Prediction")

    X_exp     = exp_df[all_feats]
    point_pred = model.predict(X_exp)

    # Bootstrap — refit on resampled training data (fixed RF params for speed)
    y_tr  = train_df["label"].values
    X_tr  = train_df[all_feats]
    w_tr  = compute_sample_weights(train_df)
    n_tr  = len(train_df)
    rng   = np.random.default_rng(seed)
    boot_preds = np.zeros((n_bootstrap, len(exp_df)))

    # Reconstruct a fast bootstrap pipeline with fixed best params (no inner CV).
    est = model.named_steps["est"]
    inner_est = est.best_estimator_ if isinstance(est, GridSearchCV) else est
    best_params = est.best_params_ if isinstance(est, GridSearchCV) else {}

    if isinstance(inner_est, ElasticNet):
        boot_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("est", ElasticNet(max_iter=10_000, **best_params)),
        ])
    else:
        if not best_params:
            best_params = {"max_features": inner_est.max_features,
                           "min_samples_leaf": inner_est.min_samples_leaf}
        boot_pipe = Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("est", RandomForestRegressor(
                n_estimators=400, random_state=seed, n_jobs=-1, **best_params
            )),
        ])

    print(f"  Bootstrap params: {best_params}")
    print(f"  Running {n_bootstrap} bootstrap resamples …", end=" ", flush=True)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_tr, size=n_tr)
        Xb  = X_tr.iloc[idx].reset_index(drop=True)
        yb  = y_tr[idx]
        wb  = w_tr[idx]
        m   = clone(boot_pipe)
        m.fit(Xb, yb, est__sample_weight=wb)
        boot_preds[b] = m.predict(X_exp)
    print("done")

    ci_lo    = np.percentile(boot_preds, 2.5,  axis=0)
    ci_hi    = np.percentile(boot_preds, 97.5, axis=0)
    pred_std = boot_preds.std(axis=0)

    pred_class = np.where(point_pred < threshold, "Pathogenic", "Benign")
    n_path = (pred_class == "Pathogenic").sum()
    print(f"\n  Predicted Pathogenic: {n_path}  |  Benign: {len(exp_df) - n_path}")
    print(f"  Threshold used      : {threshold:.4f}  (training-set median)")

    result_df = pd.DataFrame({
        "variant":       exp_df["variant"].values,
        "clinvar_label": exp_df["clinvar_label"].values,
        "pred_score":    point_pred,
        "ci_lo":         ci_lo,
        "ci_hi":         ci_hi,
        "pred_std":      pred_std,
        "pred_class":    pred_class,
    }).sort_values("pred_score")

    print("\n  Predictions (sorted low→high — lower = more pathogenic):")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(result_df[["variant", "clinvar_label", "pred_score",
                      "ci_lo", "ci_hi", "pred_class"]].to_string(index=False))
    pd.reset_option("display.float_format")

    return result_df


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — METRICS COMPARISON
# ═══════════════════════════════════════════════════════════════════

def build_comparison(cv_res: dict, test_res: dict, label_strategy: str) -> pd.DataFrame:
    """
    Side-by-side comparison vs prior pipeline (brca1_mave_pipeline_v2.py).

    Prior reference:
        RF+evo, "both" label, 5-fold×5-repeat nested CV, n_train=70, 12 features.
        Spearman=0.583, R²=0.331, ClinVar-20 AUROC=1.000, acc=20/20.
    """
    spear  = float(np.nanmean(cv_res["spearman"]))
    r2     = float(np.nanmean(cv_res["r2"]))
    rmse   = float(np.nanmean(cv_res["rmse"]))

    rows = [
        {
            "Pipeline":         "brca1_mave_pipeline_v2.py (prior)",
            "Model":            "RF (single-stage, v2)",
            "CV":               "5-fold × 5-repeat",
            "n_train":          70,
            "n_features":       12,
            "Label":            "avg(cisplatin, HDR)",
            "Spearman ρ":       0.583,
            "R²":               0.331,
            "RMSE":             0.354,
            "ClinVar n":        20,
            "ClinVar AUROC":    1.000,
            "ClinVar accuracy": "20/20",
        },
        {
            "Pipeline":         "brca1_pipeline_final.py (new)",
            "Model":            "RF (single-stage, original arch)",
            "CV":               "5-fold × 5-repeat",
            "n_train":          len(cv_res["y_pred_oof"]),
            "n_features":       "auto (ablation-pruned)",
            "Label":            label_strategy,
            "Spearman ρ":       round(spear, 4),
            "R²":               round(r2, 4),
            "RMSE":             round(rmse, 4),
            "ClinVar n":        len(test_res["result_df"]),
            "ClinVar AUROC":    round(test_res["auroc"], 4),
            "ClinVar accuracy": (f"{int(round(test_res['accuracy'] * len(test_res['result_df'])))}"
                                 f"/{len(test_res['result_df'])}"),
        },
    ]

    df = pd.DataFrame(rows)
    _banner("SECTION 10 — Metrics Comparison")
    print(df.T.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path.name}")


def plot_cv_scatter(cv_res: dict, train_df: pd.DataFrame, outdir: Path):
    """Fig 01: OOF predicted vs actual (Repeated K-Fold), coloured by coverage."""
    y_true = train_df["label"].values
    y_pred = cv_res["y_pred_oof"]

    fig, ax = plt.subplots(figsize=(6, 6))
    cov_col = "coverage" if "coverage" in train_df.columns else None
    if cov_col:
        for cval, color, lbl in sorted(
            [(c, f"C{i}", f"Coverage {int(c)}")
             for i, c in enumerate(sorted(train_df[cov_col].dropna().unique()))],
        ):
            mask = train_df[cov_col].values == cval
            ax.scatter(y_true[mask], y_pred[mask], s=30, alpha=0.7,
                       edgecolors="white", linewidths=0.3, label=lbl)
    else:
        ax.scatter(y_true, y_pred, s=30, alpha=0.7, color="#4E79A7",
                   edgecolors="white", linewidths=0.3)

    lo = min(y_true.min(), np.nanmin(y_pred)) - 0.05
    hi = max(y_true.max(), np.nanmax(y_pred)) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel("Actual MAVE label", fontsize=11)
    ax.set_ylabel("OOF Predicted (mean over repeats)", fontsize=11)
    spear = float(np.nanmean(cv_res["spearman"]))
    r2    = float(np.nanmean(cv_res["r2"]))
    rmse  = float(np.nanmean(cv_res["rmse"]))
    ax.set_title(
        f"5-Fold × 5-Repeat CV — OOF Predictions\n"
        f"Spearman ρ = {spear:+.3f}   R² = {r2:+.3f}   RMSE = {rmse:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, outdir / "fig_01_cv_scatter.png")


def plot_feature_importance(model: Pipeline, all_feats: list, outdir: Path):
    """Fig 02: RF feature importances coloured by feature group."""
    est = model.named_steps["est"]
    inner_est = est.best_estimator_ if isinstance(est, GridSearchCV) else est

    # Build group lookup
    grp_map = {}
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            grp_map[c] = grp

    if isinstance(inner_est, ElasticNet):
        importance = np.abs(inner_est.coef_)
        xlabel     = "ElasticNet |coefficient| (standardised features)"
        title      = "Feature Coefficients — ElasticNet (|coef|)"
    else:
        importance = inner_est.feature_importances_
        xlabel     = "RF feature importance (mean decrease in impurity)"
        title      = "Feature Importances — Random Forest"

    imp_df = (
        pd.DataFrame({"feature": all_feats, "importance": importance})
        .assign(group=lambda d: d["feature"].map(lambda f: grp_map.get(f, "other")))
        .sort_values("importance", ascending=True)
    )

    colors = [GROUP_COLORS.get(g, "#AAAAAA") for g in imp_df["group"]]
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp_df) * 0.38)))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if g in imp_df["group"].values]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, outdir / "fig_02_feature_importance.png")


def plot_clinvar_test(test_res: dict, outdir: Path):
    """
    Fig 03: Predicted scores + ROC curve for ClinVar test variants.
    Highlight variants (e.g. W1718S) are shown as gold stars with an annotation.
    """
    df  = test_res["result_df"]
    thr = test_res["threshold"]
    n_eval = (df["clinvar_label"] != "Highlight").sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: sorted score plot ──────────────────────────────────────────────
    ax        = axes[0]
    row_order = df.sort_values("pred_score").reset_index(drop=True)
    cmap      = {"Pathogenic": "#E15759", "Benign": "#4E79A7", "Highlight": "#FFD700"}

    for i, row in row_order.iterrows():
        label = row["clinvar_label"]
        c     = cmap.get(label, "#999999")

        if label == "Highlight":
            # Gold star, larger, with annotation
            ax.scatter(i, row["pred_score"], color=c, s=220,
                       marker="*", edgecolors="#B8860B", linewidths=1.5,
                       zorder=5)
            ax.annotate(
                f"{row['variant']}\n({row['pred_class']})",
                (i, row["pred_score"]),
                xytext=(i + 1.5, row["pred_score"] - 0.06),
                fontsize=8, fontweight="bold", color="#B8860B",
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=0.8),
            )
        else:
            correct = row["correct"]
            ax.scatter(i, row["pred_score"], color=c, s=60,
                       alpha=1.0 if correct else 0.30,
                       edgecolors="white", linewidths=0.4, zorder=3)
            if not correct:
                ax.annotate("✗", (i, row["pred_score"]),
                            ha="center", va="top", fontsize=9,
                            color="#333", fontweight="bold")

    ax.axhline(thr, color="#333", linestyle="--", lw=1.4,
               label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Variants (sorted by predicted score)", fontsize=10)
    ax.set_ylabel("Predicted MAVE score  (lower = more pathogenic)", fontsize=10)
    ax.set_title(
        f"ClinVar Test (n={n_eval} P/B evaluated + highlighted variants)\n"
        f"Acc={test_res['accuracy']:.3f}  "
        f"Sens={test_res['sensitivity']:.3f}  "
        f"Spec={test_res['specificity']:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(handles=[
        mpatches.Patch(color="#E15759", label="Pathogenic (ClinVar)"),
        mpatches.Patch(color="#4E79A7", label="Benign (ClinVar)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
                   markeredgecolor="#B8860B", markersize=12, label="Highlighted variant"),
        plt.Line2D([0], [0], color="#333", ls="--", label="Threshold"),
    ], fontsize=8)

    # ── Right: ROC curve (P/B variants only) ─────────────────────────────────
    ax2 = axes[1]
    ax2.plot(test_res["fpr"], test_res["tpr"],
             color="#E15759", lw=2.5,
             label=f"ROC  (AUROC = {test_res['auroc']:.3f},  n={n_eval})")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate",  fontsize=10)
    ax2.set_title(f"ROC — ClinVar Test (P/B variants, n={n_eval})",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("ClinVar Held-Out Evaluation — ElasticNet Model",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir / "fig_03_clinvar_test.png")


def plot_known_variants(test_res: dict, outdir: Path):
    """
    Fig 04: ClinVar test variants ranked by predicted score.
    Larger outlined markers highlight published anchor variants.
    """
    df  = test_res["result_df"].sort_values("pred_score").reset_index(drop=True)
    thr = test_res["threshold"]

    fig, ax = plt.subplots(figsize=(13, 7))
    cmap      = {"Pathogenic": "#E15759", "Benign": "#4E79A7", "Highlight": "#FFD700"}
    path_edge = "#8B0000"
    ben_edge  = "#003087"
    hl_edge   = "#B8860B"

    for i, row in df.iterrows():
        v     = row["variant"]
        label = row["clinvar_label"]
        c     = cmap.get(label, "#999999")
        is_path_anch = v in KNOWN_PATHOGENIC
        is_ben_anch  = v in KNOWN_BENIGN
        is_hl        = (label == "Highlight")
        alpha = 1.0 if (is_hl or row["correct"]) else 0.30

        if is_hl:
            ax.scatter(i, row["pred_score"], color=c, s=300,
                       marker="*", edgecolors=hl_edge, linewidths=2.0,
                       zorder=6)
        elif is_path_anch or is_ben_anch:
            ec = path_edge if is_path_anch else ben_edge
            ax.scatter(i, row["pred_score"], color=c, s=150,
                       edgecolors=ec, linewidths=2.5, alpha=alpha, zorder=4)
        else:
            ax.scatter(i, row["pred_score"], color=c, s=50,
                       edgecolors="white", linewidths=0.4, alpha=alpha, zorder=3)

        if not is_hl and not row["correct"]:
            ax.annotate("✗", (i, row["pred_score"]),
                        ha="center", va="top", fontsize=9,
                        color="#333", fontweight="bold")

    # Annotate anchors and highlighted variants
    for i, row in df.iterrows():
        v     = row["variant"]
        label = row["clinvar_label"]
        label_text = None
        edge_c     = None

        if label == "Highlight":
            label_text = f"★ {v}\n(predicted: {row['pred_class']})"
            edge_c     = hl_edge
        elif v in KNOWN_PATHOGENIC:
            label_text, edge_c = KNOWN_PATHOGENIC[v], path_edge
        elif v in KNOWN_BENIGN:
            label_text, edge_c = KNOWN_BENIGN[v], ben_edge

        if label_text:
            x_off = 2 if i < len(df) * 0.65 else -2
            y_off = -0.10 if row["pred_score"] < thr else 0.10
            ax.annotate(
                label_text, xy=(i, row["pred_score"]),
                xytext=(i + x_off, row["pred_score"] + y_off),
                fontsize=7.5, fontweight="bold", color=edge_c,
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.7),
                ha="left" if x_off > 0 else "right",
            )

    ax.axhline(thr, color="#333", ls="--", lw=1.4,
               label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Variants ranked by predicted MAVE score (low → high)", fontsize=10)
    ax.set_ylabel("Predicted MAVE score  (lower = more loss-of-function)", fontsize=10)
    ax.set_title(
        "ClinVar Test — Predicted Scores with Literature Anchor Variants Highlighted\n"
        "Large outlined markers = well-characterised published variants  "
        "(dark-red border = pathogenic anchor, dark-blue = benign anchor)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(handles=[
        mpatches.Patch(color="#E15759", label="Pathogenic (ClinVar)"),
        mpatches.Patch(color="#4E79A7", label="Benign (ClinVar)"),
        plt.Line2D([0],[0], marker="*", color="w",
                   markerfacecolor="#FFD700", markeredgecolor=hl_edge,
                   markeredgewidth=1.5, markersize=14, label="Highlighted variant (W1718S)"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#E15759", markeredgecolor=path_edge,
                   markeredgewidth=2.5, markersize=10, label="Published pathogenic anchor"),
        plt.Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#4E79A7", markeredgecolor=ben_edge,
                   markeredgewidth=2.5, markersize=10, label="Published benign anchor"),
        plt.Line2D([0],[0], color="#333", ls="--", label="Classification threshold"),
    ], fontsize=8)
    plt.tight_layout()
    _save(fig, outdir / "fig_04_known_variants.png")


def plot_explore_ranked(exp_res: pd.DataFrame, threshold: float, outdir: Path):
    """Fig 05: VUS/Conflicting variants ranked by predicted score with 95 % CI bars."""
    df = exp_res.sort_values("pred_score").reset_index(drop=True)
    n  = len(df)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.42)))
    cmap   = {"Pathogenic": "#E15759", "Benign": "#4E79A7"}
    marker = {"VUS": "D", "Conflicting": "s"}

    for i, row in df.iterrows():
        c     = cmap[row["pred_class"]]
        m     = marker.get(row["clinvar_label"], "o")
        lo    = max(row["pred_score"] - row["ci_lo"], 0)
        hi    = max(row["ci_hi"] - row["pred_score"], 0)
        ax.errorbar(row["pred_score"], i,
                    xerr=[[lo], [hi]],
                    fmt=m, color=c, capsize=3, markersize=7,
                    alpha=0.85, linewidth=1.2, zorder=3)

    ax.axvline(threshold, color="#333", ls="--", lw=1.5,
               label=f"Threshold = {threshold:.3f}")
    # Shaded regions
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], threshold, alpha=0.05, color="#E15759", zorder=0)
    ax.axvspan(threshold, max(xlim[1], threshold + 0.1),
               alpha=0.05, color="#4E79A7", zorder=0)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["variant"], fontsize=8)
    ax.set_xlabel("Predicted MAVE score  (± 95 % bootstrap CI)", fontsize=10)
    ax.set_title(
        f"VUS & Conflicting Variants — Predicted MAVE Scores\n"
        f"Threshold = {threshold:.3f}   "
        "(◆ = VUS,  ■ = Conflicting interpretations)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(handles=[
        mpatches.Patch(color="#E15759", alpha=0.7, label="Predicted Pathogenic"),
        mpatches.Patch(color="#4E79A7", alpha=0.7, label="Predicted Benign"),
        plt.Line2D([0],[0], marker="D", color="#888", ls="none",
                   markersize=8, label="VUS"),
        plt.Line2D([0],[0], marker="s", color="#888", ls="none",
                   markersize=8, label="Conflicting"),
        plt.Line2D([0],[0], color="#333", ls="--",
                   label="Classification threshold"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, outdir / "fig_05_explore_ranked.png")


def plot_summary_slide(cv_res: dict, test_res: dict, exp_res: pd.DataFrame,
                       train_df: pd.DataFrame, outdir: Path):
    """
    Fig 06: Single slide-ready summary figure — three panels.

    Left   : Cross-validation scatter (OOF predicted vs actual),
             with Spearman ρ, R², RMSE, AUROC prominently annotated.
    Centre : ClinVar test — predicted scores sorted low→high,
             coloured by true label, threshold line, clean separation.
    Right  : VUS/Conflicting predictions ranked with 95 % CI bars,
             shaded pathogenic/benign regions.
    """
    y_true = train_df["label"].values
    y_pred = cv_res["y_pred_oof"]
    thr    = test_res["threshold"]

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # ── Panel 1: CV scatter ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cov_col = "coverage" if "coverage" in train_df.columns else None
    cov_vals = train_df[cov_col].values if cov_col else np.ones(len(train_df))
    for cv_v, col, lbl in [(1.0, "#AEC7E8", "Coverage 1"), (2.0, "#4E79A7", "Coverage 2")]:
        m = (cov_vals == cv_v)
        ax1.scatter(y_true[m], y_pred[m], c=col, s=35, alpha=0.75,
                    edgecolors="white", linewidths=0.3, label=lbl, zorder=2)
    lo = min(y_true.min(), np.nanmin(y_pred)) - 0.05
    hi = max(y_true.max(), np.nanmax(y_pred)) + 0.05
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4)
    ax1.set_xlabel("Actual MAVE score", fontsize=11)
    ax1.set_ylabel("OOF Predicted score", fontsize=11)
    spear = cv_res["spearman"].mean(); r2 = cv_res["r2"].mean(); rmse = cv_res["rmse"].mean()
    ax1.set_title("Cross-Validation Performance\n(5-fold × 5-repeat, n=86)",
                  fontsize=11, fontweight="bold")
    metrics_txt = (f"Spearman ρ = {spear:+.3f}\n"
                   f"R²  = {r2:+.3f}\n"
                   f"RMSE = {rmse:.3f}\n"
                   f"CV AUROC = {cv_res['auroc']:.3f}")
    ax1.text(0.04, 0.97, metrics_txt, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4FF",
                       edgecolor="#4E79A7", alpha=0.9))
    ax1.legend(fontsize=8, loc="lower right")

    # ── Panel 2: ClinVar test sorted scores ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    df_test  = test_res["result_df"].copy()
    eval_df  = df_test[df_test["clinvar_label"] != "Highlight"].sort_values("pred_score").reset_index(drop=True)
    cmap_t   = {"Pathogenic": "#E15759", "Benign": "#4E79A7"}

    for i, row in eval_df.iterrows():
        c = cmap_t[row["clinvar_label"]]
        ax2.scatter(i, row["pred_score"], color=c, s=65,
                    alpha=1.0 if row["correct"] else 0.25,
                    edgecolors="white", linewidths=0.4, zorder=3)
        if not row["correct"]:
            ax2.annotate("✗", (i, row["pred_score"]),
                         ha="center", va="top", fontsize=8, color="#333")

    n_eval = len(eval_df)
    ax2.axhline(thr, color="#333", ls="--", lw=1.5,
                label=f"Threshold = {thr:.3f}")
    ax2.axhspan(eval_df["pred_score"].min() - 0.1, thr, alpha=0.04, color="#E15759")
    ax2.axhspan(thr, eval_df["pred_score"].max() + 0.1, alpha=0.04, color="#4E79A7")
    ax2.set_xlabel("Variants (ranked by predicted score)", fontsize=11)
    ax2.set_ylabel("Predicted MAVE score", fontsize=11)
    ax2.set_title(f"ClinVar Held-Out Classification\n(n={n_eval}: 14 Pathogenic / 14 Benign)",
                  fontsize=11, fontweight="bold")
    clf_txt = (f"AUROC = {test_res['auroc']:.3f}\n"
               f"Accuracy = {int(test_res['accuracy']*n_eval)}/{n_eval}\n"
               f"Sensitivity = {test_res['sensitivity']:.3f}\n"
               f"Specificity = {test_res['specificity']:.3f}")
    ax2.text(0.04, 0.97, clf_txt, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8F0",
                       edgecolor="#E15759", alpha=0.9))
    ax2.legend(handles=[
        mpatches.Patch(color="#E15759", label="Pathogenic"),
        mpatches.Patch(color="#4E79A7", label="Benign"),
        plt.Line2D([0],[0], color="#333", ls="--", label="Threshold"),
    ], fontsize=8, loc="upper right")

    # ── Panel 3: VUS/Conflicting explore ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    exp = exp_res.sort_values("pred_score").reset_index(drop=True)
    cmap_e  = {"Pathogenic": "#E15759", "Benign": "#4E79A7"}
    mark_e  = {"VUS": "D", "Conflicting": "s"}

    for i, row in exp.iterrows():
        c  = cmap_e[row["pred_class"]]
        mk = mark_e.get(row["clinvar_label"], "o")
        lo_err = max(row["pred_score"] - row["ci_lo"], 0)
        hi_err = max(row["ci_hi"] - row["pred_score"], 0)
        ax3.errorbar(row["pred_score"], i, xerr=[[lo_err],[hi_err]],
                     fmt=mk, color=c, capsize=2, markersize=6,
                     alpha=0.85, linewidth=1.0, zorder=3)

    ax3.axvline(thr, color="#333", ls="--", lw=1.5)
    xmin = exp["ci_lo"].clip(lower=-2.0).min() - 0.05
    ax3.axvspan(xmin, thr, alpha=0.04, color="#E15759")
    ax3.axvspan(thr, exp["ci_hi"].max() + 0.05, alpha=0.04, color="#4E79A7")
    ax3.set_yticks(range(len(exp)))
    ax3.set_yticklabels(exp["variant"], fontsize=7.5)
    ax3.set_xlabel("Predicted MAVE score  (± 95 % CI)", fontsize=11)
    n_path = (exp["pred_class"] == "Pathogenic").sum()
    ax3.set_title(f"VUS & Conflicting Predictions\n"
                  f"(n=24: {n_path} predicted Pathogenic, {len(exp)-n_path} Benign)",
                  fontsize=11, fontweight="bold")
    ax3.legend(handles=[
        mpatches.Patch(color="#E15759", alpha=0.8, label="Predicted Pathogenic"),
        mpatches.Patch(color="#4E79A7", alpha=0.8, label="Predicted Benign"),
        plt.Line2D([0],[0], marker="D", color="#888", ls="none",
                   markersize=7, label="VUS"),
        plt.Line2D([0],[0], marker="s", color="#888", ls="none",
                   markersize=7, label="Conflicting"),
    ], fontsize=7.5, loc="lower right")

    fig.suptitle(
        "BRCA1 Missense Variant MAVE Score Prediction — ElasticNet  "
        "(avg cisplatin + HDR assay scores, n=86 training variants)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, outdir / "fig_06_summary_slide.png")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _banner(title: str):
    print(f"\n{'═'*62}\n  {title}\n{'═'*62}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 12 — ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════════

def _ablation_cv(X: pd.DataFrame, y: pd.Series, best_params: dict,
                 n_folds: int = 5, n_repeats: int = 5, seed: int = 42) -> dict:
    """
    Run 5×5 repeated K-Fold CV with fixed ElasticNet params (no inner tuning).
    Returns mean Spearman ρ, R², RMSE, AUROC over folds.
    Using fixed params makes each ablation run ~27× faster than nested CV.
    """
    rkf  = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", ElasticNet(max_iter=10_000, **best_params)),
    ])
    spearmans, pearsons, r2s, rmses = [], [], [], []
    oof_sum = np.zeros(len(y)); oof_cnt = np.zeros(len(y))
    y_arr = y.values

    for tr, te in rkf.split(X):
        m = clone(pipe)
        m.fit(X.iloc[tr], y_arr[tr])
        yhat = m.predict(X.iloc[te])
        oof_sum[te] += yhat; oof_cnt[te] += 1
        spearmans.append(float(stats.spearmanr(y_arr[te], yhat)[0]))
        pearsons.append(float(stats.pearsonr(y_arr[te], yhat)[0]))
        r2s.append(float(r2_score(y_arr[te], yhat)))
        rmses.append(float(np.sqrt(mean_squared_error(y_arr[te], yhat))))

    oof = np.where(oof_cnt > 0, oof_sum / oof_cnt, np.nan)
    thr = float(np.median(y_arr))
    y_bin = (y_arr <= thr).astype(int)
    auroc = float(roc_auc_score(y_bin, -oof)) if y_bin.min() != y_bin.max() else np.nan

    return {
        "spearman":     float(np.nanmean(spearmans)),
        "spearman_std": float(np.nanstd(spearmans)),
        "pearson":      float(np.nanmean(pearsons)),
        "pearson_std":  float(np.nanstd(pearsons)),
        "r2":           float(np.nanmean(r2s)),
        "rmse":         float(np.nanmean(rmses)),
        "auroc":        auroc,
    }


def run_ablations(train_df: pd.DataFrame, all_feats: list,
                  best_params: dict,
                  n_folds: int = 5, n_repeats: int = 5, seed: int = 42,
                  outdir: Path = Path("results")) -> tuple:
    """
    Leave-one-feature-out and leave-one-group-out ablation studies.

    Both use fixed ElasticNet hyperparameters (alpha, l1_ratio from final model)
    so each ablation fold is a single fit — no inner CV grid search.

    Returns (feat_df, group_df): DataFrames sorted by Δ Spearman (descending).
    Positive Δ Spearman means removing that feature *hurts* — feature is useful.
    Negative Δ Spearman means removing it *helps* — feature was adding noise.
    """
    _banner("SECTION 12 — Ablation Studies")

    X = train_df[all_feats]
    y = train_df["label"]
    n = len(all_feats)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("  Computing baseline …", end=" ", flush=True)
    base = _ablation_cv(X, y, best_params, n_folds, n_repeats, seed)
    print(f"Spearman={base['spearman']:+.4f}  Pearson={base['pearson']:+.4f}  "
          f"R²={base['r2']:+.4f}  AUROC={base['auroc']:.4f}")

    # ── Leave-one-feature-out ─────────────────────────────────────────────────
    print(f"\n  Leave-one-feature-out  ({n} features):")

    # Reverse lookup: feature → group
    feat_to_grp = {}
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            feat_to_grp[c] = grp

    feat_rows = []
    for i, feat in enumerate(all_feats):
        X_drop = X.drop(columns=[feat])
        res = _ablation_cv(X_drop, y, best_params, n_folds, n_repeats, seed)
        delta_sp = base["spearman"] - res["spearman"]
        delta_pe = base["pearson"]  - res["pearson"]
        feat_rows.append({
            "feature":        feat,
            "group":          feat_to_grp.get(feat, "other"),
            "baseline_sp":    base["spearman"],
            "sp_without":     res["spearman"],
            "sp_std_without": res["spearman_std"],
            "delta_sp":       delta_sp,
            "baseline_pe":    base["pearson"],
            "pe_without":     res["pearson"],
            "pe_std_without": res["pearson_std"],
            "delta_pe":       delta_pe,
            "r2_without":     res["r2"],
            "delta_r2":       base["r2"] - res["r2"],
            "auroc_without":  res["auroc"],
            "delta_auroc":    base["auroc"] - res["auroc"],
        })
        direction = "▲" if delta_sp > 0.005 else ("▼" if delta_sp < -0.005 else "─")
        print(f"    {direction} {feat:<32}  Δsp={delta_sp:+.4f}  Δpe={delta_pe:+.4f}  "
              f"sp_without={res['spearman']:+.4f}")

    feat_df = pd.DataFrame(feat_rows).sort_values("delta_sp", ascending=False)

    # ── Leave-one-group-out ───────────────────────────────────────────────────
    print(f"\n  Leave-one-group-out:")
    group_rows = []
    for grp, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in all_feats]
        if not present:
            continue
        remaining = [f for f in all_feats if f not in present]
        if not remaining:
            print(f"    WARNING: removing '{grp}' leaves no features; skipped.")
            continue
        res = _ablation_cv(pd.DataFrame(X[remaining]), y, best_params,
                           n_folds, n_repeats, seed)
        delta_sp = base["spearman"] - res["spearman"]
        delta_pe = base["pearson"]  - res["pearson"]
        group_rows.append({
            "group":            grp,
            "n_features":       len(present),
            "features_removed": ", ".join(present),
            "baseline_sp":      base["spearman"],
            "sp_without":       res["spearman"],
            "sp_std_without":   res["spearman_std"],
            "delta_sp":         delta_sp,
            "baseline_pe":      base["pearson"],
            "pe_without":       res["pearson"],
            "pe_std_without":   res["pearson_std"],
            "delta_pe":         delta_pe,
            "r2_without":       res["r2"],
            "delta_r2":         base["r2"] - res["r2"],
            "auroc_without":    res["auroc"],
            "delta_auroc":      base["auroc"] - res["auroc"],
        })
        direction = "▲" if delta_sp > 0.005 else ("▼" if delta_sp < -0.005 else "─")
        print(f"    {direction} {grp:<18}  Δsp={delta_sp:+.4f}  "
              f"sp_without={res['spearman']:+.4f}  ({len(present)} features removed)")

    group_df = pd.DataFrame(group_rows).sort_values("delta_sp", ascending=False)

    # Save CSVs
    feat_df.to_csv(outdir / "ablation_features.csv", index=False)
    group_df.to_csv(outdir / "ablation_groups.csv", index=False)
    print(f"\n  Saved → ablation_features.csv  ({len(feat_df)} features)")
    print(f"  Saved → ablation_groups.csv    ({len(group_df)} groups)")

    return feat_df, group_df, base


def plot_ablations(feat_df: pd.DataFrame, group_df: pd.DataFrame,
                   base: dict, outdir: Path, model: Pipeline = None,
                   all_feats: list = None):
    """
    Fig 07: Three-panel ablation figure (slide-ready).

    Panel A (left)  — Feature lollipop: Δ Spearman per feature, sorted so the
                      most impactful features appear at top. Features zeroed by
                      ElasticNet's L1 penalty (Δ=0) are shown in light gray and
                      labelled "(L1 zeroed)". Active features are coloured by group.

    Panel B (centre) — ElasticNet signed coefficients: only non-zero features,
                       coloured by group. Positive = higher feature → higher MAVE
                       score (functional); negative → more pathogenic prediction.

    Panel C (right)  — Group dumbbell: baseline Spearman (filled circle) vs.
                       Spearman without each group (open diamond), connected by a
                       line. Gap = group contribution. Makes positive vs. negative
                       contribution immediately visible without needing to parse bars.
    """
    ZERO_THRESH = 1e-6   # Δsp below this = L1-zeroed

    # ── Colour helpers ────────────────────────────────────────────────────────
    def feat_color(row):
        if abs(row["delta_sp"]) < ZERO_THRESH:
            return "#CCCCCC"           # zeroed
        return GROUP_COLORS.get(row["group"], "#AAAAAA")

    def lollipop_color(delta):
        if abs(delta) < ZERO_THRESH:
            return "#CCCCCC"
        return "#59A14F" if delta > 0 else "#E15759"

    # ── Extract ElasticNet coefficients ───────────────────────────────────────
    coef_series = None
    if model is not None and all_feats is not None:
        est = model.named_steps["est"]
        ie  = est.best_estimator_ if isinstance(est, GridSearchCV) else est
        if isinstance(ie, ElasticNet) and hasattr(ie, "coef_"):
            coef_series = pd.Series(ie.coef_, index=all_feats)
            coef_series = coef_series[coef_series != 0].sort_values()

    # ── Layout ────────────────────────────────────────────────────────────────
    n_feat = len(feat_df)
    fig    = plt.figure(figsize=(20, max(8, n_feat * 0.38 + 2)))
    # Widths: feature lollipop (40%), coefficients (28%), group dumbbell (32%)
    gs = fig.add_gridspec(1, 3, wspace=0.38,
                          width_ratios=[2.2, 1.5, 1.8])

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — Feature lollipop
    # ══════════════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0])
    df   = feat_df.sort_values("delta_sp", ascending=True)   # most impactful at top
    y    = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        c     = lollipop_color(row["delta_sp"])
        alpha = 0.35 if abs(row["delta_sp"]) < ZERO_THRESH else 0.95
        # Stem line
        ax_a.plot([0, row["delta_sp"]], [i, i], color=c, lw=1.5, alpha=alpha)
        # Dot
        ax_a.scatter(row["delta_sp"], i, color=c, s=80, zorder=4,
                     alpha=alpha, edgecolors="white", linewidths=0.5)
        # Value annotation for non-zero
        if abs(row["delta_sp"]) >= 0.003:
            xoff = 0.0005 if row["delta_sp"] >= 0 else -0.0005
            ha   = "left" if row["delta_sp"] >= 0 else "right"
            ax_a.text(row["delta_sp"] + xoff, i, f"{row['delta_sp']:+.3f}",
                      va="center", ha=ha, fontsize=8, fontweight="bold",
                      color=lollipop_color(row["delta_sp"]))

    ax_a.axvline(0, color="#444", lw=1.0, ls="--", alpha=0.7)

    # Y-tick labels — group-coloured for active features, gray for zeroed
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(df["feature"], fontsize=8.5)
    for tick, (_, row) in zip(ax_a.get_yticklabels(), df.iterrows()):
        if abs(row["delta_sp"]) < ZERO_THRESH:
            tick.set_color("#AAAAAA")
            tick.set_style("italic")
        else:
            tick.set_color(GROUP_COLORS.get(row["group"], "#333"))
            tick.set_fontweight("bold")

    # "L1 zeroed" band annotation
    zero_rows = df[df["delta_sp"].abs() < ZERO_THRESH]
    if len(zero_rows):
        y_min = y[df.index.get_loc(zero_rows.index[0])] - 0.5
        y_max = y[df.index.get_loc(zero_rows.index[-1])] + 0.5
        ax_a.axhspan(y_min, y_max, color="#F5F5F5", zorder=0, alpha=0.8)
        ax_a.text(ax_a.get_xlim()[0] if ax_a.get_xlim()[0] < 0 else -0.001,
                  (y_min + y_max) / 2, "  L1 zeroed\n  (coef = 0)",
                  va="center", ha="right", fontsize=7.5, color="#AAAAAA",
                  style="italic")

    ax_a.set_xlabel("Δ Spearman ρ  (removing this feature changes score by …)",
                    fontsize=10)
    ax_a.set_title(
        f"Leave-One-Feature-Out\n"
        f"Baseline ρ = {base['spearman']:+.3f}  (α=0.1, l₁=0.5, fixed)",
        fontsize=10, fontweight="bold",
    )

    # Group colour legend (active features only)
    active_groups = df[df["delta_sp"].abs() >= ZERO_THRESH]["group"].unique()
    patches = [mpatches.Patch(color=GROUP_COLORS.get(g, "#AAA"), label=g)
               for g in active_groups]
    patches += [mpatches.Patch(color="#CCCCCC", label="L1 zeroed (Δ=0)")]
    ax_a.legend(handles=patches, fontsize=7.5, loc="lower right",
                title="Feature group", title_fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B — Signed ElasticNet coefficients
    # ══════════════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[1])

    if coef_series is not None and len(coef_series):
        # Build a lookup for group
        feat_to_grp = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}
        coef_df = pd.DataFrame({
            "feature": coef_series.index,
            "coef":    coef_series.values,
            "group":   [feat_to_grp.get(f, "other") for f in coef_series.index],
        }).sort_values("coef")   # ascending so negative at bottom

        yc = np.arange(len(coef_df))
        for i, (_, row) in enumerate(coef_df.iterrows()):
            c = GROUP_COLORS.get(row["group"], "#AAA")
            ax_b.plot([0, row["coef"]], [i, i], color=c, lw=2.0, alpha=0.9)
            ax_b.scatter(row["coef"], i, color=c, s=90, zorder=4,
                         edgecolors="white", linewidths=0.5)
            ha  = "left" if row["coef"] >= 0 else "right"
            off = 0.0005 if row["coef"] >= 0 else -0.0005
            ax_b.text(row["coef"] + off, i, f"  {row['coef']:+.3f}",
                      va="center", ha=ha, fontsize=8, fontweight="bold",
                      color=GROUP_COLORS.get(row["group"], "#333"))

        ax_b.axvline(0, color="#444", lw=1.0, ls="--", alpha=0.7)
        ax_b.set_yticks(yc)
        ax_b.set_yticklabels(coef_df["feature"], fontsize=8.5)
        for tick, (_, row) in zip(ax_b.get_yticklabels(), coef_df.iterrows()):
            tick.set_color(GROUP_COLORS.get(row["group"], "#333"))
            tick.set_fontweight("bold")

        # Shade sides
        xl = ax_b.get_xlim()
        ax_b.axvspan(xl[0], 0, alpha=0.04, color="#E15759", zorder=0)
        ax_b.axvspan(0, xl[1], alpha=0.04, color="#4E79A7", zorder=0)
        ax_b.text(xl[0] + 0.001, len(coef_df) - 0.3, "← more pathogenic",
                  fontsize=7, color="#E15759", style="italic")
        ax_b.text(xl[1] - 0.001, len(coef_df) - 0.3, "more functional →",
                  fontsize=7, color="#4E79A7", style="italic", ha="right")
    else:
        ax_b.text(0.5, 0.5, "Coefficient data\nnot available",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=11, color="#999")

    ax_b.set_xlabel("Coefficient  (non-zero only)", fontsize=10)
    ax_b.set_title("ElasticNet Coefficients\n(L1 sparsity — most features zeroed)",
                   fontsize=10, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel C — Group dumbbell
    # ══════════════════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[2])
    gdf  = group_df.sort_values("delta_sp", ascending=True)   # most impactful at top
    yg   = np.arange(len(gdf))
    bl   = base["spearman"]

    for i, (_, row) in enumerate(gdf.iterrows()):
        c     = GROUP_COLORS.get(row["group"], "#AAA")
        sp_wo = row["sp_without"]
        # Connecting line
        lw = 2.5 if abs(row["delta_sp"]) > 0.003 else 1.2
        ax_c.plot([bl, sp_wo], [i, i], color=c, lw=lw, alpha=0.85, zorder=2)
        # Baseline dot (filled)
        ax_c.scatter(bl, i, color=c, s=120, zorder=4,
                     edgecolors="white", linewidths=0.8)
        # Without-group dot (open diamond)
        ax_c.scatter(sp_wo, i, color="white", s=120, zorder=4,
                     marker="D", edgecolors=c, linewidths=2.0)
        # Δ annotation
        mid_x = (bl + sp_wo) / 2
        ax_c.text(mid_x, i + 0.28, f"Δ={row['delta_sp']:+.3f}",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                  color=("#59A14F" if row["delta_sp"] > 0 else "#E15759"))

    # Baseline reference line
    ax_c.axvline(bl, color="#444", lw=1.5, ls="--", alpha=0.7,
                 label=f"Baseline ρ = {bl:+.3f}")

    ax_c.set_yticks(yg)
    ax_c.set_yticklabels(gdf["group"], fontsize=10)
    for tick, (_, row) in zip(ax_c.get_yticklabels(), gdf.iterrows()):
        tick.set_color(GROUP_COLORS.get(row["group"], "#333"))
        tick.set_fontweight("bold")

    # n_features annotation
    for i, (_, row) in enumerate(gdf.iterrows()):
        ax_c.text(ax_c.get_xlim()[1] if ax_c.get_xlim()[1] > 0.69 else 0.69,
                  i, f"  n={row['n_features']}",
                  va="center", fontsize=8, color="#888")

    ax_c.set_xlabel("Spearman ρ", fontsize=10)
    ax_c.set_title("Leave-One-Group-Out\n(● baseline  ◆ without group)",
                   fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8, loc="lower right")

    # ── Shared suptitle ───────────────────────────────────────────────────────
    fig.suptitle(
        "BRCA1 MAVE Prediction — ElasticNet Feature Ablation\n"
        "Positive Δ = feature contributes  ·  Δ = 0 = L1-zeroed  ·  Negative Δ = noise",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, outdir / "fig_07_ablations.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 12b — CV SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════

def plot_cv_summary(cv_res: dict, train_df: pd.DataFrame, outdir: Path):
    """
    Fig 09: Single slide-ready figure capturing all cross-validation metrics.

    Layout: two panels side-by-side (wide 16:9 format).

    Left panel (OOF scatter, ~58% width)
    ├── Diagonal y=x reference line (perfect prediction)
    ├── OLS regression line + shaded 95 % CI band
    ├── Points coloured by true MAVE score (red=pathogenic, blue=benign)
    ├── Marginal density ticks (rug) along both axes
    └── All four metrics annotated in a clean box:
        Spearman ρ, Pearson r, R², RMSE  (mean ± std over 25 folds)

    Right panel (per-fold distribution, ~42% width)
    ├── Violin + strip chart for each of the four metrics across all 25 folds
    ├── Median line inside each violin
    ├── Mean ± std annotated above each violin
    └── Horizontal reference lines at the mean for Spearman and R²
    """
    y_true    = train_df["label"].values
    y_pred    = cv_res["y_pred_oof"]
    cv_df     = pd.DataFrame({
        "Spearman ρ": cv_res["spearman"],
        "Pearson r":  cv_res["pearson"],
        "R²":         cv_res["r2"],
        "RMSE":       cv_res["rmse"],
    })
    n = len(y_true)
    lo = min(y_true.min(), np.nanmin(y_pred)) - 0.05
    hi = max(y_true.max(), np.nanmax(y_pred)) + 0.05

    # Mean metrics over folds
    sp_m, sp_s = cv_res["spearman"].mean(), cv_res["spearman"].std()
    pe_m, pe_s = cv_res["pearson"].mean(),  cv_res["pearson"].std()
    r2_m, r2_s = cv_res["r2"].mean(),       cv_res["r2"].std()
    rm_m, rm_s = cv_res["rmse"].mean(),     cv_res["rmse"].std()

    fig = plt.figure(figsize=(16, 6.5))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1.35, 1.0])

    # ══════════════════════════════════════════════════════════════════════════
    # Left panel — OOF scatter
    # ══════════════════════════════════════════════════════════════════════════
    ax_s = fig.add_subplot(gs[0])

    # Colour each point by its true MAVE score (diverging: red=low/pathogenic,
    # blue=high/functional).  Normalise to [0,1] across the training range.
    norm   = plt.Normalize(vmin=y_true.min(), vmax=y_true.max())
    colors = plt.cm.RdBu(norm(y_true))      # RdBu: red=low, blue=high

    sc = ax_s.scatter(y_true, y_pred, c=y_true, cmap="RdBu",
                      norm=norm, s=55, alpha=0.80,
                      edgecolors="white", linewidths=0.4, zorder=3)

    # Diagonal (perfect prediction)
    ax_s.plot([lo, hi], [lo, hi], color="#888", lw=1.2,
              ls="--", alpha=0.55, label="Perfect prediction (y = x)", zorder=2)

    # OLS regression line + 95 % CI band
    from numpy.polynomial.polynomial import polyfit
    coeff  = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(lo, hi, 200)
    y_line = np.polyval(coeff, x_line)

    # Bootstrap CI for the regression line
    rng = np.random.default_rng(42)
    boot_lines = []
    for _ in range(500):
        idx = rng.integers(0, n, size=n)
        c   = np.polyfit(y_true[idx], y_pred[idx], 1)
        boot_lines.append(np.polyval(c, x_line))
    boot_arr = np.array(boot_lines)
    ci_lo    = np.percentile(boot_arr, 2.5,  axis=0)
    ci_hi    = np.percentile(boot_arr, 97.5, axis=0)

    ax_s.plot(x_line, y_line, color="#333", lw=2.0,
              label="OLS regression", zorder=4)
    ax_s.fill_between(x_line, ci_lo, ci_hi,
                      color="#333", alpha=0.12,
                      label="95 % bootstrap CI", zorder=1)

    # Rug plots
    rug_kw = dict(alpha=0.25, linewidth=0.8, length=0.025)
    for val, col in zip(y_true, colors):
        ax_s.axvline(val, ymin=0, ymax=rug_kw["length"],
                     color=col, lw=rug_kw["linewidth"],
                     alpha=rug_kw["alpha"], zorder=0)
        ax_s.axhline(y_pred[np.where(y_true == val)[0][0]],
                     xmin=0, xmax=rug_kw["length"],
                     color=col, lw=rug_kw["linewidth"],
                     alpha=rug_kw["alpha"], zorder=0)

    # Colourbar
    cbar = plt.colorbar(sc, ax=ax_s, shrink=0.75, pad=0.02)
    cbar.set_label("True MAVE score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Metrics box
    metrics_text = (
        f"Spearman ρ = {sp_m:+.3f} ± {sp_s:.3f}\n"
        f"Pearson r   = {pe_m:+.3f} ± {pe_s:.3f}\n"
        f"R²            = {r2_m:+.3f} ± {r2_s:.3f}\n"
        f"RMSE         = {rm_m:.3f} ± {rm_s:.3f}\n"
        f"n = {n} variants  ·  25 folds"
    )
    ax_s.text(0.97, 0.04, metrics_text,
              transform=ax_s.transAxes,
              fontsize=9.5, verticalalignment="bottom",
              horizontalalignment="right",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="#999", alpha=0.92))

    ax_s.set_xlim(lo, hi); ax_s.set_ylim(lo, hi)
    ax_s.set_xlabel("Actual MAVE score", fontsize=12)
    ax_s.set_ylabel("OOF Predicted MAVE score\n(mean over 5 repeats)", fontsize=12)
    ax_s.set_title("Cross-Validation: OOF Predicted vs Actual\n"
                   "(5-fold × 5-repeat, ElasticNet, n=86)",
                   fontsize=11, fontweight="bold")
    ax_s.legend(fontsize=8.5, loc="upper left")

    # ══════════════════════════════════════════════════════════════════════════
    # Right panel — per-fold metric distributions
    # ══════════════════════════════════════════════════════════════════════════
    ax_v = fig.add_subplot(gs[1])

    metric_order  = ["Spearman ρ", "Pearson r", "R²", "RMSE"]
    metric_colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"]
    metric_means  = [sp_m, pe_m, r2_m, rm_m]
    metric_stds   = [sp_s, pe_s, r2_s, rm_s]

    cv_long = cv_df.melt(var_name="Metric", value_name="Value")

    positions = np.arange(len(metric_order))
    for pos, metric, color, mean, std in zip(
        positions, metric_order, metric_colors, metric_means, metric_stds
    ):
        vals = cv_df[metric].values

        # Violin
        parts = ax_v.violinplot(vals, positions=[pos], widths=0.6,
                                showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            pc.set_edgecolor(color)

        # IQR box
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax_v.vlines(pos, q25, q75, color=color, lw=4, alpha=0.7, zorder=3)
        ax_v.scatter(pos, q50, color="white", s=50, zorder=4,
                     edgecolors=color, linewidths=2.0)

        # Individual fold points (strip)
        jitter = np.random.default_rng(42 + pos).uniform(-0.12, 0.12, size=len(vals))
        ax_v.scatter(pos + jitter, vals, color=color,
                     s=18, alpha=0.55, zorder=2)

        # Mean ± std annotation above
        y_top = max(vals) + (0.04 if metric != "RMSE" else 0.02)
        ax_v.text(pos, y_top, f"{mean:.3f}\n±{std:.3f}",
                  ha="center", va="bottom", fontsize=8.5,
                  fontweight="bold", color=color)

    ax_v.set_xticks(positions)
    ax_v.set_xticklabels(metric_order, fontsize=11)
    ax_v.set_ylabel("Value per fold  (25 folds)", fontsize=11)
    ax_v.set_title("Per-Fold Metric Distributions\n"
                   "(violin + IQR box + individual folds)",
                   fontsize=11, fontweight="bold")

    # Reference lines at mean for Spearman and R²
    ax_v.axhline(sp_m, color="#4E79A7", lw=0.8, ls=":", alpha=0.5)
    ax_v.axhline(r2_m, color="#F28E2B", lw=0.8, ls=":", alpha=0.5)

    # ── Shared title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "BRCA1 MAVE Score Prediction — Cross-Validation Performance\n"
        "ElasticNet  ·  avg(cisplatin, HDR) label  ·  n = 86 training variants",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, outdir / "fig_09_cv_summary.png")


# ═══════════════════════════════════════════════════════════════════
# SECTION 13 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def run_shap_analysis(model: Pipeline,
                      train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      exp_df: pd.DataFrame,
                      all_feats: list,
                      outdir: Path) -> dict:
    """
    Compute SHAP values for the fitted ElasticNet using LinearExplainer.

    shap.LinearExplainer gives exact, closed-form SHAP values for linear models:
        φᵢ(x) = coefᵢ × (xᵢ − E[Xᵢ])
    where xᵢ is the standardised feature value and E[Xᵢ] is the training mean.
    This is both fast and exact (no sampling approximation needed).

    Preprocessing (imputation + standardisation) is applied through the pipeline
    before passing data to the explainer so that SHAP values are in units of the
    raw (unstandardised) feature space.

    Returns a dict with:
        shap_train   — (n_train, n_feats) SHAP matrix on training set
        shap_test    — (n_test,  n_feats) SHAP matrix on test set
        shap_explore — (n_exp,   n_feats) SHAP matrix on explore set
        expected_value — model's expected prediction (baseline)
        feature_names  — list of feature names
        shap_df_train  — DataFrame: variant × feature SHAP values (training)
        shap_df_test   — DataFrame: variant × feature SHAP values (test)
        shap_df_explore— DataFrame: variant × feature SHAP values (explore)
        mean_abs_df    — DataFrame: mean |SHAP| per feature, sorted
    """
    _banner("SECTION 13 — SHAP Analysis")

    # ── Extract preprocessing pipeline and linear model ───────────────────────
    imp   = model.named_steps["imp"]
    scl   = model.named_steps["scl"]
    est   = model.named_steps["est"]
    ie    = est.best_estimator_ if isinstance(est, GridSearchCV) else est

    if not isinstance(ie, ElasticNet):
        print("  SHAP LinearExplainer requires ElasticNet — skipping.")
        return {}

    # ── Transform all sets through imputer + scaler ───────────────────────────
    def preprocess(df):
        return scl.transform(imp.transform(df[all_feats]))

    X_tr_sc  = preprocess(train_df)
    X_te_sc  = preprocess(test_df[test_df["clinvar_label"] != "Highlight"])
    X_exp_sc = preprocess(exp_df)

    te_variants  = test_df[test_df["clinvar_label"] != "Highlight"]["variant"].values
    exp_variants = exp_df["variant"].values

    # ── Fit LinearExplainer on training background ─────────────────────────────
    print(f"  Fitting LinearExplainer on {len(X_tr_sc)} training samples …",
          end=" ", flush=True)
    explainer = shap.LinearExplainer(ie, X_tr_sc,
                                      feature_perturbation="correlation_dependent")
    print("done")

    shap_train   = explainer.shap_values(X_tr_sc)
    shap_test    = explainer.shap_values(X_te_sc)
    shap_explore = explainer.shap_values(X_exp_sc)

    ev = float(explainer.expected_value)
    print(f"  Expected value (model intercept + mean): {ev:.4f}")
    print(f"  SHAP arrays: train {shap_train.shape}, "
          f"test {shap_test.shape}, explore {shap_explore.shape}")

    # ── Mean |SHAP| per feature ───────────────────────────────────────────────
    mean_abs = np.abs(shap_train).mean(axis=0)
    mean_abs_df = (
        pd.DataFrame({"feature": all_feats, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    # Add group and coef
    feat_to_grp = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}
    mean_abs_df["group"] = mean_abs_df["feature"].map(
        lambda f: feat_to_grp.get(f, "other"))
    mean_abs_df["coef"]  = [ie.coef_[all_feats.index(f)]
                             for f in mean_abs_df["feature"]]

    print("\n  Mean |SHAP| per feature (top 10):")
    print(mean_abs_df.head(10)[["feature","group","mean_abs_shap","coef"]]
          .to_string(index=False, float_format="{:.4f}".format))

    # ── DataFrames of per-variant SHAP values ─────────────────────────────────
    shap_df_train = pd.DataFrame(
        shap_train, columns=all_feats,
        index=train_df["variant"].values,
    )
    shap_df_test = pd.DataFrame(
        shap_test, columns=all_feats,
        index=te_variants,
    )
    shap_df_explore = pd.DataFrame(
        shap_explore, columns=all_feats,
        index=exp_variants,
    )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    mean_abs_df.to_csv(outdir / "shap_mean_abs.csv", index=False)
    shap_df_train.to_csv(outdir / "shap_train.csv")
    shap_df_test.to_csv(outdir / "shap_test.csv")
    shap_df_explore.to_csv(outdir / "shap_explore.csv")
    print(f"\n  Saved → shap_mean_abs.csv, shap_train.csv, "
          f"shap_test.csv, shap_explore.csv")

    return {
        "shap_train":    shap_train,
        "shap_test":     shap_test,
        "shap_explore":  shap_explore,
        "expected_value":ev,
        "feature_names": all_feats,
        "shap_df_train": shap_df_train,
        "shap_df_test":  shap_df_test,
        "shap_df_explore": shap_df_explore,
        "mean_abs_df":   mean_abs_df,
        "X_tr_sc":       X_tr_sc,
        "X_te_sc":       X_te_sc,
        "X_exp_sc":      X_exp_sc,
        "te_variants":   te_variants,
        "exp_variants":  exp_variants,
    }


def plot_shap(shap_res: dict, train_df: pd.DataFrame,
              test_df: pd.DataFrame, exp_df: pd.DataFrame,
              outdir: Path, all_feats: list = None):
    """
    Fig 08: Four-panel SHAP figure.

    A (top-left)  — Mean |SHAP| bar chart: global feature importance ranked by
                    average impact across ALL variants (train + test + explore).
                    Coloured by feature group.

    B (top-right) — SHAP beeswarm (summary plot): each dot is one training variant,
                    x = SHAP value, y = feature (sorted by mean |SHAP|), colour =
                    feature value (blue=low, red=high). Shows direction of effect.

    C (bottom-left) — SHAP heatmap: training variants × features, SHAP values as
                       a colour grid. Variants sorted by predicted score (left=most
                       pathogenic). Reveals which features drive each prediction.

    D (bottom-right) — SHAP waterfall: mean prediction broken down by feature
                        contribution. Shows how the model arrives at a typical
                        pathogenic vs. benign prediction.
    """
    if not shap_res:
        print("  SHAP results not available — skipping figure.")
        return

    fnames      = all_feats if all_feats is not None else shap_res["feature_names"]
    shap_tr     = shap_res["shap_train"]
    X_tr_sc     = shap_res["X_tr_sc"]
    mean_abs_df = shap_res["mean_abs_df"]
    ev          = shap_res["expected_value"]

    # Pool SHAP across train + test + explore for global importance
    shap_all = np.vstack([shap_tr,
                           shap_res["shap_test"],
                           shap_res["shap_explore"]])
    mean_abs_all = np.abs(shap_all).mean(axis=0)
    order = np.argsort(mean_abs_all)[::-1]   # descending importance

    feat_to_grp = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — Mean |SHAP| bar chart (global importance)
    # ══════════════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    # Top 15 features by mean |SHAP| across all variants
    top_n  = min(15, len(fnames))
    top_idx = order[:top_n][::-1]   # plot ascending so most important at top
    top_feats  = [fnames[i] for i in top_idx]
    top_vals   = mean_abs_all[top_idx]
    top_colors = [GROUP_COLORS.get(feat_to_grp.get(f, "other"), "#AAA")
                  for f in top_feats]

    bars = ax_a.barh(top_feats, top_vals, color=top_colors,
                     alpha=0.85, edgecolor="white", height=0.7)
    for bar, val in zip(bars, top_vals):
        if val > 0.001:
            ax_a.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                      f"{val:.4f}", va="center", ha="left",
                      fontsize=8, fontweight="bold")

    ax_a.set_xlabel("Mean |SHAP value|  (average impact across all variants)", fontsize=10)
    ax_a.set_title("Global Feature Importance\n(mean |SHAP|, pooled across train + test + explore)",
                   fontsize=10, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
               if g in [feat_to_grp.get(f, "other") for f in top_feats]]
    ax_a.legend(handles=patches, fontsize=8, loc="lower right")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B — Beeswarm / summary plot (training variants)
    # ══════════════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    # Top 15 features sorted by mean |SHAP| on training set
    mean_abs_tr = np.abs(shap_tr).mean(axis=0)
    order_tr    = np.argsort(mean_abs_tr)[::-1][:top_n][::-1]

    for plot_i, feat_i in enumerate(order_tr):
        shap_vals = shap_tr[:, feat_i]
        feat_vals = X_tr_sc[:, feat_i]   # standardised feature values
        # Jitter vertically
        rng = np.random.default_rng(42 + feat_i)
        jitter = rng.uniform(-0.3, 0.3, size=len(shap_vals))
        # Colour by standardised feature value (blue=low, red=high)
        norm   = plt.Normalize(vmin=feat_vals.min(), vmax=feat_vals.max())
        colors = plt.cm.RdBu_r(norm(feat_vals))
        ax_b.scatter(shap_vals, plot_i + jitter,
                     c=colors, s=18, alpha=0.7, linewidths=0,
                     rasterized=True, zorder=3)

    ax_b.axvline(0, color="#444", lw=1.0, ls="--", alpha=0.6)
    ax_b.set_yticks(range(top_n))
    ax_b.set_yticklabels([fnames[i] for i in order_tr], fontsize=8.5)
    for tick, fi in zip(ax_b.get_yticklabels(), order_tr):
        tick.set_color(GROUP_COLORS.get(feat_to_grp.get(fnames[fi], "other"), "#333"))

    # Colorbar for feature value
    sm = plt.cm.ScalarMappable(cmap="RdBu_r")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_b, shrink=0.5, pad=0.02)
    cbar.set_label("Feature value\n(standardised)", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=8)

    ax_b.set_xlabel("SHAP value  (impact on predicted MAVE score)", fontsize=10)
    ax_b.set_title("SHAP Beeswarm — Training Variants\n"
                   "(each dot = 1 variant; colour = feature value)",
                   fontsize=10, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel C — SHAP heatmap across all variants (train + test + explore)
    # ══════════════════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    # Top 10 features × all variants, sorted by predicted score
    top10_idx   = order[:10]
    top10_feats = [fnames[i] for i in top10_idx]
    shap_heat   = shap_all[:, top10_idx]   # (n_all, 10)

    # Sort variants by sum of SHAP (most pathogenic left)
    row_sort = np.argsort(shap_heat.sum(axis=1))[::-1]

    im = ax_c.imshow(shap_heat[row_sort].T, aspect="auto",
                     cmap="RdBu_r", interpolation="nearest",
                     vmin=-np.abs(shap_heat).max(), vmax=np.abs(shap_heat).max())
    ax_c.set_yticks(range(10))
    ax_c.set_yticklabels(top10_feats, fontsize=8.5)
    for tick, fi in zip(ax_c.get_yticklabels(), top10_idx):
        tick.set_color(GROUP_COLORS.get(feat_to_grp.get(fnames[fi], "other"), "#333"))
    ax_c.set_xlabel("Variants (sorted by Σ SHAP, most pathogenic → left)", fontsize=9)
    ax_c.set_xticks([])
    ax_c.set_title("SHAP Heatmap — Top 10 Features\n"
                   "(all variants; red = pushes toward pathogenic)",
                   fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax_c, shrink=0.6, pad=0.02,
                 label="SHAP value")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel D — Group-level SHAP: mean |SHAP| per group, stacked for each set
    # ══════════════════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    sets = {
        "Training (n={})".format(len(shap_tr)):         shap_tr,
        "ClinVar test (n={})".format(len(shap_res["shap_test"])): shap_res["shap_test"],
        "Explore (n={})".format(len(shap_res["shap_explore"])): shap_res["shap_explore"],
    }
    groups_ordered = list(FEATURE_GROUPS.keys())
    feat_grp_idx   = {g: [all_feats.index(f) for f in cols if f in all_feats]
                      for g, cols in FEATURE_GROUPS.items()}

    x     = np.arange(len(groups_ordered))
    width = 0.25
    for j, (set_name, sv) in enumerate(sets.items()):
        vals = [np.abs(sv[:, feat_grp_idx[g]]).mean() if feat_grp_idx[g] else 0
                for g in groups_ordered]
        offset = (j - 1) * width
        bars_d = ax_d.bar(x + offset, vals, width,
                          color=[GROUP_COLORS.get(g, "#AAA") for g in groups_ordered],
                          alpha=0.6 + j * 0.15, edgecolor="white",
                          label=set_name)
        for bar, val in zip(bars_d, vals):
            if val > 0.0005:
                ax_d.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.0001,
                          f"{val:.3f}", ha="center", va="bottom",
                          fontsize=7.5, rotation=45)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(groups_ordered, rotation=20, ha="right", fontsize=10)
    ax_d.set_ylabel("Mean |SHAP value| per group", fontsize=10)
    ax_d.set_title("Group-Level SHAP Importance\n"
                   "by Dataset Split",
                   fontsize=10, fontweight="bold")
    ax_d.legend(fontsize=8)

    fig.suptitle(
        "SHAP Analysis — BRCA1 MAVE Prediction (ElasticNet)\n"
        "LinearExplainer: exact Shapley values for linear models",
        fontsize=13, fontweight="bold",
    )
    _save(fig, outdir / "fig_08_shap.png")


# ═══════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="BRCA1 MAVE Final Pipeline — ElasticNet + RF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _here = Path(__file__).parent
    p.add_argument("--feats",       default=str(_here / "data" / "brca1_final_feats.csv"))
    p.add_argument("--mave",        default=str(_here / "data" / "brca1_mave.csv"))
    p.add_argument("--cv-test",     default=str(_here / "data" / "clinvar_test.csv"),    dest="cv_test")
    p.add_argument("--cv-exp",      default=str(_here / "data" / "clinvar_explore.csv"), dest="cv_exp")
    p.add_argument("--outdir",      default=str(_here / "results"))
    p.add_argument("--model",       default="elasticnet",
                   choices=["elasticnet", "rf"],
                   help="Model architecture (elasticnet: full features; rf: pruned features)")
    p.add_argument("--label",       default="both",
                   choices=["both", "any", "cisplatin", "hdr"],
                   help="Label strategy for the regression target")
    p.add_argument("--n-folds",     type=int, default=5, dest="n_folds")
    p.add_argument("--n-repeats",   type=int, default=5, dest="n_repeats")
    p.add_argument("--n-bootstrap", type=int, default=200, dest="n_bootstrap")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no-tune",     dest="tune", action="store_false", default=True,
                   help="Disable inner-CV hyperparameter tuning (faster)")
    p.add_argument("--split",       default="extended",
                   choices=["strict", "extended"],
                   help=("strict: exclude all 48 test + 24 explore from training, "
                         "evaluate on 48. "
                         "extended: put CLINVAR_ORIGINAL_20 back in training, "
                         "evaluate on the clean 28-variant holdout."))
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_label = ("ElasticNet (SimpleImputer → StandardScaler → ElasticNet)"
                   if args.model == "elasticnet"
                   else "RF (KNNImputer → RandomForest)")

    print("═" * 62)
    print("  BRCA1 MAVE Pathogenicity Final Pipeline")
    print(f"  {model_label}")
    print("═" * 62)
    print(f"  model       : {args.model}")
    print(f"  label       : {args.label}")
    print(f"  split       : {args.split}")
    print(f"  CV          : {args.n_folds}-fold × {args.n_repeats}-repeat")
    print(f"  tune        : {args.tune}")
    print(f"  n_bootstrap : {args.n_bootstrap}")
    print(f"  seed        : {args.seed}")

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    merged, cv_test_meta, cv_explore_meta = load_and_merge(
        args.feats, args.mave, args.cv_test, args.cv_exp
    )

    # ── 2. Feature engineering ────────────────────────────────────────────────
    # ElasticNet: full FEATURE_GROUPS (L1 handles irrelevant features).
    # RF: ablation-pruned FEATURE_GROUPS_RF (avoids overfitting at n=70).
    if args.model == "rf":
        global FEATURE_GROUPS
        FEATURE_GROUPS = FEATURE_GROUPS_RF
    merged, all_feats = engineer_features(merged)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    train_df, test_df, exp_df = split_data(
        merged, cv_test_meta, cv_explore_meta, all_feats, args.label,
        split_mode=args.split,
    )

    X_train = train_df[all_feats]
    y_train = train_df["label"]
    w_train = compute_sample_weights(train_df)

    threshold = float(y_train.median())
    print(f"\n  Classification threshold (training median): {threshold:.4f}")

    # ── 4. Build model ────────────────────────────────────────────────────────
    if args.model == "elasticnet":
        pipe = build_elasticnet_model(seed=args.seed, tune=args.tune)
    else:
        pipe = build_model(seed=args.seed, tune=args.tune)

    # ── 5. Cross-validated evaluation ─────────────────────────────────────────
    _banner("SECTION 5/6 — Repeated K-Fold CV")
    print(f"  {args.n_folds}-fold × {args.n_repeats}-repeat  "
          f"(n={len(train_df)}, {len(all_feats)} features)")

    cv_res = evaluate_cv(
        pipe, X_train, y_train, w_train,
        n_folds=args.n_folds, n_repeats=args.n_repeats, seed=args.seed,
    )
    spear = float(np.nanmean(cv_res["spearman"]))
    r2    = float(np.nanmean(cv_res["r2"]))
    rmse  = float(np.nanmean(cv_res["rmse"]))
    print(f"\n  ── CV Results (mean ± std over {args.n_folds*args.n_repeats} folds) ──")
    print(f"  Spearman ρ : {spear:+.4f} ± {np.nanstd(cv_res['spearman']):.4f}")
    print(f"  Pearson r  : {np.nanmean(cv_res['pearson']):+.4f} ± {np.nanstd(cv_res['pearson']):.4f}")
    print(f"  R²         : {r2:+.4f} ± {np.nanstd(cv_res['r2']):.4f}")
    print(f"  RMSE       : {rmse:.4f} ± {np.nanstd(cv_res['rmse']):.4f}")
    print(f"  AUROC      : {cv_res['auroc']:.4f}  (OOF median-split binarisation)")

    # ── 6. Final model ────────────────────────────────────────────────────────
    final_model = train_final_model(pipe, X_train, y_train, w_train)

    # ── 7. ClinVar test ───────────────────────────────────────────────────────
    test_res = evaluate_clinvar_test(final_model, test_df, all_feats, threshold)

    # ── 8. ClinVar explore ────────────────────────────────────────────────────
    exp_res = predict_clinvar_explore(
        final_model, exp_df, all_feats, threshold,
        train_df, seed=args.seed, n_bootstrap=args.n_bootstrap,
    )

    # ── 9. Comparison ─────────────────────────────────────────────────────────
    comp_df = build_comparison(cv_res, test_res, args.label)

    # ── 10. Save CSVs ─────────────────────────────────────────────────────────
    _banner("Saving Outputs")

    cv_out = pd.DataFrame({
        "fold":     range(1, len(cv_res["r2"]) + 1),
        "r2":       cv_res["r2"],
        "rmse":     cv_res["rmse"],
        "pearson":  cv_res["pearson"],
        "spearman": cv_res["spearman"],
    })
    cv_out.to_csv(outdir / "cv_metrics.csv", index=False)
    print(f"  cv_metrics.csv                  ({len(cv_out)} folds)")

    # Save OOF predictions for the CV summary figure
    oof_out = pd.DataFrame({
        "variant":     train_df["variant"].values,
        "y_true":      y_train.values,
        "y_pred_oof":  cv_res["y_pred_oof"],
        "residual":    y_train.values - cv_res["y_pred_oof"],
    })
    oof_out.to_csv(outdir / "cv_oof_predictions.csv", index=False)
    print(f"  cv_oof_predictions.csv          ({len(oof_out)} variants)")

    test_res["result_df"].to_csv(
        outdir / "clinvar_test_predictions.csv", index=False)
    print(f"  clinvar_test_predictions.csv    ({len(test_res['result_df'])} variants)")

    exp_res.to_csv(outdir / "clinvar_explore_predictions.csv", index=False)
    print(f"  clinvar_explore_predictions.csv ({len(exp_res)} variants)")

    comp_df.to_csv(outdir / "metrics_comparison.csv", index=False)
    print(f"  metrics_comparison.csv")

    # ── 11. Ablation studies ─────────────────────────────────────────────────
    # Extract best ElasticNet params from the fitted model for fast ablation.
    est_step  = final_model.named_steps["est"]
    inner_est = est_step.best_estimator_ if isinstance(est_step, GridSearchCV) else est_step
    if isinstance(inner_est, ElasticNet):
        best_params = {"alpha": inner_est.alpha, "l1_ratio": inner_est.l1_ratio}
    else:  # RF fallback
        best_params = {}

    feat_abl_df, grp_abl_df, abl_base = run_ablations(
        train_df, all_feats, best_params,
        n_folds=args.n_folds, n_repeats=args.n_repeats,
        seed=args.seed, outdir=outdir,
    )

    # ── 12. SHAP analysis ─────────────────────────────────────────────────────
    shap_res = run_shap_analysis(
        final_model, train_df, test_df, exp_df, all_feats, outdir
    )

    # ── 13. Figures ───────────────────────────────────────────────────────────
    _banner("Generating Figures")
    plot_cv_scatter(cv_res, train_df, outdir)
    plot_cv_summary(cv_res, train_df, outdir)
    plot_feature_importance(final_model, all_feats, outdir)
    plot_clinvar_test(test_res, outdir)
    plot_known_variants(test_res, outdir)
    plot_explore_ranked(exp_res, threshold, outdir)
    plot_summary_slide(cv_res, test_res, exp_res, train_df, outdir)
    plot_ablations(feat_abl_df, grp_abl_df, abl_base, outdir,
                   model=final_model, all_feats=all_feats)
    plot_shap(shap_res, train_df, test_df, exp_df, outdir, all_feats=all_feats)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"  CV  Spearman ρ  : {spear:+.4f}")
    print(f"  CV  R²          : {r2:+.4f}")
    print(f"  CV  RMSE        : {rmse:.4f}")
    print(f"  CV  AUROC       : {cv_res['auroc']:.4f}")
    print(f"  Test AUROC      : {test_res['auroc']:.4f}")
    n_test = len(test_res["result_df"])
    print(f"  Test Accuracy   : {int(round(test_res['accuracy']*n_test))}/{n_test}  "
          f"({test_res['accuracy']*100:.1f} %)")
    print(f"  Test Sensitivity: {test_res['sensitivity']:.4f}")
    print(f"  Test Specificity: {test_res['specificity']:.4f}")
    print(f"\n  All outputs → {outdir}/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
