#!/usr/bin/env python3
"""
BRCA1 Missense Variant Severity Prediction Pipeline  (v2)
==========================================================
Goal
----
Train ML models that estimate MaveDB functional scores for BRCA1 missense
variants, then express those estimates as calibrated severity scores with
95% confidence intervals.

What the models output
----------------------
All three models (Ridge, Random Forest, XGBoost) are regressors that predict
`overall_mave_score` — the same continuous functional score that MaveDB
reports.  A higher value means greater loss-of-function (more pathogenic).
Models do NOT natively output probabilities or severity tiers; those are
derived downstream by SeverityScaler.

  Raw MAVE prediction  →  SeverityScaler  →  severity ∈ [0, 1]
                                           →  tier: Benign / VUS / Pathogenic
                                           →  95% CI propagated linearly

The AUROC and PR-AUC metrics you see during training are computed post-hoc by
binarising predictions at the 67th-percentile threshold of the training labels
(top tertile = pathogenic).  They do not change the regression objective.

Inputs
------
  Aim 1 : brca1_features_2.csv  — AlphaFold3 structural features (new format)
             • variant string "P1579C" parsed → ref_aa / alt_aa join keys
             • 100% null columns (rsa, ss_helix, ss_strand, ss_coil) dropped
  Aim 2 : BRCA1_Final.csv       — MAVE scores + AlphaMissense (unchanged)
             • joined on [uniprot_position, ref_aa, alt_aa]

Outputs  (all written to results/)
-------
  model_summary.csv             — CV + test metrics for all three models
  test_predictions.csv          — raw MAVE predictions + 95% CIs, test set
  test_severity_scores.csv      — [0,1] severity + CIs + tiers, test set
  lr_coefficients.csv           — Ridge regression signed coefficients
  ablation_results.csv          — group-level leave-one-out + forward ablation
  ablation_per_feature.csv      — leave-one-feature-out (all individual cols)
  shap_group_contributions.csv  — mean |SHAP| per feature group × model
  ci_comparison.png             — raw MAVE predictions + CIs, all models
  severity_scores.png           — severity waterfall with CIs + tier colouring
  ablation_results.png          — group-level AUROC heatmap + uncertainty bar
  ablation_per_feature.png      — Δ AUROC per individual feature, ranked
  shap_summary_random_forest.png
  shap_summary_xgboost.png
  shap_group_contributions.png
  variance_by_coverage.png

Pipeline stages
---------------
  1.  Data loading & merging
  2.  Preprocessing, median imputation, confidence-weight construction
  3.  Stratified 80/20 train–test split
  4.  SeverityScaler calibration (fit on training labels only)
  5.  5-fold cross-validation (AUROC, PR-AUC, RMSE)
  6.  Model training + variance-aware CIs
        Ridge       → residual-variance prediction intervals
        RandomForest → across-tree variance
        XGBoost     → bootstrap variance
  7.  Ablation studies
        (a) leave-one-group-out   → group importance
        (b) add-one-group-in      → cumulative gain
        (c) leave-one-feature-out → per-column importance  ← NEW
  8.  SHAP feature importance
  9.  Results export + all plots
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, mean_squared_error,
)

import xgboost as xgb
import shap
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED         = 42
N_FOLDS             = 5
TEST_SIZE           = 0.20
N_BOOTSTRAP         = 100   # XGBoost bootstrap: final models
N_BOOTSTRAP_ABL     = 30    # XGBoost bootstrap: group-level ablation
N_BOOTSTRAP_FEAT    = 20    # XGBoost bootstrap: per-feature ablation (lighter)
CONFIDENCE_LEVEL    = 0.95
CONFORMAL_CALIBRATION_FRACTION = 0.2

# Severity tier boundaries (applied to [0, 1] severity scores)
TIER_BENIGN     = 0.33      # severity < 0.33 → Benign / Likely Benign
TIER_PATHOGENIC = 0.67      # severity ≥ 0.67 → Likely Pathogenic / Pathogenic
                             # 0.33–0.67 → VUS

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Feature groups ────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "alphamissense": [
        "alphamissense_score",
    ],
    "global_structural": [
        "ca_rmsd", "backbone_rmsd", "plddt",
    ],
    "local_structural": [
        "local_displacement",
        "shell_rmsd_5", "shell_rmsd_8", "shell_rmsd_12",
        "ramachandran_clash",
    ],
    "biochemical": [
        "pam_score", "delta_size", "delta_hydrophobicity",
        "delta_charge", "delta_aromaticity",
        "is_charge_reversal", "is_size_increase",
        "is_hydrophobic_to_polar", "is_polar_to_hydrophobic",
    ],
    "structural_energy": [
        "evoef2_ddg_total", "ddg_evoef2",
    ],
    "residue_context": [
        "is_interface_residue", "is_disordered_variant",
    ],
}

ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]

# Reverse lookup: feature → group name (used in per-feature ablation plot)
FEATURE_TO_GROUP = {
    feat: grp
    for grp, feats in FEATURE_GROUPS.items()
    for feat in feats
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(aim1_path: str, aim2_path: str) -> pd.DataFrame:
    """
    Inner-join Aim 1 (brca1_features_2.csv) and Aim 2 (BRCA1_Final.csv) on
    [uniprot_position, ref_aa, alt_aa].

    Aim 1 new-format handling:
      • 100% null columns dropped (rsa, ss_helix, ss_strand, ss_coil).
      • variant string (e.g. "P1579C") parsed → ref_aa / alt_aa for join.
      • mutant_residue → uniprot_position for join.
    """
    aim1 = pd.read_csv(aim1_path)
    aim2 = pd.read_csv(aim2_path)

    # Drop 100% null columns
    all_null = [c for c in aim1.columns if aim1[c].isna().all()]
    if all_null:
        print(f"[Load] Dropping 100% null columns from Aim 1: {all_null}")
        aim1 = aim1.drop(columns=all_null)

    # Parse variant string → ref_aa / alt_aa
    parsed = aim1["variant"].str.extract(r"^([A-Za-z])(\d+)([A-Za-z])$")
    n_bad  = parsed.isna().any(axis=1).sum()
    if n_bad:
        print(f"[Load] ⚠  {n_bad} variant string(s) could not be parsed; "
              f"they will be excluded from the merge.")
    aim1["ref_aa"] = parsed[0].str.upper()
    aim1["alt_aa"] = parsed[2].str.upper()
    aim1 = aim1.rename(columns={"mutant_residue": "uniprot_position"})

    keys = ["uniprot_position", "ref_aa", "alt_aa"]
    df   = pd.merge(aim1, aim2, on=keys, how="inner")
    _log_merge(aim1, aim2, df)

    df = _normalize_columns(df)
    return df


def load_combined(csv_path: str) -> pd.DataFrame:
    """Legacy loader for a single pre-merged CSV with pipeline-internal names."""
    df = pd.read_csv(csv_path)
    print(f"[Data] Loaded {len(df)} variants from '{csv_path}'")
    print(f"[Data] Coverage:\n{df['coverage'].value_counts().sort_index().to_string()}\n")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw CSV column names → pipeline-internal names.

    Aim 2 (unchanged):
        am_pathogenicity  → alphamissense_score
        mave_score        → overall_mave_score

    Aim 1 (brca1_features_2.csv new format):
        mutant_plddt          → plddt
        mutant_ca_displacement→ local_displacement
        shell_rmsd_5A/8A/12A  → shell_rmsd_5/8/12
        ramachandran_violation→ ramachandran_clash
        evoef2_ddg_Total      → evoef2_ddg_total
        pam250_score          → pam_score
    """
    rename_map = {
        "am_pathogenicity":       "alphamissense_score",
        "mave_score":             "overall_mave_score",
        "mutant_plddt":           "plddt",
        "mutant_ca_displacement": "local_displacement",
        "shell_rmsd_5A":          "shell_rmsd_5",
        "shell_rmsd_8A":          "shell_rmsd_8",
        "shell_rmsd_12A":         "shell_rmsd_12",
        "ramachandran_violation": "ramachandran_clash",
        "evoef2_ddg_Total":       "evoef2_ddg_total",
        "pam250_score":           "pam_score",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing)
    print(f"[Normalize] Renamed {len(existing)} columns to pipeline-internal names.")
    return df


def _log_merge(aim1, aim2, merged):
    print(f"[Merge] Aim 1 variants : {len(aim1)}")
    print(f"[Merge] Aim 2 variants : {len(aim2)}")
    print(f"[Merge] After inner join: {len(merged)}")
    if "coverage" in merged.columns:
        print(f"[Merge] Coverage distribution:\n"
              f"{merged['coverage'].value_counts().sort_index().to_string()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Returns
    -------
    X                : feature DataFrame (available features, NaNs median-imputed)
    y                : continuous label  (overall_mave_score)
    y_binary         : binary label — top tertile = high loss-of-function (pathogenic)
    coverage         : MAVE assay coverage {1…5}
    confidence_weight: (coverage/5) × (pLDDT/100)  — pTM removed from new format

    Note: partially-null columns (shell_rmsd_* ~15% missing) are imputed with
    column median.  Fully-null columns were already dropped at load time.
    """
    df = df.copy().dropna(subset=["overall_mave_score"])

    threshold = df["overall_mave_score"].quantile(0.67)
    df["mave_binary"] = (df["overall_mave_score"] >= threshold).astype(int)

    available = [f for f in ALL_FEATURES if f in df.columns]
    missing   = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"[Preprocess] ⚠ Features not in data (skipped): {missing}")

    X = df[available].copy()

    partial_null = [c for c in X.columns if X[c].isna().any()]
    if partial_null:
        print(f"[Preprocess] Median-imputing {len(partial_null)} partially-null "
              f"column(s): {partial_null}")
        for c in partial_null:
            X[c] = X[c].fillna(X[c].median())

    y        = df["overall_mave_score"].copy()
    y_binary = df["mave_binary"].copy()
    coverage = df["coverage"].copy()

    plddt_col         = df["plddt"] if "plddt" in df.columns else pd.Series(np.ones(len(df)) * 80)
    confidence_weight = (coverage / 5.0) * (plddt_col / 100.0)
    confidence_weight = confidence_weight.clip(lower=1e-3)

    print(f"[Preprocess] Samples  : {len(X)}")
    print(f"[Preprocess] Features : {len(available)}")
    print(f"[Preprocess] Label range : [{y.min():.3f}, {y.max():.3f}]")
    print(f"[Preprocess] Pathogenic rate: {y_binary.mean():.1%}\n")

    return X, y, y_binary, coverage, confidence_weight.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X, y, y_binary, coverage, confidence_weight) -> dict:
    """Stratified 80/20 split, preserving binary class balance."""
    idx_tr, idx_te = train_test_split(
        np.arange(len(X)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_binary,
    )
    def _sel(s, idx):
        return s.iloc[idx].reset_index(drop=True)

    split = dict(
        X_train   = _sel(X, idx_tr),   X_test  = _sel(X, idx_te),
        y_train   = _sel(y, idx_tr),   y_test  = _sel(y, idx_te),
        yb_train  = _sel(y_binary, idx_tr), yb_test = _sel(y_binary, idx_te),
        cov_train = _sel(coverage, idx_tr), cov_test= _sel(coverage, idx_te),
        cw_train  = _sel(confidence_weight, idx_tr),
        cw_test   = _sel(confidence_weight, idx_te),
    )
    print(f"[Split] Train n={len(idx_tr)}, pathogenic={split['yb_train'].mean():.1%}")
    print(f"[Split] Test  n={len(idx_te)}, pathogenic={split['yb_test'].mean():.1%}\n")
    return split


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEVERITY SCALER
# ─────────────────────────────────────────────────────────────────────────────

class SeverityScaler:
    """
    Calibrates raw overall_mave_score predictions onto a [0, 1] severity scale.

    The mapping is a percentile-robust linear rescaling fit on training labels:

        severity = clip( (y − y_p5) / (y_p95 − y_p5), 0, 1 )

    where y_p5 / y_p95 are the 5th and 95th percentiles of the training
    overall_mave_score distribution.  This makes the scale interpretable and
    robust to outliers without distorting the rank ordering.

    Convention (consistent with the binary label used in training):
        0.0  ≡  WT-like function  (benign)
        1.0  ≡  complete loss-of-function  (pathogenic)

    Tiers (default boundaries TIER_BENIGN=0.33, TIER_PATHOGENIC=0.67):
        [0.00, 0.33)  →  Benign / Likely Benign
        [0.33, 0.67)  →  Variant of Uncertain Significance (VUS)
        [0.67, 1.00]  →  Likely Pathogenic / Pathogenic

    Because the transform is linear, CIs in raw MAVE space transform cleanly:
        severity_std  = pred_std_raw / (y_p95 − y_p5)
        severity_CI   = clip( (CI_bound − y_p5) / (y_p95 − y_p5), 0, 1 )

    Important: always call fit() on training labels ONLY to avoid leakage.
    """

    def __init__(
        self,
        tier_benign: float = TIER_BENIGN,
        tier_pathogenic: float = TIER_PATHOGENIC,
    ):
        self.tier_benign     = tier_benign
        self.tier_pathogenic = tier_pathogenic
        self.y_p5  = None
        self.y_p95 = None
        self._range = None

    # ------------------------------------------------------------------
    def fit(self, y_train):
        """Fit calibration percentiles from training labels."""
        y = np.asarray(y_train)
        self.y_p5   = float(np.percentile(y, 5))
        self.y_p95  = float(np.percentile(y, 95))
        self._range = max(self.y_p95 - self.y_p5, 1e-8)
        print(
            f"[SeverityScaler] Calibrated on {len(y)} training samples.\n"
            f"  y_p5  = {self.y_p5:.4f}  (maps to severity 0.0)\n"
            f"  y_p95 = {self.y_p95:.4f}  (maps to severity 1.0)\n"
            f"  Tier boundaries: Benign < {self.tier_benign} ≤ VUS "
            f"< {self.tier_pathogenic} ≤ Pathogenic\n"
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, y) -> np.ndarray:
        """Map raw MAVE scores → [0, 1] severity (vectorised)."""
        y = np.asarray(y, dtype=float)
        return np.clip((y - self.y_p5) / self._range, 0.0, 1.0)

    # ------------------------------------------------------------------
    def transform_ci(self, ci_dict: dict) -> dict:
        """
        Apply the severity transform to a full predict_with_ci() output dict.

        Returns a dict with keys:
          severity, severity_lower, severity_upper, severity_std, severity_tier
        """
        sev       = self.transform(ci_dict["y_pred"])
        sev_lower = self.transform(ci_dict["ci_lower"])
        sev_upper = self.transform(ci_dict["ci_upper"])
        sev_std   = np.asarray(ci_dict["pred_std"]) / self._range

        tiers = self._assign_tiers(sev)

        return dict(
            severity        = sev,
            severity_lower  = sev_lower,
            severity_upper  = sev_upper,
            severity_std    = sev_std,
            severity_tier   = tiers,
        )

    # ------------------------------------------------------------------
    def _assign_tiers(self, severity: np.ndarray) -> np.ndarray:
        """Return string tier labels for an array of severity scores."""
        tiers = np.where(
            severity < self.tier_benign,
            "Benign",
            np.where(severity < self.tier_pathogenic, "VUS", "Pathogenic"),
        )
        return tiers

    # ------------------------------------------------------------------
    def summary_table(self, severity: np.ndarray) -> pd.DataFrame:
        """Return a tier-count summary DataFrame."""
        tiers, counts = np.unique(self._assign_tiers(severity), return_counts=True)
        return pd.DataFrame({"tier": tiers, "count": counts}).set_index("tier")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cv(estimator, X: pd.DataFrame, y: pd.Series, y_binary: pd.Series,
                sample_weight=None) -> dict:
    """
    5-fold stratified CV.  Models are fit on continuous y; AUROC and PR-AUC
    are evaluated on binarised y_binary; RMSE on continuous y.
    Features are z-scored independently within each fold.
    """
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aurocs, aps, rmses = [], [], []

    for tr_idx, va_idx in kf.split(X, y_binary):
        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X.iloc[tr_idx])
        X_va    = scaler.transform(X.iloc[va_idx])
        y_tr    = y.iloc[tr_idx]
        y_va    = y.iloc[va_idx]
        yb_va   = y_binary.iloc[va_idx]
        sw_fold = sample_weight[tr_idx] if sample_weight is not None else None

        fit_kw = {"sample_weight": sw_fold} if sw_fold is not None else {}
        estimator.fit(X_tr, y_tr, **fit_kw)
        preds = estimator.predict(X_va)

        aurocs.append(roc_auc_score(yb_va, preds))
        aps.append(average_precision_score(yb_va, preds))
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))

    return dict(
        auroc_mean  = float(np.mean(aurocs)),
        auroc_std   = float(np.std(aurocs)),
        ap_mean     = float(np.mean(aps)),
        ap_std      = float(np.std(aps)),
        rmse_mean   = float(np.mean(rmses)),
        rmse_std    = float(np.std(rmses)),
        auroc_folds = aurocs,
        ap_folds    = aps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELS WITH VARIANCE ESTIMATION & CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionWithCI:
    """
    Ridge regression with closed-form prediction intervals from residual variance.

        PI(x̃) = ŷ ± z_{α/2} · √[ σ² · (1 + x̃ᵀ (XᵀX)⁻¹ x̃) ] / w

    where σ² is weighted residual variance and w is the confidence weight
    (lower confidence → wider interval).

    Output units: raw overall_mave_score  (MaveDB functional score scale).
    Convert to severity via SeverityScaler.transform_ci().
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha   = alpha
        self.scaler  = StandardScaler()
        self.model   = None
        self.sigma2  = None
        self.XtX_inv = None

    def fit(self, X, y, sample_weight=None):
        Xs = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=self.alpha)
        fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.model.fit(Xs, y, **fit_kw)

        residuals = np.asarray(y) - self.model.predict(Xs)
        n, p = Xs.shape
        if sample_weight is not None:
            w = np.asarray(sample_weight)
            self.sigma2 = float(np.sum(w * residuals**2) / (np.sum(w) - p - 1))
        else:
            self.sigma2 = float(np.sum(residuals**2) / (n - p - 1))

        X_aug        = np.hstack([np.ones((n, 1)), Xs])
        self.XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
        return self

    def predict_with_ci(self, X, confidence_weights=None,
                        level: float = CONFIDENCE_LEVEL) -> dict:
        Xs    = self.scaler.transform(X)
        y_hat = self.model.predict(Xs)
        z     = stats.norm.ppf((1 + level) / 2)

        X_aug    = np.hstack([np.ones((len(Xs), 1)), Xs])
        pred_var = np.array([self.sigma2 * (1.0 + xi @ self.XtX_inv @ xi)
                             for xi in X_aug])
        pred_std = np.sqrt(np.maximum(pred_var, 0.0))

        if confidence_weights is not None:
            cw       = np.clip(np.asarray(confidence_weights), 1e-3, 1.0)
            pred_std = pred_std / cw

        return dict(
            y_pred         = y_hat,
            ci_lower       = y_hat - z * pred_std,
            ci_upper       = y_hat + z * pred_std,
            pred_std       = pred_std,
            model_variance = self.sigma2,
        )

    def get_coefficients(self, feature_names: list) -> pd.Series:
        return (pd.Series(self.model.coef_, index=feature_names)
                .sort_values(key=abs, ascending=False))

    def clone_base(self):
        return Ridge(alpha=self.alpha)


# ─────────────────────────────────────────────────────────────────────────────

class RandomForestWithCI:
    """
    Random Forest with per-prediction variance from tree spread:

        Var_RF(x) = (1/B) Σ_b [ f_b(x) − f̄(x) ]²

    Output units: raw overall_mave_score.
    """

    def __init__(self, n_estimators: int = 300, **rf_kwargs):
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.model  = RandomForestRegressor(
            n_estimators=n_estimators, random_state=RANDOM_SEED,
            n_jobs=-1, **rf_kwargs,
        )

    def fit(self, X, y, sample_weight=None):
        Xs = self.scaler.fit_transform(X)
        fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.model.fit(Xs, y, **fit_kw)
        return self

    def predict_with_ci(self, X, confidence_weights=None,
                        level: float = CONFIDENCE_LEVEL) -> dict:
        Xs = self.scaler.transform(X)
        tree_preds = np.stack(
            [tree.predict(Xs) for tree in self.model.estimators_], axis=0
        )
        y_hat    = tree_preds.mean(axis=0)
        tree_var = tree_preds.var(axis=0)
        pred_std = np.sqrt(tree_var)

        z = stats.norm.ppf((1 + level) / 2)
        if confidence_weights is not None:
            cw       = np.clip(np.asarray(confidence_weights), 1e-3, 1.0)
            pred_std = pred_std / cw

        return dict(
            y_pred             = y_hat,
            ci_lower           = y_hat - z * pred_std,
            ci_upper           = y_hat + z * pred_std,
            pred_std           = pred_std,
            tree_variance      = tree_var,
            mean_tree_variance = float(tree_var.mean()),
        )

    def clone_base(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=RANDOM_SEED, n_jobs=-1,
        )


# ─────────────────────────────────────────────────────────────────────────────

class XGBoostWithBootstrapCI:
    """
    XGBoost with bootstrap variance:

        Var_boot(x) = (1/B) Σ_b [ f_b(x) − f̄_boot(x) ]²

    Primary prediction ŷ comes from the base model (full training set).
    Output units: raw overall_mave_score.
    """

    def __init__(self, n_bootstrap: int = N_BOOTSTRAP, **xgb_kwargs):
        self.n_bootstrap      = n_bootstrap
        self.scaler           = StandardScaler()
        self.xgb_kwargs       = xgb_kwargs
        self.base_model       = None
        self.bootstrap_models = []

    def _make_model(self, seed: int = RANDOM_SEED):
        return xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, **self.xgb_kwargs,
        )

    def fit(self, X, y, sample_weight=None):
        Xs  = self.scaler.fit_transform(X)
        y_a = np.asarray(y)

        self.base_model = self._make_model()
        fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.base_model.fit(Xs, y_a, **fit_kw)

        self.bootstrap_models = []
        rng = np.random.RandomState(RANDOM_SEED)
        n   = len(Xs)
        for i in range(self.n_bootstrap):
            idx    = rng.choice(n, size=n, replace=True)
            Xb, yb = Xs[idx], y_a[idx]
            swb    = sample_weight[idx] if sample_weight is not None else None
            m      = self._make_model(seed=RANDOM_SEED + i)
            fkw    = {"sample_weight": swb} if swb is not None else {}
            m.fit(Xb, yb, **fkw)
            self.bootstrap_models.append(m)
        return self

    def predict_with_ci(self, X, confidence_weights=None,
                        level: float = CONFIDENCE_LEVEL) -> dict:
        Xs    = self.scaler.transform(X)
        y_hat = self.base_model.predict(Xs)

        boot_preds = np.stack(
            [m.predict(Xs) for m in self.bootstrap_models], axis=0
        )
        boot_var = boot_preds.var(axis=0)
        pred_std = np.sqrt(boot_var)

        z = stats.norm.ppf((1 + level) / 2)
        if confidence_weights is not None:
            cw       = np.clip(np.asarray(confidence_weights), 1e-3, 1.0)
            pred_std = pred_std / cw

        return dict(
            y_pred                  = y_hat,
            ci_lower                = y_hat - z * pred_std,
            ci_upper                = y_hat + z * pred_std,
            pred_std                = pred_std,
            bootstrap_variance      = boot_var,
            mean_bootstrap_variance = float(boot_var.mean()),
        )

    def clone_base(self):
        return xgb.XGBRegressor(
            n_estimators=100, random_state=RANDOM_SEED, verbosity=0,
        )




class SplitConformalRegressor:
    """Wraps a base regressor and produces split conformal intervals."""
    def __init__(self, base_model):
        self.base_model=base_model
    def fit(self,X,y,sample_weight=None):
        from sklearn.model_selection import train_test_split
        Xtr,Xcal,ytr,ycal=train_test_split(X,y,test_size=CONFORMAL_CALIBRATION_FRACTION,random_state=RANDOM_SEED)
        kw={"sample_weight":sample_weight[:len(Xtr)]} if sample_weight is not None else {}
        self.base_model.fit(Xtr,ytr,**kw)
        cal_pred=self.base_model.predict(Xcal)
        self.qhat=float(np.quantile(np.abs(ycal-cal_pred),CONFIDENCE_LEVEL))
        return self
    def predict_with_ci(self,X,confidence_weights=None,level=CONFIDENCE_LEVEL):
        pred=self.base_model.predict(X)
        return dict(y_pred=pred,ci_lower=pred-self.qhat,ci_upper=pred+self.qhat,pred_std=np.repeat(self.qhat/1.96,len(pred)))

class QuantileRegressionModel:
    def __init__(self, alpha=1.0):
        self.scaler=StandardScaler()
        self.mid=Ridge(alpha=alpha)
        self.low=QuantileRegressor(quantile=0.025, alpha=alpha, solver="highs")
        self.high=QuantileRegressor(quantile=0.975, alpha=alpha, solver="highs")
    def fit(self,X,y,sample_weight=None):
        Xs=self.scaler.fit_transform(X)
        self.mid.fit(Xs,y)
        self.low.fit(Xs,y)
        self.high.fit(Xs,y)
        return self
    def predict_with_ci(self,X,confidence_weights=None,level=CONFIDENCE_LEVEL):
        Xs=self.scaler.transform(X)
        pred=self.mid.predict(Xs)
        lo=self.low.predict(Xs); hi=self.high.predict(Xs)
        return dict(y_pred=pred,ci_lower=lo,ci_upper=hi,pred_std=(hi-lo)/(2*1.96))
    def clone_base(self):
        return Ridge(alpha=1.0)

def tune_ridge_alpha(X,y,y_binary,sample_weight=None):
    alphas=[0.1,1,3,10,30,100]
    best_a,best=-1,None
    for a in alphas:
        cv=evaluate_cv(Ridge(alpha=a),X,y,y_binary,sample_weight=sample_weight)
        if cv["auroc_mean"]>best_a:
            best_a=cv["auroc_mean"]; best=a
    return best

# ─────────────────────────────────────────────────────────────────────────────
# 7. ABLATION STUDIES
# ─────────────────────────────────────────────────────────────────────────────
#
# Three complementary strategies:
#
#  (a) Leave-one-group-out  — drop one feature group; measure Δ AUROC vs full.
#  (b) Add-one-group-in     — start from AlphaMissense, accumulate groups.
#  (c) Leave-one-feature-out — drop each individual column; Δ AUROC per feature.
#      This is the most granular view of per-column importance.
#      Uses N_BOOTSTRAP_FEAT lighter XGBoost bootstrap to keep runtime feasible.
# ─────────────────────────────────────────────────────────────────────────────

def run_ablations(split: dict, feature_names: list) -> tuple:
    """
    Run all three ablation strategies.

    Returns
    -------
    group_df   : pd.DataFrame — group-level ablation results (a + b + baseline)
    feature_df : pd.DataFrame — per-feature ablation results (c)
    """
    # ── (a) + (b) + baseline — group level ──────────────────────────────────
    group_rows = []
    available  = {g: [f for f in feats if f in feature_names]
                  for g, feats in FEATURE_GROUPS.items()}

    print("\n[Ablation] (a) Leave-one-group-out")
    for removed in available:
        remaining = [f for g, fs in available.items() if g != removed for f in fs]
        if not remaining:
            continue
        _ablation_trial(split, remaining, f"drop_{removed}", group_rows,
                        strategy="leave_one_out", removed_group=removed,
                        n_bootstrap=N_BOOTSTRAP_ABL)

    print("\n[Ablation] (b) Add-one-group-in (forward)")
    accumulated = []
    for group, feats in available.items():
        accumulated += feats
        _ablation_trial(split, accumulated.copy(), f"add_{group}", group_rows,
                        strategy="forward", added_group=group,
                        n_bootstrap=N_BOOTSTRAP_ABL)

    print("\n[Ablation] (baseline) Full model")
    _ablation_trial(split, feature_names, "full_model", group_rows,
                    strategy="baseline", n_bootstrap=N_BOOTSTRAP_ABL)

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"\n[Ablation] Group-level results saved → ablation_results.csv")

    # ── (c) Leave-one-feature-out — per individual column ───────────────────
    print(f"\n[Ablation] (c) Leave-one-feature-out  ({len(feature_names)} features)")
    feature_rows = []
    for i, feat in enumerate(feature_names):
        remaining = [f for f in feature_names if f != feat]
        if not remaining:
            continue
        group = FEATURE_TO_GROUP.get(feat, "unknown")
        print(f"  [{i+1:2d}/{len(feature_names)}] drop '{feat}' (group: {group})")
        _ablation_trial(split, remaining, f"drop_feat_{feat}", feature_rows,
                        strategy="leave_one_feature_out",
                        removed_feature=feat, removed_feature_group=group,
                        n_bootstrap=N_BOOTSTRAP_FEAT)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(OUTPUT_DIR / "ablation_per_feature.csv", index=False)
    print(f"[Ablation] Per-feature results saved → ablation_per_feature.csv")

    return group_df, feature_df


def _ablation_trial(split, feature_subset, label, rows, n_bootstrap=N_BOOTSTRAP_ABL,
                    **meta):
    """Run all three models on a given feature subset; append CV + CI metrics."""
    Xtr  = split["X_train"][feature_subset]
    Xte  = split["X_test"][feature_subset]
    ytr  = split["y_train"];  yte = split["y_test"]
    ybt  = split["yb_train"]; ybv = split["yb_test"]
    cw_tr = split["cw_train"].values
    cw_te = split["cw_test"].values

    configs = [
        ("linear_regression", LinearRegressionWithCI()),
        ("random_forest",     RandomForestWithCI(n_estimators=100)),
        ("xgboost",           XGBoostWithBootstrapCI(n_bootstrap=n_bootstrap)),
    ]

    for model_name, model in configs:
        cv  = evaluate_cv(model.clone_base(), Xtr, ytr, ybt, sample_weight=cw_tr)
        model.fit(Xtr, ytr, sample_weight=cw_tr)
        ci  = model.predict_with_ci(Xte, confidence_weights=cw_te)

        test_auroc  = roc_auc_score(ybv, ci["y_pred"])
        ci_coverage = float(
            ((yte.values >= ci["ci_lower"]) & (yte.values <= ci["ci_upper"])).mean()
        )

        rows.append(dict(
            ablation       = label,
            model          = model_name,
            n_features     = len(feature_subset),
            cv_auroc_mean  = cv["auroc_mean"],
            cv_auroc_std   = cv["auroc_std"],
            cv_ap_mean     = cv["ap_mean"],
            cv_ap_std      = cv["ap_std"],
            cv_rmse_mean   = cv["rmse_mean"],
            cv_rmse_std    = cv["rmse_std"],
            test_auroc     = test_auroc,
            mean_pred_std  = float(ci["pred_std"].mean()),
            ci_coverage    = ci_coverage,
            **meta,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# 8. FINAL MODEL EVALUATION  (raw MAVE + severity scores)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_final_models(
    split: dict,
    feature_names: list,
    scaler: SeverityScaler,
) -> tuple:
    """
    Train each model on the full training split and evaluate on the held-out
    test set.  Produces both raw MAVE predictions and calibrated severity scores.

    Returns
    -------
    summary_df   : per-model metrics (CV + test)
    pred_df      : raw MAVE predictions + CIs for all three models
    severity_df  : [0,1] severity scores + CIs + tiers for all three models
    """
    Xtr  = split["X_train"][feature_names]
    Xte  = split["X_test"][feature_names]
    ytr  = split["y_train"];  yte = split["y_test"]
    ybt  = split["yb_train"]; ybv = split["yb_test"]
    cw_tr = split["cw_train"].values
    cw_te = split["cw_test"].values

    configs = [
        ("linear_regression", LinearRegressionWithCI()),
        ("random_forest",     RandomForestWithCI(n_estimators=300)),
        ("xgboost",           XGBoostWithBootstrapCI(n_bootstrap=N_BOOTSTRAP)),
    ]

    summary_rows = []
    pred_records     = {"y_true": yte.values, "y_binary": ybv.values}
    severity_records = {
        "y_true":          yte.values,
        "severity_true":   scaler.transform(yte.values),
        "severity_tier_true": np.where(
            scaler.transform(yte.values) < scaler.tier_benign, "Benign",
            np.where(scaler.transform(yte.values) < scaler.tier_pathogenic,
                     "VUS", "Pathogenic"),
        ),
    }

    for name, model in configs:
        print(f"\n[Final] ── {name} ──────────────────────────")

        cv = evaluate_cv(model.clone_base(), Xtr, ytr, ybt, sample_weight=cw_tr)
        print(
            f"  5-fold CV  AUROC : {cv['auroc_mean']:.3f} ± {cv['auroc_std']:.3f}\n"
            f"             PR-AUC: {cv['ap_mean']:.3f} ± {cv['ap_std']:.3f}\n"
            f"             RMSE  : {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}"
        )

        model.fit(Xtr, ytr, sample_weight=cw_tr)
        ci = model.predict_with_ci(Xte, confidence_weights=cw_te)

        test_auroc = roc_auc_score(ybv, ci["y_pred"])
        test_ap    = average_precision_score(ybv, ci["y_pred"])
        test_rmse  = float(np.sqrt(mean_squared_error(yte, ci["y_pred"])))
        in_ci      = float(
            ((yte.values >= ci["ci_lower"]) & (yte.values <= ci["ci_upper"])).mean()
        )

        # ── Severity scores ──────────────────────────────────────────────────
        sev = scaler.transform_ci(ci)
        sev_in_ci = float(
            ((severity_records["severity_true"] >= sev["severity_lower"])
             & (severity_records["severity_true"] <= sev["severity_upper"])).mean()
        )
        tier_summary = pd.Series(sev["severity_tier"]).value_counts().to_dict()

        print(
            f"  Test       AUROC : {test_auroc:.3f}\n"
            f"             PR-AUC: {test_ap:.3f}\n"
            f"             RMSE  : {test_rmse:.3f}\n"
            f"  CI coverage (95%)     : {in_ci:.1%}  (raw MAVE scale)\n"
            f"  Severity CI coverage  : {sev_in_ci:.1%}  (severity scale)\n"
            f"  Mean pred. std (raw)  : {ci['pred_std'].mean():.4f}\n"
            f"  Mean severity std     : {sev['severity_std'].mean():.4f}\n"
            f"  Severity tier counts  : {tier_summary}"
        )

        if isinstance(model, LinearRegressionWithCI):
            coefs = model.get_coefficients(feature_names)
            print(f"\n  Feature coefficients (top 10):\n{coefs.head(10).to_string()}")
            coefs.to_csv(OUTPUT_DIR / "lr_coefficients.csv", header=["coefficient"])

        # Store raw predictions
        pred_records[f"{name}_pred"]  = ci["y_pred"]
        pred_records[f"{name}_lower"] = ci["ci_lower"]
        pred_records[f"{name}_upper"] = ci["ci_upper"]
        pred_records[f"{name}_std"]   = ci["pred_std"]

        # Store severity scores
        severity_records[f"{name}_severity"]       = sev["severity"]
        severity_records[f"{name}_severity_lower"]  = sev["severity_lower"]
        severity_records[f"{name}_severity_upper"]  = sev["severity_upper"]
        severity_records[f"{name}_severity_std"]    = sev["severity_std"]
        severity_records[f"{name}_severity_tier"]   = sev["severity_tier"]

        summary_rows.append(dict(
            model             = name,
            cv_auroc          = f"{cv['auroc_mean']:.3f}±{cv['auroc_std']:.3f}",
            cv_ap             = f"{cv['ap_mean']:.3f}±{cv['ap_std']:.3f}",
            cv_rmse           = f"{cv['rmse_mean']:.3f}±{cv['rmse_std']:.3f}",
            test_auroc        = round(test_auroc, 4),
            test_ap           = round(test_ap, 4),
            test_rmse         = round(test_rmse, 4),
            ci_coverage_raw   = round(in_ci, 4),
            ci_coverage_sev   = round(sev_in_ci, 4),
            mean_pred_std_raw = round(float(ci["pred_std"].mean()), 4),
            mean_severity_std = round(float(sev["severity_std"].mean()), 4),
        ))

    summary_df  = pd.DataFrame(summary_rows)
    pred_df     = pd.DataFrame(pred_records)
    severity_df = pd.DataFrame(severity_records)

    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv",          index=False)
    pred_df.to_csv   (OUTPUT_DIR / "test_predictions.csv",       index=False)
    severity_df.to_csv(OUTPUT_DIR / "test_severity_scores.csv",  index=False)
    print(f"\n[Final] model_summary.csv, test_predictions.csv, "
          f"test_severity_scores.csv saved.")

    return summary_df, pred_df, severity_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(split: dict, feature_names: list) -> dict:
    """
    SHAP values for Random Forest and XGBoost.

    Outputs:
      shap_group_contributions.csv  — mean |SHAP| per group × model
      shap_summary_{model}.png      — beeswarm plot
      shap_group_contributions.png  — group bar chart
    """
    Xtr = split["X_train"][feature_names]
    Xte = split["X_test"][feature_names]
    ytr = split["y_train"]
    cw  = split["cw_train"].values

    shap_vals  = {}
    model_objs = {}

    for tag, model in [
        ("random_forest", RandomForestWithCI(n_estimators=300)),
        ("xgboost",       XGBoostWithBootstrapCI(n_bootstrap=N_BOOTSTRAP)),
    ]:
        print(f"[SHAP] Fitting {tag}...")
        model.fit(Xtr, ytr, sample_weight=cw)
        Xte_s     = model.scaler.transform(Xte)
        base_est  = model.model if hasattr(model, "model") else model.base_model
        explainer = shap.TreeExplainer(base_est)
        sv        = explainer.shap_values(Xte_s)

        shap_vals[tag]  = sv
        model_objs[tag] = (model, Xte_s)
        print(f"[SHAP] {tag}: shape={sv.shape}")

    # Per-group mean |SHAP|
    group_rows = []
    for model_tag, sv in shap_vals.items():
        row = {"model": model_tag}
        for g, feats in FEATURE_GROUPS.items():
            idxs = [feature_names.index(f) for f in feats if f in feature_names]
            row[g] = float(np.abs(sv[:, idxs]).mean()) if idxs else 0.0
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows).set_index("model")
    group_df.to_csv(OUTPUT_DIR / "shap_group_contributions.csv")
    print(f"\n[SHAP] Group contributions:\n{group_df.round(4).to_string()}\n")

    # Beeswarm plots
    for tag, sv in shap_vals.items():
        _, Xte_s = model_objs[tag]
        plt.figure(figsize=(8, max(4, len(feature_names) * 0.4)))
        shap.summary_plot(sv, Xte_s, feature_names=feature_names,
                          plot_type="dot", show=False, max_display=20)
        plt.title(f"SHAP Beeswarm — {tag.replace('_', ' ').title()}", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_summary_{tag}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved shap_summary_{tag}.png")

    # Group bar chart
    group_df.T.plot(kind="bar", figsize=(10, 5), colormap="tab10",
                    edgecolor="black", linewidth=0.4)
    plt.title("Mean |SHAP| by Feature Group — Structural Features vs AlphaMissense",
              fontsize=12)
    plt.ylabel("Mean |SHAP value|")
    plt.xlabel("Feature Group")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_group_contributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[SHAP] Saved shap_group_contributions.png")

    return dict(shap_values=shap_vals, group_contributions=group_df)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ci_comparison(pred_df: pd.DataFrame):
    """Raw MAVE predictions with 95% CIs, sorted by true score."""
    model_tags    = ["linear_regression", "random_forest", "xgboost"]
    display_names = ["Linear Regression", "Random Forest", "XGBoost"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    order = np.argsort(pred_df["y_true"].values)
    x     = np.arange(len(order))

    for ax, tag, dname in zip(axes, model_tags, display_names):
        y_true  = pred_df["y_true"].values[order]
        y_pred  = pred_df[f"{tag}_pred"].values[order]
        y_lower = pred_df[f"{tag}_lower"].values[order]
        y_upper = pred_df[f"{tag}_upper"].values[order]

        ax.fill_between(x, y_lower, y_upper, alpha=0.25, color="steelblue",
                        label="95% CI")
        ax.plot(x, y_pred, color="steelblue", lw=1.5, label="Predicted")
        ax.scatter(x, y_true, s=16, color="firebrick", zorder=3, alpha=0.75,
                   label="True")
        ax.set_title(dname, fontsize=11, fontweight="bold")
        ax.set_xlabel("Variants (sorted by true MAVE score)", fontsize=9)
        ax.axhline(0, color="grey", lw=0.5, ls="--")

    axes[0].set_ylabel("Overall MAVE Score (raw)", fontsize=10)
    axes[0].legend(fontsize=8)
    plt.suptitle("Test-Set Raw MAVE Predictions with 95% CIs", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ci_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved ci_comparison.png")


def plot_severity_scores(severity_df: pd.DataFrame, scaler: SeverityScaler):
    """
    Severity score waterfall plot for each model.

    Variants are sorted by XGBoost predicted severity (highest severity first).
    CI bars are shown; background bands mark Benign / VUS / Pathogenic regions.
    True severity is overlaid as red dots.
    """
    model_tags    = ["linear_regression", "random_forest", "xgboost"]
    display_names = ["Linear Regression", "Random Forest", "XGBoost"]

    tier_colors = {
        "Pathogenic": "#d73027",
        "VUS":        "#fc8d59",
        "Benign":     "#4dac26",
    }

    # Sort by XGBoost severity descending
    order = np.argsort(severity_df["xgboost_severity"].values)[::-1]
    x     = np.arange(len(order))

    fig, axes = plt.subplots(len(model_tags), 1,
                             figsize=(14, 4 * len(model_tags)), sharex=True)

    for ax, tag, dname in zip(axes, model_tags, display_names):
        sev       = severity_df[f"{tag}_severity"].values[order]
        sev_lower = severity_df[f"{tag}_severity_lower"].values[order]
        sev_upper = severity_df[f"{tag}_severity_upper"].values[order]
        sev_true  = severity_df["severity_true"].values[order]
        tiers     = severity_df[f"{tag}_severity_tier"].values[order]

        # Background tier bands
        ax.axhspan(0,                      scaler.tier_benign,     alpha=0.07,
                   color=tier_colors["Benign"],     zorder=0)
        ax.axhspan(scaler.tier_benign,     scaler.tier_pathogenic, alpha=0.07,
                   color=tier_colors["VUS"],        zorder=0)
        ax.axhspan(scaler.tier_pathogenic, 1.0,                    alpha=0.07,
                   color=tier_colors["Pathogenic"], zorder=0)

        # Tier-coloured CI bars (thin; use errorbar for each point)
        for xi, sv, sl, su, tier in zip(x, sev, sev_lower, sev_upper, tiers):
            color = tier_colors[tier]
            ax.plot([xi, xi], [sl, su], color=color, lw=0.7, alpha=0.6, zorder=1)
            ax.scatter(xi, sv, s=12, color=color, zorder=2)

        # True severity overlay
        ax.scatter(x, sev_true, s=10, color="black", marker="x", zorder=3,
                   alpha=0.7, label="True severity")

        # Tier boundary lines
        ax.axhline(scaler.tier_benign,     color="grey", lw=0.8, ls="--")
        ax.axhline(scaler.tier_pathogenic, color="grey", lw=0.8, ls="--")

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Severity score", fontsize=9)
        ax.set_title(dname, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

        # Tier count annotations on y-axis right side
        tier_counts = pd.Series(tiers).value_counts()
        annotation  = "  |  ".join(
            f"{t}: {tier_counts.get(t, 0)}" for t in ["Benign", "VUS", "Pathogenic"]
        )
        ax.set_title(f"{dname}   ({annotation})", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel(
        "Variants ranked by predicted severity (XGBoost, high → low)", fontsize=9
    )
    plt.suptitle(
        f"Predicted Severity Scores with 95% CIs\n"
        f"(0 = WT-like / benign,  1 = full loss-of-function / pathogenic)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "severity_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved severity_scores.png")


def plot_ablation_results(group_df: pd.DataFrame):
    """Group-level AUROC heatmap + epistemic uncertainty bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    pivot_auroc = group_df.pivot_table(
        index="ablation", columns="model", values="cv_auroc_mean"
    )
    sns.heatmap(pivot_auroc, ax=axes[0], annot=True, fmt=".3f",
                cmap="RdYlGn", vmin=0.4, vmax=1.0, linewidths=0.4,
                cbar_kws={"label": "CV AUROC"})
    axes[0].set_title("5-Fold CV AUROC — Group Ablations × Model", fontsize=11)
    axes[0].set_xlabel(""); axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=20)

    pivot_std = group_df.pivot_table(
        index="ablation", columns="model", values="mean_pred_std"
    )
    pivot_std.plot(kind="bar", ax=axes[1], colormap="Set2",
                   edgecolor="black", linewidth=0.4)
    axes[1].set_title("Mean Prediction Std by Group Ablation\n(Epistemic Uncertainty)",
                      fontsize=11)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean Prediction Std (raw MAVE units)")
    axes[1].legend(title="Model", fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved ablation_results.png")


def plot_per_feature_ablation(
    feature_df: pd.DataFrame,
    group_df: pd.DataFrame,
):
    """
    Horizontal bar chart of Δ AUROC per individual feature, one panel per model.

    Δ AUROC = full_model_auroc − drop_feature_auroc.
    A larger bar → removing this feature hurts more → feature is more important.
    Bars are coloured by feature group.
    """
    model_tags    = ["linear_regression", "random_forest", "xgboost"]
    display_names = ["Linear Regression", "Random Forest", "XGBoost"]

    # Full model baseline AUROC per model
    full_row = group_df[group_df["ablation"] == "full_model"]
    full_auroc = {
        row["model"]: row["cv_auroc_mean"]
        for _, row in full_row.iterrows()
    }

    # Unique feature names from the per-feature df (strip "drop_feat_" prefix)
    feature_df = feature_df.copy()
    feature_df["feature"] = feature_df["ablation"].str.replace(
        "^drop_feat_", "", regex=True
    )

    # Mean Δ AUROC across models — used for sorting
    delta_rows = []
    for feat, sub in feature_df.groupby("feature"):
        row = {"feature": feat}
        for m_tag in model_tags:
            m_sub = sub[sub["model"] == m_tag]
            if m_sub.empty:
                row[m_tag] = 0.0
                continue
            ablated_auroc = m_sub["cv_auroc_mean"].values[0]
            row[m_tag] = max(0.0, full_auroc.get(m_tag, ablated_auroc) - ablated_auroc)
        row["mean_delta"] = np.mean([row[m] for m in model_tags])
        row["group"] = FEATURE_TO_GROUP.get(feat, "unknown")
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows).sort_values("mean_delta", ascending=True)

    # Colour palette: one colour per feature group
    groups    = list(FEATURE_GROUPS.keys())
    palette   = plt.cm.tab10.colors
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(delta_df) * 0.35 + 2)),
                             sharey=True)

    for ax, m_tag, dname in zip(axes, model_tags, display_names):
        vals  = delta_df[m_tag].values
        feats = delta_df["feature"].values
        grps  = delta_df["group"].values
        colors = [group_color.get(g, "grey") for g in grps]

        bars = ax.barh(range(len(feats)), vals, color=colors, edgecolor="white",
                       linewidth=0.4)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("Δ AUROC (full − ablated)", fontsize=9)
        ax.set_title(dname, fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", lw=0.6)

        # Annotate bar values
        for bar, val in zip(bars, vals):
            if val > 0.001:
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7)

    # Legend for feature groups
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_color[g], label=g)
        for g in groups if g in group_color
    ]
    axes[-1].legend(handles=handles, title="Feature group",
                    bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.suptitle(
        "Per-Feature Importance: Δ AUROC from Leave-One-Feature-Out Ablation\n"
        "(sorted by mean Δ AUROC across models; larger = more important)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_per_feature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved ablation_per_feature.png")


def plot_variance_by_coverage(pred_df: pd.DataFrame, split: dict):
    """Prediction std vs. confidence weight, coloured by MAVE coverage."""
    cov = split["cov_test"].values
    cw  = split["cw_test"].values

    model_tags    = ["linear_regression", "random_forest", "xgboost"]
    display_names = ["Linear Regression", "Random Forest", "XGBoost"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, tag, dname in zip(axes, model_tags, display_names):
        std = pred_df[f"{tag}_std"].values
        for c in sorted(np.unique(cov)):
            mask = cov == c
            ax.scatter(cw[mask], std[mask], label=f"Coverage={int(c)}",
                       s=25, alpha=0.7)
        ax.set_xlabel("Confidence Weight\n(coverage × pLDDT)", fontsize=9)
        ax.set_ylabel("Prediction Std (raw MAVE units)", fontsize=9)
        ax.set_title(dname, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, title="MAVE coverage")

    plt.suptitle("Prediction Uncertainty vs Confidence Weight", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "variance_by_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved variance_by_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(aim1_path: str = "brca1_features_2.csv",
         aim2_path: str = "BRCA1_Final.csv"):
    print("=" * 70)
    print("  BRCA1 Missense Severity — ML Pipeline  (v2)")
    print("  Raw MAVE + Severity Scores · CIs · Per-Feature Ablation · SHAP")
    print("=" * 70 + "\n")
    print(f"[Config] Aim 1 (structural) : {aim1_path}")
    print(f"[Config] Aim 2 (MAVE / AM)  : {aim2_path}\n")

    # 1 ── Load & merge
    df = load_and_merge(aim1_path, aim2_path)

    # 2 ── Preprocess
    X, y, y_binary, coverage, confidence_weight = preprocess(df)
    feature_names = list(X.columns)

    # 3 ── Split
    split = split_data(X, y, y_binary, coverage, confidence_weight)

    # 4 ── Calibrate severity scaler on training labels only
    print("\n── Severity Scaler Calibration ──────────────────────────────────────")
    scaler = SeverityScaler().fit(split["y_train"].values)

    # 5 ── Ablation studies (group-level + per-feature)
    print("\n── Ablation Studies ─────────────────────────────────────────────────")
    group_df, feature_df = run_ablations(split, feature_names)
    plot_ablation_results(group_df)
    plot_per_feature_ablation(feature_df, group_df)

    # 6 ── Final model evaluation (raw MAVE + severity scores)
    print("\n── Final Model Evaluation ───────────────────────────────────────────")
    summary_df, pred_df, severity_df = evaluate_final_models(
        split, feature_names, scaler
    )

    print("\n── Summary Table ────────────────────────────────────────────────────")
    print(summary_df.to_string(index=False))

    # 7 ── Plots
    plot_ci_comparison(pred_df)
    plot_severity_scores(severity_df, scaler)
    plot_variance_by_coverage(pred_df, split)

    # 8 ── SHAP
    print("\n── SHAP Analysis ────────────────────────────────────────────────────")
    run_shap_analysis(split, feature_names)

    print("\n" + "=" * 70)
    print(f"  Done.  All outputs written to ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    # Usage:
    #   python pipeline_v2.py
    #   python pipeline_v2.py brca1_features_2.csv BRCA1_Final.csv
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
