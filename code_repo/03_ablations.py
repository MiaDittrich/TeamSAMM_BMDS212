#!/usr/bin/env python3
"""
Standalone ablation re-run for the BRCA1 MAVE ElasticNet model.

Uses the same extended split + "both" label + full feature set as the main pipeline.
Fits the ElasticNet (with inner 3-fold GridSearchCV) to find best params, then
runs leave-one-feature-out and leave-one-group-out ablations with those fixed params.

Outputs: results_ext_both/ablation_features.csv
         results_ext_both/ablation_groups.csv
"""

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV
from sklearn.pipeline import Pipeline

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
OUTDIR = BASE / "results"
OUTDIR.mkdir(exist_ok=True)

# ─── Constants (mirror brca1_pipeline_final.py) ──────────────────────────────
AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

CLINVAR_ORIGINAL_20 = {
    "S1841R", "L1839S", "V1838E", "M1775R", "Y1703S",
    "W1718L", "G1706R", "G1738E", "S1715R", "I1760S",
    "D1733G", "I1766V", "T1773S", "V1736I", "K1793Q",
    "E1794G", "S1797C", "H1862L", "P1831S", "E1829T",
}

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

SPEARMAN_SCORER = __import__("sklearn.metrics", fromlist=["make_scorer"]).make_scorer(
    lambda y, yhat: float(stats.spearmanr(y, yhat)[0]),
    greater_is_better=True,
)


# ─── Data loading ────────────────────────────────────────────────────────────
def parse_variant(v):
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def load_training_data():
    feats      = pd.read_csv(BASE / "data" / "brca1_final_feats.csv")
    mave       = pd.read_csv(BASE / "data" / "brca1_mave.csv")
    cv_test    = pd.read_csv(BASE / "data" / "clinvar_test.csv",    usecols=[0, 1])
    cv_explore = pd.read_csv(BASE / "data" / "clinvar_explore.csv", usecols=[0, 1])

    # Parse variant → join keys
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

    # Feature engineering
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

    # Extended split: put CLINVAR_ORIGINAL_20 back into training
    test_variants_all  = set(cv_test["variant"])
    explore_variants   = set(cv_explore["variant"])
    test_variants_held = test_variants_all - CLINVAR_ORIGINAL_20   # 28 clean holdout
    train_exclude      = test_variants_held | explore_variants

    train_all = merged[~merged["variant"].isin(train_exclude)].copy()

    # "both" label: mean of cisplatin + HDR (only rows where both exist)
    mask  = train_all["cisplatin_score"].notna() & train_all["hdr_activity_score"].notna()
    label = train_all.loc[mask, ["cisplatin_score", "hdr_activity_score"]].mean(axis=1)
    train_all["label"] = label.reindex(train_all.index)

    train_df = train_all[train_all["label"].notna()].copy().reset_index(drop=True)

    print(f"Training set: n={len(train_df)}, features={len(all_feats)}")
    return train_df, all_feats


# ─── ElasticNet fit ───────────────────────────────────────────────────────────
def fit_elasticnet(train_df, all_feats, seed=42):
    X = train_df[all_feats]
    y = train_df["label"]

    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    enet_est = GridSearchCV(
        ElasticNet(max_iter=10_000, random_state=seed),
        param_grid={"alpha": [1e-3, 1e-2, 1e-1], "l1_ratio": [0.2, 0.5, 0.8]},
        cv=inner,
        scoring=SPEARMAN_SCORER,
        n_jobs=-1,
        refit=True,
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", enet_est),
    ])
    pipe.fit(X, y)

    best = pipe.named_steps["est"].best_estimator_
    best_params = {"alpha": best.alpha, "l1_ratio": best.l1_ratio}
    print(f"Best params: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    return best_params


# ─── Ablation CV ─────────────────────────────────────────────────────────────
def ablation_cv(X, y, best_params, n_folds=5, n_repeats=5, seed=42):
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
        m.fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr],
              y_arr[tr])
        yhat = m.predict(X.iloc[te] if hasattr(X, "iloc") else X[te])
        oof_sum[te] += yhat
        oof_cnt[te] += 1
        spearmans.append(float(stats.spearmanr(y_arr[te], yhat)[0]))
        pearsons.append(float(stats.pearsonr(y_arr[te], yhat)[0]))
        r2s.append(float(r2_score(y_arr[te], yhat)))
        rmses.append(float(np.sqrt(mean_squared_error(y_arr[te], yhat))))

    oof = np.where(oof_cnt > 0, oof_sum / oof_cnt, np.nan)
    thr   = float(np.median(y_arr))
    y_bin = (y_arr <= thr).astype(int)
    auroc = float(roc_auc_score(y_bin, -oof)) if y_bin.min() != y_bin.max() else np.nan

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


# ─── Main ablation logic ──────────────────────────────────────────────────────
def run_ablations(train_df, all_feats, best_params, n_folds=5, n_repeats=5, seed=42):
    X = train_df[all_feats]
    y = train_df["label"]

    print("\nComputing baseline …", end=" ", flush=True)
    base = ablation_cv(X, y, best_params, n_folds, n_repeats, seed)
    print(f"Spearman={base['spearman']:+.4f} ± {base['spearman_std']:.4f}  "
          f"R²={base['r2']:+.4f} ± {base['r2_std']:.4f}  "
          f"AUROC={base['auroc']:.4f}")

    feat_to_grp = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}

    # Leave-one-feature-out
    print(f"\nLeave-one-feature-out ({len(all_feats)} features):")
    feat_rows = []
    for feat in all_feats:
        X_drop = X.drop(columns=[feat])
        res = ablation_cv(X_drop, y, best_params, n_folds, n_repeats, seed)
        row = {
            "feature":         feat,
            "group":           feat_to_grp.get(feat, "other"),
            "baseline_sp":     base["spearman"],
            "sp_without":      res["spearman"],
            "sp_std_without":  res["spearman_std"],
            "delta_sp":        base["spearman"] - res["spearman"],
            "baseline_pe":     base["pearson"],
            "pe_without":      res["pearson"],
            "pe_std_without":  res["pearson_std"],
            "delta_pe":        base["pearson"] - res["pearson"],
            "baseline_r2":     base["r2"],
            "baseline_r2_std": base["r2_std"],
            "r2_without":      res["r2"],
            "r2_std_without":  res["r2_std"],
            "delta_r2":        base["r2"] - res["r2"],
            "auroc_without":   res["auroc"],
            "delta_auroc":     base["auroc"] - res["auroc"],
        }
        feat_rows.append(row)
        dsym = "▲" if row["delta_sp"] > 0.005 else ("▼" if row["delta_sp"] < -0.005 else "─")
        print(f"  {dsym} {feat:<32}  Δsp={row['delta_sp']:+.4f}  "
              f"Δr2={row['delta_r2']:+.4f}  sp_w={res['spearman']:+.4f}")

    feat_df = pd.DataFrame(feat_rows).sort_values("delta_r2", ascending=False)

    # Leave-one-group-out
    print("\nLeave-one-group-out:")
    group_rows = []
    for grp, cols in FEATURE_GROUPS.items():
        present   = [c for c in cols if c in all_feats]
        remaining = [f for f in all_feats if f not in present]
        if not present or not remaining:
            continue
        res = ablation_cv(train_df[remaining], y, best_params, n_folds, n_repeats, seed)
        row = {
            "group":            grp,
            "n_features":       len(present),
            "features_removed": ", ".join(present),
            "baseline_sp":      base["spearman"],
            "sp_without":       res["spearman"],
            "sp_std_without":   res["spearman_std"],
            "delta_sp":         base["spearman"] - res["spearman"],
            "baseline_pe":      base["pearson"],
            "pe_without":       res["pearson"],
            "pe_std_without":   res["pearson_std"],
            "delta_pe":         base["pearson"] - res["pearson"],
            "baseline_r2":      base["r2"],
            "baseline_r2_std":  base["r2_std"],
            "r2_without":       res["r2"],
            "r2_std_without":   res["r2_std"],
            "delta_r2":         base["r2"] - res["r2"],
            "auroc_without":    res["auroc"],
            "delta_auroc":      base["auroc"] - res["auroc"],
        }
        group_rows.append(row)
        dsym = "▲" if row["delta_sp"] > 0.005 else ("▼" if row["delta_sp"] < -0.005 else "─")
        print(f"  {dsym} {grp:<18}  Δsp={row['delta_sp']:+.4f}  "
              f"Δr2={row['delta_r2']:+.4f}  ({len(present)} features removed)")

    group_df = pd.DataFrame(group_rows).sort_values("delta_r2", ascending=False)

    feat_df.to_csv(OUTDIR / "ablation_features.csv", index=False)
    group_df.to_csv(OUTDIR / "ablation_groups.csv", index=False)
    print(f"\nSaved → {OUTDIR}/ablation_features.csv  ({len(feat_df)} features)")
    print(f"Saved → {OUTDIR}/ablation_groups.csv    ({len(group_df)} groups)")
    return feat_df, group_df, base


if __name__ == "__main__":
    import sys
    seed = 42
    n_folds, n_repeats = 5, 5

    print("=" * 60)
    print("  BRCA1 Ablation Re-run (ElasticNet, extended split, both)")
    print("=" * 60)

    train_df, all_feats = load_training_data()

    print("\nFitting ElasticNet to find best hyperparameters …")
    best_params = fit_elasticnet(train_df, all_feats, seed=seed)

    run_ablations(train_df, all_feats, best_params,
                  n_folds=n_folds, n_repeats=n_repeats, seed=seed)

    print("\nDone.")
