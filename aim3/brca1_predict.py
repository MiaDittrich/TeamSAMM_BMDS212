#!/usr/bin/env python3
"""
BRCA1 ClinVar Variant Predictor
================================
Trains a model to predict BRCA1 MAVE functional scores from structural,
biochemical, evolutionary, and AlphaMissense features, then classifies
20 held-out ClinVar variants as pathogenic or benign.

Quick start
-----------
  python brca1_predict.py                   # RF, 'both' label (defaults)
  python brca1_predict.py --model ridge     # Ridge regression instead
  python brca1_predict.py --labels cisplatin
  python brca1_predict.py --save-model      # write trained_model.joblib
  python brca1_predict.py --load-model      # skip training, reuse saved model

Outputs  (written to --outdir, default: results/)
-------------------------------------------------
  clinvar_predictions.csv   20-variant hold-out: predicted scores + ClinVar labels
  metrics_summary.csv       CV regression metrics + ClinVar classification metrics
  clinvar_plot.png          predicted score plot with 95 % bootstrap CIs
  trained_model.joblib      serialised pipeline  (--save-model only)
"""

import re, sys, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import joblib

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  ← edit here to change behaviour
# ══════════════════════════════════════════════════════════════════════════════

MODEL          = "rf"    # "rf"  | "ridge"
LABEL_STRATEGY = "both"  # "both"| "any" | "cisplatin" | "hdr"

# Cross-validation
N_FOLDS     = 5    # outer folds
N_REPEATS   = 5    # repeats of the full k-fold (25 total folds)
SEED        = 42

# Bootstrap confidence intervals for ClinVar predictions
N_BOOTSTRAP = 200

# ── Random Forest  (hyperparameters selected via nested CV in full pipeline) ──
RF_PARAMS = dict(
    n_estimators     = 400,    # number of trees
    max_features     = "sqrt", # features per split
    min_samples_leaf = 2,      # minimum leaf size
    n_jobs           = -1,     # use all CPU cores
    random_state     = SEED,
)

# ── Ridge Regression ──────────────────────────────────────────────────────────
RIDGE_ALPHA = 1.0

# ── KNN Imputer (shared by both models) ──────────────────────────────────────
KNN_NEIGHBORS = 5   # neighbours used to impute missing feature values

# ── Classification threshold ──────────────────────────────────────────────────
# Predicted MAVE score below this value → classified as "pathogenic".
# None → use training-set median (data-driven; recommended).
CLASSIFICATION_THRESHOLD = None

# ── File paths ────────────────────────────────────────────────────────────────
FEATS_CSV  = "BRCA1_FEATS.csv"
MAVE_CSV   = "BRCA1_MAVE.csv"
OUTDIR     = "results"
MODEL_FILE = "trained_model.joblib"


# ══════════════════════════════════════════════════════════════════════════════
# CLINVAR HELD-OUT TEST SET
# All 20 variants confirmed present in both BRCA1_FEATS and BRCA1_MAVE,
# guaranteeing complete AlphaMissense scores.
# ══════════════════════════════════════════════════════════════════════════════

CLINVAR_PATHOGENIC = [
    "S1841R", "L1839S", "V1838E", "M1775R", "Y1703S",
    "W1718L", "G1706R", "G1738E", "S1715R", "I1760S",
]
CLINVAR_BENIGN = [
    "D1733G", "I1766V", "T1773S", "V1736I", "K1793Q",
    "E1794G", "S1797C", "H1862L", "P1831S", "E1829T",
]
CLINVAR_ALL = CLINVAR_PATHOGENIC + CLINVAR_BENIGN

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

AM_CLASS_MAP = {"benign": 0, "ambiguous": 1, "pathogenic": 2}

BASE_FEATURES = [
    # Structural (AlphaFold2-derived)
    "mutant_plddt", "ca_rmsd", "backbone_rmsd", "mutant_ca_displacement",
    "shell_rmsd_5A", "shell_rmsd_8A", "shell_rmsd_12A",
    "ramachandran_violation", "is_interface_residue", "is_disordered_variant",
    # Biochemical (amino-acid property changes)
    "pam250_score", "delta_hydrophobicity", "delta_size",
    "delta_charge", "delta_aromaticity",
    "is_charge_reversal", "is_size_increase",
    "is_hydrophobic_to_polar", "is_polar_to_hydrophobic",
    # AlphaMissense
    "am_pathogenicity", "am_class_enc",
    # Evolutionary (EvoEF2 ΔΔG)
    "evoef2_ddg_Total", "ddg_evoef2",
]
# Three engineered features added during preprocessing
ENGINEERED_FEATURES = ["am_pathogenicity_sq", "am_x_evo", "plddt_rmsd"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_variant(v):
    m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", str(v).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)


def load_and_merge(feats_path: str, mave_path: str) -> pd.DataFrame:
    feats = pd.read_csv(feats_path)
    mave  = pd.read_csv(mave_path)

    parsed = feats["variant"].apply(
        lambda v: pd.Series(_parse_variant(v), index=["_wt", "_pos", "_mut"])
    )
    feats = pd.concat([feats, parsed], axis=1)

    mave_cols = ["uniprot_position", "ref_aa", "alt_aa",
                 "am_pathogenicity", "am_class", "cisplatin_score", "hdr_activity_score"]
    merged = feats.merge(
        mave[mave_cols],
        left_on=["mutant_residue", "_wt", "_mut"],
        right_on=["uniprot_position", "ref_aa", "alt_aa"],
        how="left", validate="m:1",
    ).drop(columns=["_wt", "_pos", "_mut", "uniprot_position", "ref_aa", "alt_aa"])

    print(f"  Loaded {len(feats)} variants from FEATS, "
          f"{merged['am_pathogenicity'].notna().sum()} with AlphaMissense scores")
    return merged


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode am_class, drop all-NaN columns, add 3 engineered features."""
    df = df.copy()
    df["am_class_enc"] = df["am_class"].map(AM_CLASS_MAP)

    # Keep only features actually present and non-empty
    features = [f for f in BASE_FEATURES
                if f in df.columns and not df[f].isna().all()]

    # Engineered feature 1: am_pathogenicity² (non-linear AlphaMissense signal)
    if "am_pathogenicity" in features:
        df["am_pathogenicity_sq"] = df["am_pathogenicity"] ** 2
        features.append("am_pathogenicity_sq")

    # Engineered feature 2: AlphaMissense × log-|EvoEF2| interaction
    if "am_pathogenicity" in features and "evoef2_ddg_Total" in features:
        evo = df["evoef2_ddg_Total"].fillna(0)
        df["am_x_evo"] = df["am_pathogenicity"] * np.sign(evo) * np.log1p(evo.abs())
        features.append("am_x_evo")

    # Engineered feature 3: pLDDT-weighted CA displacement
    if "mutant_plddt" in features and "ca_rmsd" in features:
        df["plddt_rmsd"] = (df["mutant_plddt"] / 100.0) * df["ca_rmsd"]
        features.append("plddt_rmsd")

    return df, features


def build_label_set(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Return rows with a non-null 'label' column for the chosen strategy."""
    if strategy == "both":
        mask = df["cisplatin_score"].notna() & df["hdr_activity_score"].notna()
        out = df[mask].copy()
        out["label"] = (out["cisplatin_score"] + out["hdr_activity_score"]) / 2
    elif strategy == "any":
        def _mean(r):
            v = [x for x in [r["cisplatin_score"], r["hdr_activity_score"]] if pd.notna(x)]
            return float(np.mean(v)) if v else np.nan
        out = df.copy()
        out["label"] = out.apply(_mean, axis=1)
        out = out[out["label"].notna()]
    elif strategy == "cisplatin":
        out = df[df["cisplatin_score"].notna()].copy()
        out["label"] = out["cisplatin_score"]
    elif strategy == "hdr":
        out = df[df["hdr_activity_score"].notna()].copy()
        out["label"] = out["hdr_activity_score"]
    else:
        raise ValueError(f"Unknown label strategy: {strategy!r}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(model_name: str) -> Pipeline:
    """Return a sklearn Pipeline ready for fit/predict."""
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    if model_name == "rf":
        return Pipeline([
            ("imp", imputer),
            ("est", RandomForestRegressor(**RF_PARAMS)),
        ])
    elif model_name == "ridge":
        return Pipeline([
            ("imp", imputer),
            ("scl", StandardScaler()),
            ("est", Ridge(alpha=RIDGE_ALPHA)),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name!r}  (choose 'rf' or 'ridge')")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(pipe, X: pd.DataFrame, y: pd.Series) -> dict:
    """Repeated K-Fold CV; returns per-fold metric arrays and OOF predictions."""
    rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    r2s, rmses, maes, pearsons, spearmans = [], [], [], [], []
    oof_sum = np.zeros(len(y))
    oof_cnt = np.zeros(len(y))

    for tr, te in rkf.split(X):
        m = clone(pipe)
        m.fit(X.iloc[tr], y.values[tr])
        yhat = m.predict(X.iloc[te])
        oof_sum[te] += yhat
        oof_cnt[te] += 1
        r2s.append(r2_score(y.values[te], yhat))
        rmses.append(float(np.sqrt(mean_squared_error(y.values[te], yhat))))
        maes.append(float(mean_absolute_error(y.values[te], yhat)))
        pearsons.append(_safe_corr(stats.pearsonr,  y.values[te], yhat))
        spearmans.append(_safe_corr(stats.spearmanr, y.values[te], yhat))

    oof_pred = np.where(oof_cnt > 0, oof_sum / oof_cnt, np.nan)
    # Classification view (low score = damaging)
    thr = float(np.median(y.values))
    y_bin = (y.values <= thr).astype(int)
    auroc = float(roc_auc_score(y_bin, -oof_pred)) if y_bin.min() != y_bin.max() else np.nan

    return dict(
        spearman = np.array(spearmans),
        r2       = np.array(r2s),
        pearson  = np.array(pearsons),
        rmse     = np.array(rmses),
        mae      = np.array(maes),
        auroc    = auroc,
    )


def _safe_corr(fn, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    try:
        return float(fn(a, b)[0])
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def _strip_gridsearch(fitted_pipe) -> Pipeline:
    """Replace any GridSearchCV steps with their best_estimator_ for fast bootstrap."""
    from sklearn.model_selection import GridSearchCV
    return Pipeline([
        (name, clone(step.best_estimator_) if isinstance(step, GridSearchCV) else clone(step))
        for name, step in fitted_pipe.steps
    ])


def bootstrap_predict(pipe, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame) -> dict:
    """
    Bootstrap resample training data N_BOOTSTRAP times, refit pipe, predict X_test.
    Returns mean, 2.5th and 97.5th percentile predictions.
    """
    rng  = np.random.default_rng(SEED)
    n    = len(X_train)
    preds = np.zeros((N_BOOTSTRAP, len(X_test)))
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        m   = clone(pipe)
        m.fit(X_train.iloc[idx].reset_index(drop=True), y_train.values[idx])
        preds[i] = m.predict(X_test)
    return dict(
        mean  = preds.mean(axis=0),
        ci_lo = np.percentile(preds, 2.5,  axis=0),
        ci_hi = np.percentile(preds, 97.5, axis=0),
        std   = preds.std(axis=0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLINVAR CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_clinvar(clinvar_df: pd.DataFrame, bs: dict,
                     threshold: float) -> pd.DataFrame:
    """Return per-variant prediction DataFrame and print classification metrics."""
    y_true = (clinvar_df["clinvar_label"] == "pathogenic").astype(int).values
    y_mean = bs["mean"]

    auroc = roc_auc_score(y_true, -y_mean)
    auprc = average_precision_score(y_true, -y_mean)
    y_pred = (y_mean < threshold).astype(int)
    acc     = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  ── ClinVar classification ──────────────────────────────────")
    print(f"     AUROC              : {auroc:.3f}")
    print(f"     AUPRC              : {auprc:.3f}")
    print(f"     Accuracy           : {acc:.3f}  ({int(acc*20)}/20 correct)")
    print(f"     Balanced accuracy  : {bal_acc:.3f}")
    print(f"     Sensitivity (path) : {sens:.3f}  ({tp}/{tp+fn} pathogenic)")
    print(f"     Specificity (ben)  : {spec:.3f}  ({tn}/{tn+fp} benign)")
    print(f"     Threshold          : {threshold:.4f}  (training-set median)")

    detail = pd.DataFrame({
        "variant":       clinvar_df["variant"].values,
        "clinvar_label": clinvar_df["clinvar_label"].values,
        "am_pathogenicity": clinvar_df["am_pathogenicity"].values,
        "pred_score":    y_mean.round(4),
        "ci_lo":         bs["ci_lo"].round(4),
        "ci_hi":         bs["ci_hi"].round(4),
        "pred_class":    np.where(y_pred == 1, "pathogenic", "benign"),
        "correct":       (y_pred == y_true),
        "auroc":         auroc,
        "accuracy":      acc,
        "balanced_accuracy": bal_acc,
        "sensitivity":   sens,
        "specificity":   spec,
    }).sort_values("pred_score")

    return detail, dict(auroc=auroc, auprc=auprc, accuracy=acc,
                        balanced_accuracy=bal_acc, sensitivity=sens,
                        specificity=spec, threshold=threshold)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictions(detail: pd.DataFrame, threshold: float,
                     clf_metrics: dict, model: str,
                     label_strategy: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = {"pathogenic": "#E15759", "benign": "#4E79A7"}
    markers = {"pathogenic": "^",       "benign": "o"}
    for _, row in detail.sort_values(["clinvar_label", "pred_score"]).iterrows():
        c = colors[row["clinvar_label"]]
        ax.errorbar(
            row["variant"], row["pred_score"],
            yerr=[[row["pred_score"] - row["ci_lo"]],
                  [row["ci_hi"]     - row["pred_score"]]],
            fmt=markers[row["clinvar_label"]], color=c, capsize=4,
            markersize=9, alpha=1.0 if row["correct"] else 0.35, linewidth=1.5,
        )
        if not row["correct"]:
            ax.annotate("✗", (row["variant"], row["pred_score"]),
                        ha="center", va="bottom", fontsize=10, color="#333")
    ax.axhline(threshold, color="#555", linestyle="--", linewidth=1.3,
               label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Predicted MAVE score  (± 95 % bootstrap CI)")
    ax.set_title(
        f"ClinVar hold-out predictions  |  model: {model}  |  label: {label_strategy}\n"
        f"AUROC = {clf_metrics['auroc']:.3f}   "
        f"Accuracy = {int(clf_metrics['accuracy']*20)}/20   "
        f"Sensitivity = {clf_metrics['sensitivity']:.3f}   "
        f"Specificity = {clf_metrics['specificity']:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E15759", label="Pathogenic (ClinVar)"),
        Patch(color="#4E79A7", label="Benign (ClinVar)"),
        plt.Line2D([0], [0], color="#555", linestyle="--", label="Classification threshold"),
    ], fontsize=9, loc="lower right")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feats",   default=FEATS_CSV)
    p.add_argument("--mave",    default=MAVE_CSV)
    p.add_argument("--outdir",  default=OUTDIR)
    p.add_argument("--model",   default=MODEL,   choices=["rf", "ridge"])
    p.add_argument("--labels",  default=LABEL_STRATEGY,
                   choices=["both", "any", "cisplatin", "hdr"])
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, dest="n_bootstrap")
    p.add_argument("--threshold",   type=float, default=None)
    p.add_argument("--save-model",  action="store_true", dest="save_model")
    p.add_argument("--load-model",  action="store_true", dest="load_model")
    p.add_argument("--model-file",  default=MODEL_FILE, dest="model_file")
    p.add_argument("--seed",        type=int, default=SEED)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Propagate CLI overrides to module-level constants used by helpers
    global N_BOOTSTRAP, SEED
    N_BOOTSTRAP = args.n_bootstrap
    SEED        = args.seed

    print(f"\n{'═'*60}")
    print(f"  BRCA1 ClinVar Predictor")
    print(f"{'═'*60}")
    print(f"  model          : {args.model}")
    print(f"  label strategy : {args.labels}")
    print(f"  n_bootstrap    : {N_BOOTSTRAP}")
    print(f"  outdir         : {outdir}")

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("\n  Loading data …")
    merged = load_and_merge(args.feats, args.mave)
    merged, features = add_features(merged)

    # ── Split: training set vs ClinVar hold-out ───────────────────────────────
    clinvar_df = merged[merged["variant"].isin(CLINVAR_ALL)].copy()
    clinvar_df["clinvar_label"] = clinvar_df["variant"].apply(
        lambda v: "pathogenic" if v in CLINVAR_PATHOGENIC else "benign"
    )
    missing = [v for v in CLINVAR_ALL if v not in clinvar_df["variant"].values]
    if missing:
        print(f"  WARNING: these ClinVar variants not found: {missing}", file=sys.stderr)

    labelled     = build_label_set(merged, args.labels)
    train_df     = labelled[~labelled["variant"].isin(CLINVAR_ALL)]
    X_train      = train_df[features]
    y_train      = train_df["label"]
    train_median = float(y_train.median())
    threshold    = args.threshold if args.threshold is not None else train_median

    print(f"\n  Training set    : {len(train_df)} variants  (label: {args.labels})")
    print(f"  ClinVar test set: {len(clinvar_df)} variants  "
          f"({(clinvar_df['clinvar_label']=='pathogenic').sum()} path / "
          f"{(clinvar_df['clinvar_label']=='benign').sum()} benign)")
    print(f"  Features used   : {len(features)}")
    print(f"  Threshold       : {threshold:.4f}  (training-set median)")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipe = build_pipeline(args.model)

    # ── Train (or load) ───────────────────────────────────────────────────────
    if args.load_model:
        model_path = Path(args.model_file)
        if not model_path.exists():
            print(f"  Model file not found: {model_path}. Training from scratch.")
            args.load_model = False
        else:
            print(f"\n  Loading saved model from {model_path} …")
            pipe = joblib.load(model_path)

    if not args.load_model:
        print(f"\n  Running {N_FOLDS}-fold × {N_REPEATS}-repeat CV …")
        cv = cross_validate(pipe, X_train, y_train)
        sp_m, sp_s = np.nanmean(cv["spearman"]), np.nanstd(cv["spearman"])
        r2_m, r2_s = np.nanmean(cv["r2"]),       np.nanstd(cv["r2"])
        pe_m       = np.nanmean(cv["pearson"])
        rm_m       = cv["rmse"].mean()
        print(f"\n  CV results ({args.labels} label, n={len(train_df)}):")
        print(f"    Spearman ρ : {sp_m:+.3f} ± {sp_s:.3f}")
        print(f"    R²         : {r2_m:+.3f} ± {r2_s:.3f}")
        print(f"    Pearson r  : {pe_m:+.3f}")
        print(f"    RMSE       : {rm_m:.3f}")
        print(f"    AUROC (OOF): {cv['auroc']:.3f}")

        # Fit final model on full training set
        print("\n  Fitting final model on full training set …")
        pipe.fit(X_train, y_train)

    if args.save_model:
        joblib.dump(pipe, args.model_file)
        print(f"  Model saved → {args.model_file}")

    # ── Bootstrap CIs on ClinVar test set ────────────────────────────────────
    X_cv = clinvar_df[features]
    print(f"\n  Computing {N_BOOTSTRAP} bootstrap predictions for ClinVar variants …",
          end=" ", flush=True)
    bs = bootstrap_predict(pipe, X_train, y_train, X_cv)
    print("done")

    # ── Classify & report ─────────────────────────────────────────────────────
    detail, clf_metrics = classify_clinvar(clinvar_df, bs, threshold)

    # ── Save outputs ──────────────────────────────────────────────────────────
    pred_csv = outdir / "clinvar_predictions.csv"
    detail.to_csv(pred_csv, index=False)
    print(f"\n  Saved → {pred_csv}")

    metrics_rows = []
    if not args.load_model:
        metrics_rows += [
            ("model",          args.model),
            ("label_strategy", args.labels),
            ("n_train",        len(train_df)),
            ("n_cv_folds",     N_FOLDS),
            ("n_cv_repeats",   N_REPEATS),
            ("cv_spearman_mean", round(float(sp_m), 4)),
            ("cv_spearman_std",  round(float(sp_s), 4)),
            ("cv_r2_mean",       round(float(r2_m), 4)),
            ("cv_r2_std",        round(float(r2_s), 4)),
            ("cv_pearson_mean",  round(float(pe_m), 4)),
            ("cv_rmse_mean",     round(float(rm_m), 4)),
            ("cv_auroc",         round(float(cv["auroc"]), 4)),
        ]
    metrics_rows += [
        ("clinvar_threshold",         round(threshold, 4)),
        ("clinvar_auroc",             round(clf_metrics["auroc"], 4)),
        ("clinvar_auprc",             round(clf_metrics["auprc"], 4)),
        ("clinvar_accuracy",          round(clf_metrics["accuracy"], 4)),
        ("clinvar_balanced_accuracy", round(clf_metrics["balanced_accuracy"], 4)),
        ("clinvar_sensitivity",       round(clf_metrics["sensitivity"], 4)),
        ("clinvar_specificity",       round(clf_metrics["specificity"], 4)),
        ("clinvar_n_correct",         int(clf_metrics["accuracy"] * 20)),
        ("clinvar_n_total",           20),
    ]
    metrics_df = pd.DataFrame(metrics_rows, columns=["metric", "value"])
    metrics_csv = outdir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  Saved → {metrics_csv}")

    plot_predictions(detail, threshold, clf_metrics,
                     args.model, args.labels, outdir / "clinvar_plot.png")

    print(f"\n{'═'*60}")
    print(f"  Done.  Results in: {outdir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
