"""
feature_engineering.py
=======================
Post-processes the raw brca1_features.csv produced by brca1_pipeline.py
into a model-ready feature matrix.

Steps
-----
  1.  Annotate disordered variants (is_disordered_variant flag) — no data
      modification, all rows kept.
  2.  ΔΔG — run EvoEF2 BuildMutant + ComputeStability for every variant
      (Total energy at WT − variant). Requires the EvoEF2 binary.
  3.  Drop zero-variance / redundant columns.
  4.  Encode any remaining raw φ/ψ columns as sin/cos.
  5.  Write the final CSV and print a summary.

Usage
-----
  python3 feature_engineering.py brca1_features.csv \
      --wt-pdb fold_brca_wt_v2_model_0.pdb \
      --evoef2-bin /path/to/EvoEF2/EvoEF2 \
      --output brca1_model_features.csv

Dependencies
------------
  pip install biopython numpy pandas
  EvoEF2 binary: github.com/tommyhuangthu/EvoEF2
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DROP ZERO-VARIANCE AND REDUNDANT FEATURES
# ═══════════════════════════════════════════════════════════════════════════

# These are always dropped regardless of variance:
#   wt_global_plddt  — constant (same WT structure every row)
#   var_global_plddt — near-constant; dominated by disordered linker noise
#   domain_RING, domain_central — always 0 for c_term variants
#   wt_aa, mut_aa    — string labels already encoded biochemically
#   wt_/mut_ absolute biochemical properties — only the deltas / flags are
#       needed to avoid multicollinearity.
_ALWAYS_DROP = {
    "wt_global_plddt",
    "var_global_plddt",
    "domain_RING",
    "domain_central",
    "wt_aa",
    "mut_aa",
    "wt_charge",
    "mut_charge",
    "wt_aromatic",
    "mut_aromatic",
    "wt_size",
    "mut_size",
    "wt_hydrophobicity",
    "mut_hydrophobicity",
}

# Raw angle columns (degrees) — replaced by sin/cos and delta versions.
# If the new pipeline was used these will already be accompanied by
# sin_phi_+0, cos_phi_+0, delta_phi_+0 etc.; the raw columns are then
# redundant for modelling and are dropped. Set KEEP_RAW_ANGLES=True below
# if you want them retained for interpretability.
KEEP_RAW_ANGLES = False


def drop_low_signal_features(
    df: pd.DataFrame,
    variance_threshold: float = 1e-6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove:
      1. Columns in _ALWAYS_DROP
      2. Any numeric column with variance below variance_threshold
      3. Raw phi/psi degree columns when sin/cos versions exist (if KEEP_RAW_ANGLES=False)
      4. Identity / metadata columns (variant, mutant_residue, error) are preserved

    Parameters
    ----------
    variance_threshold : columns whose variance across the dataset falls below
                         this threshold are dropped automatically.  The default
                         catches near-constant columns like var_global_plddt
                         (stdev ≈ 0.30 → var ≈ 0.09, well above threshold), but
                         will catch truly flat columns.  Lower to 0 to only drop
                         perfectly constant columns.
    """
    metadata_cols = {"variant", "mutant_residue", "error", "label", "pathogenicity"}
    df_out = df.copy()
    dropped = []

    # 1. Always-drop list
    for col in _ALWAYS_DROP:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)
            dropped.append(col)

    # 2. Zero/near-zero variance
    numeric = df_out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric:
        if col in metadata_cols:
            continue
        if df_out[col].var(skipna=True) < variance_threshold:
            df_out.drop(columns=[col], inplace=True)
            dropped.append(col)

    # 3. Raw angle columns when sin/cos versions are present
    if not KEEP_RAW_ANGLES:
        raw_angle_pat = re.compile(r"^(wt_)?(phi|psi)_[+-]\d$")
        for col in list(df_out.columns):
            if col in metadata_cols:
                continue
            if raw_angle_pat.match(col):
                # check that a sin_ version exists
                sin_col = "sin_" + col
                if sin_col in df_out.columns:
                    df_out.drop(columns=[col], inplace=True)
                    dropped.append(col)

    if verbose and dropped:
        print(f"  Dropped {len(dropped)} features: {dropped[:10]}"
              f"{'...' if len(dropped) > 10 else ''}")

    return df_out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — ΔΔG VIA EVOEF2
# ═══════════════════════════════════════════════════════════════════════════

def _write_evoef2_mutfile(variants: list[tuple[str, int, str]], path: Path,
                          chain: str = "A") -> None:
    """
    Write an EvoEF2 mutation file.
    EvoEF2 format: <WT><chain><resnum><MUT>;  e.g. VA1838E;
    """
    with open(path, "w") as f:
        for wt_aa, resnum, mut_aa in variants:
            f.write(f"{wt_aa}{chain}{resnum}{mut_aa};\n")


def compute_evoef2_ddg(
    df: pd.DataFrame,
    wt_pdb: str,
    evoef2_bin: str = "EvoEF2",
    chain: str = "A",
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute per-variant ΔΔG with EvoEF2 (BuildMutant + ComputeStability).
    Source: github.com/tommyhuangthu/EvoEF2

    Parameters
    ----------
    df         : feature DataFrame (must have wt_aa, mut_aa, mutant_residue)
    wt_pdb     : path to wildtype PDB (EvoEF2 does not accept mmCIF; convert first)
    evoef2_bin : path to EvoEF2 executable
    chain      : chain ID in the wildtype structure

    Returns
    -------
    df with added column 'ddg_evoef2' (kcal/mol; positive = destabilising)
    """
    if not Path(evoef2_bin).exists() and shutil.which(evoef2_bin) is None:
        if verbose:
            print(f"  [WARN] EvoEF2 binary not found at '{evoef2_bin}' — skipping")
        df["ddg_evoef2"] = np.nan
        return df

    evoef2_abs = str(Path(evoef2_bin).resolve())
    wt_path = Path(wt_pdb)
    if wt_path.suffix.lower() not in (".pdb",):
        if verbose:
            print(f"  [WARN] EvoEF2 cannot read {wt_path.suffix} — pass a .pdb. Skipping.")
        df["ddg_evoef2"] = np.nan
        return df

    tmpdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="evoef2_"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    wt_copy = tmpdir / wt_path.name
    shutil.copy(wt_path, wt_copy)

    def _run_stability(pdb_name: str) -> float:
        """Run EvoEF2 ComputeStability on pdb_name (relative to tmpdir); return Total."""
        r = subprocess.run(
            [evoef2_abs, "--command=ComputeStability", f"--pdb={pdb_name}"],
            cwd=tmpdir, capture_output=True, text=True, timeout=120,
        )
        for line in r.stdout.splitlines():
            if line.strip().startswith("Total"):
                m = re.search(r"Total\s*=\s*([-\d.]+)", line)
                if m:
                    return float(m.group(1))
        return float("nan")

    ddg_values = []
    try:
        required = {"wt_aa", "mut_aa", "mutant_residue"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame missing columns: {required - set(df.columns)}")

        # Compute WT energy once (reused for every variant)
        e_wt = _run_stability(wt_copy.name)
        if np.isnan(e_wt):
            raise RuntimeError("Could not parse WT total energy from EvoEF2 output")
        if verbose:
            print(f"  WT total energy = {e_wt:.2f}")

        for _, row in df.iterrows():
            wt_aa  = row["wt_aa"]
            mut_aa = row["mut_aa"]
            resnum = int(row["mutant_residue"])

            mutfile = tmpdir / f"mut_{wt_aa}{resnum}{mut_aa}.txt"
            _write_evoef2_mutfile([(wt_aa, resnum, mut_aa)], mutfile, chain=chain)

            # BuildMutant writes <wt_stem>_Model_0001.pdb in cwd
            build = subprocess.run(
                [evoef2_abs, "--command=BuildMutant",
                 f"--pdb={wt_copy.name}", f"--mutant_file={mutfile.name}"],
                cwd=tmpdir, capture_output=True, text=True, timeout=300,
            )
            mut_pdb = tmpdir / f"{wt_copy.stem}_Model_0001.pdb"
            if build.returncode != 0 or not mut_pdb.exists():
                ddg_values.append(float("nan"))
                continue

            e_mut = _run_stability(mut_pdb.name)
            ddg_values.append(e_mut - e_wt if not np.isnan(e_mut) else float("nan"))

            # Clean per-iteration output to avoid stale files biasing later rows
            mut_pdb.unlink(missing_ok=True)

        df = df.copy()
        df["ddg_evoef2"] = ddg_values

        if verbose:
            valid = [v for v in ddg_values if not np.isnan(v)]
            if valid:
                print(f"  EvoEF2 ΔΔG: {len(valid)}/{len(ddg_values)} succeeded  "
                      f"mean={np.mean(valid):.2f}  "
                      f"range=[{np.min(valid):.2f}, {np.max(valid):.2f}]")
            else:
                print("  EvoEF2: no ΔΔG values parsed")

    except Exception as e:
        if verbose:
            print(f"  [WARN] EvoEF2 failed: {e} — ddg_evoef2 set to NaN")
        df = df.copy()
        df["ddg_evoef2"] = np.nan

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — SIN/COS ENCODING FOR REMAINING RAW DIHEDRAL COLUMNS
# ═══════════════════════════════════════════════════════════════════════════

def encode_dihedrals_sincos(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    For any raw phi/psi degree column that does not already have a sin_ version,
    add sin/cos encoded copies.

    This step is a safety net for CSVs produced by older versions of the
    pipeline that lack the sin/cos columns.  If the new pipeline was used,
    all sin/cos columns will already exist and this function is a no-op.
    """
    angle_pat = re.compile(r"^(phi|psi|wt_phi|wt_psi|delta_phi|delta_psi)_[+-]\d$")
    added = []
    df = df.copy()

    for col in list(df.columns):
        if not angle_pat.match(col):
            continue
        sin_col = f"sin_{col}"
        cos_col = f"cos_{col}"
        if sin_col not in df.columns:
            df[sin_col] = np.sin(np.radians(pd.to_numeric(df[col], errors="coerce")))
            added.append(sin_col)
        if cos_col not in df.columns:
            df[cos_col] = np.cos(np.radians(pd.to_numeric(df[col], errors="coerce")))
            added.append(cos_col)

    if verbose and added:
        print(f"  Sin/cos encoding added for {len(added)//2} angle columns")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# FINAL FEATURE MATRIX SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def summarise_features(df: pd.DataFrame) -> None:
    """Print a compact summary of the final feature matrix."""
    metadata = {"variant", "mutant_residue", "error"}
    feature_cols = [c for c in df.columns if c not in metadata
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    print(f"\n{'─'*60}")
    print(f"  Final feature matrix")
    print(f"  Variants  : {len(df)}")
    print(f"  Features  : {len(feature_cols)}")

    nan_frac = df[feature_cols].isna().mean().sort_values(ascending=False)
    high_nan = nan_frac[nan_frac > 0.1]
    if not high_nan.empty:
        print(f"\n  Features with >10% missing values:")
        for col, frac in high_nan.items():
            print(f"    {col:<45s} {frac*100:.0f}% missing")
    else:
        print(f"  No features with >10% missing values ✓")

    print(f"\n  Feature groups present:")
    groups = {
        "structural (RMSD/displacement)":  [c for c in feature_cols if any(k in c for k in ("rmsd","displaced","displacement"))],
        "shell RMSDs":                      [c for c in feature_cols if "shell" in c],
        "dihedral (sin/cos)":               [c for c in feature_cols if "sin_" in c or "cos_" in c],
        "dihedral (Δ)":                     [c for c in feature_cols if "delta_phi" in c or "delta_psi" in c],
        "biochemical":                      [c for c in feature_cols if any(k in c for k in ("pam250","delta_hydro","delta_size","delta_charge","is_charge","is_size","is_hydro","is_polar"))],
        "ΔΔG":                              [c for c in feature_cols if "ddg" in c],
        "pLDDT / anchors":                  [c for c in feature_cols if any(k in c for k in ("plddt","anchors"))],
    }
    for group, cols in groups.items():
        if cols:
            print(f"    {group:<40s} {len(cols)} features")

    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_engineering(
    raw_csv: str,
    output_csv: str,
    wt_pdb: str,
    evoef2_bin: str,
    chain: str = "A",
    workdir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Annotate, compute EvoEF2 ΔΔG, drop low-signal columns, encode dihedrals.

    Parameters
    ----------
    raw_csv     : path to brca1_features_with_evoef2.csv (stage-2 output)
    output_csv  : path for the cleaned feature matrix
    wt_pdb      : wildtype PDB (EvoEF2 cannot read mmCIF — pass a .pdb)
    evoef2_bin  : path to compiled EvoEF2 executable
    chain       : chain ID in the wildtype structure (default "A")
    workdir     : temp directory for EvoEF2 runs (auto-created if None)
    """
    print(f"\n{'═'*60}")
    print(f"  Feature engineering pipeline")
    print(f"  Input : {raw_csv}")
    print(f"  Output: {output_csv}")
    print(f"{'═'*60}")

    df = pd.read_csv(raw_csv)
    print(f"\n  Loaded {len(df)} variants × {len(df.columns)} raw columns")

    # Step 1 — annotate disordered variants (no data modification)
    df["is_disordered_variant"] = (df["domain_c_term"] == 0).astype(int)

    # Step 2 — EvoEF2 BuildMutant + ComputeStability ΔΔG.
    # Must run before drop_low_signal so wt_aa/mut_aa columns survive.
    print("\n[2] EvoEF2 ΔΔG computation")
    df = compute_evoef2_ddg(
        df, wt_pdb, evoef2_bin=evoef2_bin,
        chain=chain, workdir=workdir, verbose=verbose,
    )

    # Step 3 — drop zero-variance / redundant features
    print("\n[3] Dropping low-signal features")
    df = drop_low_signal_features(df, verbose=verbose)

    # Step 4 — ensure sin/cos encoding (no-op if pipeline already supplied them)
    print("\n[4] Dihedral sin/cos encoding")
    df = encode_dihedrals_sincos(df, verbose=verbose)

    df.to_csv(output_csv, index=False)
    print(f"\n  Saved → {output_csv}")

    summarise_features(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Post-process brca1_features*.csv into a model-ready feature matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("raw_csv",      help="Raw features CSV from brca1_pipeline.py")
    p.add_argument("--output",     default="brca1_model_features.csv",
                   help="Output path for cleaned feature matrix")
    p.add_argument("--wt-pdb",     required=True,
                   help="Wildtype PDB path (EvoEF2 requires .pdb, not .cif)")
    p.add_argument("--evoef2-bin", default=os.environ.get("EVOEF2_BIN"),
                   help="Path to EvoEF2 executable (or set EVOEF2_BIN env var)")
    p.add_argument("--chain",      default="A", help="Chain ID (default: A)")
    p.add_argument("--workdir",    default=None, help="Temp directory for EvoEF2")

    args = p.parse_args()
    run_feature_engineering(
        raw_csv    = args.raw_csv,
        output_csv = args.output,
        wt_pdb     = args.wt_pdb,
        evoef2_bin = args.evoef2_bin,
        chain      = args.chain,
        workdir    = args.workdir,
    )


if __name__ == "__main__":
    _cli()
