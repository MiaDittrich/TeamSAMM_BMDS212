"""
prune_features.py
=================
Take the full `brca1_model_features.csv` produced by Stage 3 of run_all.py
and drop redundant / low-signal columns, writing the model-ready lean CSV.

Drop policy
-----------
  A. Admin / near-constant / duplicate columns
       n_alignment_anchors, domain_c_term, domain_RING, domain_central,
       shell_n_5A, shell_n_8A, shell_n_12A,
       n_displaced_upstream, n_displaced_downstream, n_displaced_total
  B. EvoEF2 — drop everything except the two summary ΔΔG numbers
       keep:  evoef2_ddg_Total  (Stage 2: ComputeStability variant − WT)
              ddg_evoef2        (Stage 3: BuildMutant + ComputeStability)
  C. Dihedrals — drop ALL phi/psi-derived columns
       keep only ramachandran_violation as a binary window summary

Final kept set (≈22 columns)
----------------------------
    variant, mutant_residue, mutant_plddt
    ca_rmsd, backbone_rmsd, mutant_ca_displacement
    shell_rmsd_5A, shell_rmsd_8A, shell_rmsd_12A
    ramachandran_violation
    pam250_score, delta_hydrophobicity, delta_size, delta_charge, delta_aromaticity
    is_charge_reversal, is_size_increase, is_hydrophobic_to_polar, is_polar_to_hydrophobic
    evoef2_ddg_Total, ddg_evoef2
    is_disordered_variant

Usage
-----
    python3 prune_features.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent
INPUT_CSV  = ROOT / "brca1_model_features.csv"
OUTPUT_CSV = ROOT / "brca1_model_features_lean.csv"


# Admin / near-constant / duplicate columns
EXPLICIT_DROPS = {
    "n_alignment_anchors",
    "domain_c_term", "domain_RING", "domain_central",
    "shell_n_5A", "shell_n_8A", "shell_n_12A",
    "n_displaced_upstream", "n_displaced_downstream", "n_displaced_total",
}

# Only these phi/psi-derived columns survive pruning.
DIHEDRAL_KEEP = {"ramachandran_violation"}

# Matches every raw, sin/cos, wt_, or delta_ phi/psi column.
_DIHEDRAL_RE = re.compile(
    r"^(sin_|cos_)?(wt_)?(delta_)?(phi|psi)(_[+-]\d+)?$"
)

# Only summary EvoEF2 ΔΔG numbers survive pruning.
EVOEF2_KEEP = {"evoef2_ddg_Total", "ddg_evoef2"}


def build_drop_set(columns: list[str]) -> set[str]:
    """Return the set of columns to drop given the input column list."""
    to_drop: set[str] = set(EXPLICIT_DROPS) & set(columns)

    for c in columns:
        if (c.startswith("evoef2_") or c == "ddg_evoef2") and c not in EVOEF2_KEEP:
            to_drop.add(c)
        if _DIHEDRAL_RE.match(c) and c not in DIHEDRAL_KEEP:
            to_drop.add(c)

    return to_drop


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"[ERR] Input CSV not found: {INPUT_CSV}")
        print("      Run run_all.py first.")
        return 1

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV.name}  shape = {df.shape}")

    drop = build_drop_set(df.columns.tolist())
    kept = [c for c in df.columns if c not in drop]
    df_lean = df[kept]
    df_lean.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDropped {len(drop)} columns")
    print(f"Kept    {len(kept)} columns")
    print(f"Saved → {OUTPUT_CSV.name}  shape = {df_lean.shape}\n")

    groups = {
        "identity":       [c for c in kept if c in {"variant","mutant_residue"}],
        "confidence":     [c for c in kept if "plddt" in c.lower()],
        "structural":     [c for c in kept if any(k in c for k in
                          ("ca_rmsd","backbone_rmsd","mutant_ca_displacement"))],
        "shell RMSDs":    [c for c in kept if c.startswith("shell_")],
        "dihedral":       [c for c in kept if _DIHEDRAL_RE.match(c) or c == "ramachandran_violation"],
        "biochemical":    [c for c in kept if c.startswith(("pam250","delta_hydro","delta_size",
                                                            "delta_charge","delta_arom","is_charge",
                                                            "is_size","is_hydro","is_polar"))],
        "ΔΔG (EvoEF2)":   [c for c in kept if c.startswith("evoef2_ddg_") or c == "ddg_evoef2"],
        "flags":          [c for c in kept if c == "is_disordered_variant"],
    }
    leftover = set(kept) - {c for grp in groups.values() for c in grp}
    if leftover:
        groups["other"] = sorted(leftover)

    print("Kept columns by group:")
    for g, cols in groups.items():
        if cols:
            print(f"  {g:<20s} {len(cols):>3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
