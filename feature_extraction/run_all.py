"""
run_all.py
==========
One-command driver for the full BRCA1 variant feature pipeline.

Stages
------
  1. brca1_pipeline.py    →  brca1_features.csv
       Parse every CIF in pdb_s/, compute structural + biochemical
       features against the wildtype reference.

  2. evoef2_stability.py  →  brca1_features_with_evoef2.csv
       Run EvoEF2 ComputeStability on every variant + WT, write each
       per-term energy + ΔΔG vs WT, merge into Stage-1 output.

  3. feature_engineering  →  brca1_model_features.csv
       Annotate disordered variants, run EvoEF2 BuildMutant +
       ComputeStability for the summary ΔΔG, drop near-zero-variance
       columns, encode dihedrals as sin/cos.

  4. prune_features.py    →  brca1_model_features_lean.csv
       Final aggressive pruning to the modelling-ready feature set.

Usage
-----
  python3 run_all.py            # requires biopython, numpy, pandas
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Config ────────────────────────────────────────────────────────────────
CIF_FOLDER         = ROOT / "pdb_s"
WT_CIF             = ROOT / "fold_brca_wt_v2_model_0.cif"
EVOEF2_BIN         = ROOT / "EvoEF2" / "EvoEF2"
# EvoEF2 only reads PDB; Stage 2 converts every CIF (incl. WT) to PDB in
# pdb_s_converted/, and Stage 3 reuses the converted WT.
WT_PDB_CONVERTED   = ROOT / "pdb_s_converted" / "fold_brca_wt_v2_model_0.pdb"

# Per-stage outputs (each stage reads the previous stage's output)
STAGE1_CSV         = ROOT / "brca1_features.csv"
STAGE2_CSV         = ROOT / "brca1_features_with_evoef2.csv"
FINAL_CSV          = ROOT / "brca1_model_features.csv"
LEAN_CSV           = ROOT / "brca1_model_features_lean.csv"


def banner(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def main() -> int:
    # Pre-flight sanity checks
    for p, name in [
        (CIF_FOLDER, "pdb_s/ folder"),
        (WT_CIF,     "wildtype CIF"),
        (EVOEF2_BIN, "compiled EvoEF2 binary"),
    ]:
        if not p.exists():
            print(f"[ERR] Missing {name}: {p}")
            return 1

    # ── Stage 1: structural + biochemical features ────────────────────────
    banner("STAGE 1 — brca1_pipeline.py (structural + biochemical features)")
    from cif_loader import run_pipeline
    from brca1_pipeline import DOMAINS

    run_pipeline(
        cif_folder      = str(CIF_FOLDER),
        wt_pdb          = str(WT_CIF),
        output_csv      = str(STAGE1_CSV),
        plddt_threshold = 70.0,
        renumber_offset = 0,
        domain_range    = DOMAINS["c_term"],
    )

    # ── Stage 2: per-term EvoEF2 ComputeStability + ΔΔG ──────────────────
    banner("STAGE 2 — evoef2_stability.py (per-term EvoEF2 energies + ΔΔG)")
    rc = subprocess.call([sys.executable, str(ROOT / "evoef2_stability.py")])
    if rc != 0:
        print(f"[ERR] evoef2_stability.py exited with code {rc}")
        return rc
    if not STAGE2_CSV.exists():
        print(f"[ERR] Stage 2 did not produce {STAGE2_CSV}")
        return 1

    # ── Stage 3: clean + summary EvoEF2 ΔΔG ──────────────────────────────
    banner("STAGE 3 — feature_engineering.py (clean + EvoEF2 ΔΔG)")
    from feature_engineering import run_feature_engineering

    if not WT_PDB_CONVERTED.exists():
        print(f"[ERR] Expected converted WT PDB not found at {WT_PDB_CONVERTED}")
        return 1
    run_feature_engineering(
        raw_csv    = str(STAGE2_CSV),
        output_csv = str(FINAL_CSV),
        wt_pdb     = str(WT_PDB_CONVERTED),
        evoef2_bin = str(EVOEF2_BIN),
    )

    # ── Stage 4: prune redundant columns ─────────────────────────────────
    banner("STAGE 4 — prune_features.py (drop redundant/low-signal columns)")
    rc = subprocess.call([sys.executable, str(ROOT / "prune_features.py")])
    if rc != 0:
        print(f"[ERR] prune_features.py exited with code {rc}")
        return rc

    # ── Done ──────────────────────────────────────────────────────────────
    banner("DONE")
    print(f"  Lean CSV  → {LEAN_CSV}  (recommended for modelling)")
    print(f"  Full CSV  → {FINAL_CSV}  (everything, for inspection)")
    print(f"  Intermediates kept:")
    print(f"    {STAGE1_CSV.name}")
    print(f"    {STAGE2_CSV.name}")
    print(f"    evoef2_energies.csv")
    print(f"    pdb_s_converted/")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
