"""
evoef2_stability.py
===================
Run EvoEF2 `ComputeStability` on every CIF in `pdb_s/` (plus the wildtype),
parse the per-term energies, and merge them into `brca1_features.csv` as
new columns, including a ΔΔG = E(variant) − E(wildtype) for the total
and for every individual energy term.

Output
------
  pdb_s_converted/<stem>.pdb          - PDB conversions of each CIF
  evoef2_energies.csv                 - one row per variant, all EvoEF2 terms
  brca1_features_with_evoef2.csv      - brca1_features.csv with EvoEF2
                                        columns appended (prefixed `evoef2_`)

Usage
-----
  python3 evoef2_stability.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO

# ── Paths (edit if you move things) ────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
CIF_FOLDER    = ROOT / "pdb_s"
WT_CIF        = ROOT / "fold_brca_wt_v2_model_0.cif"
EVOEF2_DIR    = ROOT / "EvoEF2"
EVOEF2_BIN    = EVOEF2_DIR / "EvoEF2"
CONVERT_DIR   = ROOT / "pdb_s_converted"
FEATURES_CSV  = ROOT / "brca1_features.csv"
ENERGIES_CSV  = ROOT / "evoef2_energies.csv"
MERGED_CSV    = ROOT / "brca1_features_with_evoef2.csv"

# Same filename-parsing regex used by cif_loader.py
THREE_TO_ONE: dict[str, str] = {
    "ala":"A","arg":"R","asn":"N","asp":"D","cys":"C",
    "gln":"Q","glu":"E","gly":"G","his":"H","ile":"I",
    "leu":"L","lys":"K","met":"M","phe":"F","pro":"P",
    "ser":"S","thr":"T","trp":"W","tyr":"Y","val":"V",
}
_MUTATION_RE = re.compile(r"brca1_([a-z]{3})(\d+)([a-z]{3})_model", re.IGNORECASE)


def cif_label(cif: Path) -> str | None:
    """Return the variant label (e.g. 'A1708E') or None for the WT file."""
    m = _MUTATION_RE.search(cif.name.lower())
    if not m:
        return None
    wt3, res, mut3 = m.group(1), m.group(2), m.group(3)
    wt, mut = THREE_TO_ONE.get(wt3), THREE_TO_ONE.get(mut3)
    if wt is None or mut is None:
        return None
    return f"{wt}{res}{mut}"


def cif_to_pdb(cif: Path, out_pdb: Path) -> None:
    """Convert a CIF file to a clean PDB readable by EvoEF2."""
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if out_pdb.exists():
        return
    struct = MMCIFParser(QUIET=True).get_structure(cif.stem, str(cif))
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_pdb))


# Parse lines like:  "interS_vdwatt         =            -5015.44"
_ENERGY_LINE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(-?\d+\.\d+)")


def run_compute_stability(pdb: Path) -> dict[str, float]:
    """
    Invoke `EvoEF2 --command=ComputeStability --pdb=...` and return a dict
    of {energy_term: value}, including a final 'Total' key.

    Must be run with EVOEF2_DIR as cwd so EvoEF2 can find its built-in
    parameter library via relative paths.
    """
    proc = subprocess.run(
        [str(EVOEF2_BIN),
         "--command=ComputeStability",
         f"--pdb={pdb.resolve()}"],
        cwd=str(EVOEF2_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"EvoEF2 failed on {pdb.name}: rc={proc.returncode}\n"
            f"stderr:\n{proc.stderr[-400:]}"
        )

    terms: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        m = _ENERGY_LINE.match(line.strip())
        if m:
            terms[m.group(1)] = float(m.group(2))
        elif line.strip().startswith("Total"):
            # "Total                 =             3945.45"
            parts = line.split("=")
            if len(parts) == 2:
                try:
                    terms["Total"] = float(parts[1].strip())
                except ValueError:
                    pass
    if "Total" not in terms:
        raise RuntimeError(f"Could not parse Total energy from EvoEF2 output for {pdb.name}")
    return terms


def main() -> int:
    if not EVOEF2_BIN.exists():
        print(f"[ERR] EvoEF2 binary not found at {EVOEF2_BIN}")
        return 1
    if not WT_CIF.exists():
        print(f"[ERR] Wildtype CIF not found at {WT_CIF}")
        return 1
    if not CIF_FOLDER.exists():
        print(f"[ERR] Variant folder not found at {CIF_FOLDER}")
        return 1

    # ── 1. Build the work list: WT + every parseable variant ──────────────
    work: list[tuple[str, Path]] = []  # (label, cif_path)
    work.append(("WT", WT_CIF))
    for cif in sorted(CIF_FOLDER.glob("fold_brca1_*_model_*.cif")):
        lbl = cif_label(cif)
        if lbl is None:
            print(f"  [skip] could not parse label from {cif.name}")
            continue
        work.append((lbl, cif))

    print(f"\nConverting {len(work)} CIFs → PDB and running EvoEF2 ComputeStability")
    print(f"  EvoEF2 dir : {EVOEF2_DIR}")
    print(f"  PDB outdir : {CONVERT_DIR}")
    print(f"  Energies   : {ENERGIES_CSV}\n")

    rows: list[dict] = []
    for i, (label, cif) in enumerate(work, 1):
        out_pdb = CONVERT_DIR / f"{cif.stem}.pdb"
        try:
            cif_to_pdb(cif, out_pdb)
        except Exception as exc:
            print(f"  [{i:>2d}/{len(work)}] [ERR convert] {label}: {exc}")
            continue
        try:
            terms = run_compute_stability(out_pdb)
        except Exception as exc:
            print(f"  [{i:>2d}/{len(work)}] [ERR EvoEF2] {label}: {exc}")
            continue
        row = {"variant": label, **terms}
        rows.append(row)
        print(f"  [{i:>2d}/{len(work)}] [OK] {label:<8s}  Total = {terms['Total']:>12.2f}")

    if not rows:
        print("No rows produced. Aborting.")
        return 1

    # ── 2. Build energy table and add ΔΔG columns ─────────────────────────
    df = pd.DataFrame(rows).set_index("variant")
    # Prefix every term with `evoef2_` to avoid collisions with brca1_features.csv
    df.columns = [f"evoef2_{c}" for c in df.columns]

    if "WT" in df.index:
        wt = df.loc["WT"]
        ddg = df.subtract(wt, axis=1)
        ddg.columns = [c.replace("evoef2_", "evoef2_ddg_") for c in ddg.columns]
        df = pd.concat([df, ddg], axis=1)
    else:
        print("[WARN] WT row missing — ΔΔG columns will not be created.")

    df.to_csv(ENERGIES_CSV)
    print(f"\nWrote {len(df)} rows × {df.shape[1]} cols → {ENERGIES_CSV}")

    # ── 3. Merge into brca1_features.csv ──────────────────────────────────
    if FEATURES_CSV.exists():
        feats = pd.read_csv(FEATURES_CSV)
        # Drop the WT row from the energies before merging (variants only).
        energies_for_merge = df.drop(index="WT", errors="ignore").reset_index()
        merged = feats.merge(energies_for_merge, on="variant", how="left")
        merged.to_csv(MERGED_CSV, index=False)
        n_matched = energies_for_merge["variant"].isin(feats["variant"]).sum()
        print(f"Merged with {FEATURES_CSV.name}: "
              f"{n_matched}/{len(energies_for_merge)} variants matched")
        print(f"  → {MERGED_CSV}")
    else:
        print(f"[WARN] {FEATURES_CSV} not found — skipped merge.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
