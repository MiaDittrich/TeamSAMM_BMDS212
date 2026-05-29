"""
cif_loader.py
=============
Parses AlphaFold2 CIF filenames of the form:

    fold_brca1_{wt_3letter}{residue}{mut_3letter}_model_0.cif

e.g.  fold_brca1_ala1708glu_model_0.cif
      fold_brca1_cys61gly_model_0.cif
      fold_brca1_met1775arg_model_0.cif

and builds the variant list expected by VariantFeatureExtractor.extract_batch().

Usage
-----
    from cif_loader import load_cif_folder

    variants = load_cif_folder("pdb_s/")
    # returns list of (path, wt_aa_1letter, mut_aa_1letter, residue_num, label)

    from brca1_pipeline import VariantFeatureExtractor, DOMAINS
    extractor = VariantFeatureExtractor(
        "fold_brca_wt_v2_model_0.cif",
        domain_range=DOMAINS["c_term"],
    )
    df = extractor.extract_batch(variants)
    df.to_csv("brca1_features.csv", index=False)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── Three-letter → one-letter amino acid mapping ───────────────────────────
THREE_TO_ONE: dict[str, str] = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}

# Regex: captures (wt_3letter)(residue_number)(mut_3letter)
_MUTATION_RE = re.compile(
    r"brca1_([a-z]{3})(\d+)([a-z]{3})_model",
    re.IGNORECASE,
)


def parse_cif_filename(path: Path) -> Optional[tuple[str, str, int, str]]:
    """
    Parse a single CIF filename.

    Returns
    -------
    (wt_aa_1letter, mut_aa_1letter, residue_num, label)  or  None if no match.

    Examples
    --------
    fold_brca1_ala1708glu_model_0.cif  →  ("A", "E", 1708, "A1708E")
    fold_brca1_cys61gly_model_0.cif    →  ("C", "G",   61, "C61G")
    """
    name = path.name.lower()
    m = _MUTATION_RE.search(name)
    if not m:
        return None

    wt_3, res_str, mut_3 = m.group(1), m.group(2), m.group(3)

    wt_aa  = THREE_TO_ONE.get(wt_3)
    mut_aa = THREE_TO_ONE.get(mut_3)

    if wt_aa is None or mut_aa is None:
        print(f"  [WARN] Unrecognised amino acid code in: {path.name}")
        return None

    res_num = int(res_str)
    label   = f"{wt_aa}{res_num}{mut_aa}"

    return wt_aa, mut_aa, res_num, label


def load_cif_folder(
    folder: str,
    pattern: str = "fold_brca1_*_model_*.cif",
    verbose: bool = True,
) -> list[tuple]:
    """
    Scan a folder for AlphaFold2 CIF files matching the naming convention
    and return a variant list ready for VariantFeatureExtractor.extract_batch().

    Parameters
    ----------
    folder  : path to the folder containing CIF files (e.g. "pdb_s/")
    pattern : glob pattern to match filenames
    verbose : print a summary of parsed and skipped files

    Returns
    -------
    List of tuples:
        (str path, str wt_aa, str mut_aa, int residue_num, str label)

    Sorted by residue number then mutation label for reproducibility.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    cif_files = sorted(folder_path.glob(pattern))
    if not cif_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {folder_path.resolve()}\n"
            f"Check the folder path and that files follow the naming convention."
        )

    variants = []
    skipped  = []

    for cif in cif_files:
        parsed = parse_cif_filename(cif)
        if parsed is None:
            skipped.append(cif.name)
            continue
        wt_aa, mut_aa, res_num, label = parsed
        variants.append((str(cif), wt_aa, mut_aa, res_num, label))

    # Sort by residue number
    variants.sort(key=lambda x: (x[3], x[4]))

    if verbose:
        print(f"\nCIF folder : {folder_path.resolve()}")
        print(f"Files found: {len(cif_files)}")
        print(f"Parsed OK  : {len(variants)}")
        if skipped:
            print(f"Skipped    : {len(skipped)}")
            for s in skipped:
                print(f"  • {s}")
        if variants:
            print(f"\nFirst 5 variants:")
            for path, wt, mut, res, lbl in variants[:5]:
                print(f"  {lbl:<12s}  res {res:<5d}  {wt}→{mut}  {Path(path).name}")
            if len(variants) > 5:
                print(f"  … and {len(variants) - 5} more")

    return variants


# ── Convenience: run the full pipeline from a folder in one call ────────────

def run_pipeline(
    cif_folder: str,
    wt_pdb: str,
    output_csv: str = "brca1_features.csv",
    plddt_threshold: float = 70.0,
    renumber_offset: int = 0,
    pattern: str = "fold_brca1_*_model_*.cif",
    domain_range: Optional[tuple[int, int]] = None,
) -> "pd.DataFrame":
    """
    One-call convenience wrapper:  folder of CIFs → feature CSV.

    Parameters
    ----------
    cif_folder      : path to folder containing AlphaFold2 CIF files
    wt_pdb          : path to wildtype PDB or CIF (AlphaFold2/3)
    output_csv      : where to save the feature matrix
    plddt_threshold : minimum pLDDT for alignment anchors
    renumber_offset : residue renumbering shift (0 if already UniProt-numbered)
    pattern         : glob pattern for CIF files
    domain_range    : optional (start, end) to restrict alignment and RMSD
                      computations to a folded domain, e.g. (1650, 1863).
                      Strongly recommended for largely disordered proteins.

    Returns
    -------
    pd.DataFrame — one row per variant
    """
    import pandas as pd
    from brca1_pipeline import VariantFeatureExtractor

    variants  = load_cif_folder(cif_folder, pattern=pattern)
    extractor = VariantFeatureExtractor(
        wt_pdb,
        plddt_threshold=plddt_threshold,
        domain_range=domain_range,
    )

    print(f"\nRunning pipeline on {len(variants)} variants…")
    df = extractor.extract_batch(variants, renumber_offset=renumber_offset)

    df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(df)} rows saved → {output_csv}")
    return df


# ── CLI: print parsed variants without running the pipeline ─────────────────

if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "pdb_s"
    print(f"Scanning: {folder}")
    try:
        variants = load_cif_folder(folder, verbose=True)
        print(f"\nTotal variants ready for extract_batch(): {len(variants)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")