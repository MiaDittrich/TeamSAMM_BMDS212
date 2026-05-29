"""
BRCA1 Missense Variant — Full Structural & Biochemical Feature Pipeline
========================================================================
Supports AlphaFold2 .cif variant files and AlphaFold2/3 .cif/.pdb wildtype files.

Steps
-----
  0.  Parse CIF/PDB; extract per-residue pLDDT and global pLDDT
  1.  Structural alignment  — pLDDT-filtered CA Superimposer, domain-scoped
  2.  Global features       — CA RMSD, backbone RMSD (domain-scoped)
  3.  Local features        — displacement at mutant site + flanking counts (domain-scoped)
  4.  Shell RMSDs           — 5, 8, 12 Å shells around mutant CA (domain-scoped)
  5.  Dihedral features     — φ/ψ ±2 residues for both WT and variant;
                              Δφ/Δψ with circular arithmetic; sin/cos encoding;
                              Ramachandran violation flag
  6.  Domain encoding       — RING / central / C-terminus (Clark et al. 2012)
  7.  Biochemical features  — PAM250, Δsize, Δhydrophobicity, Δcharge, Δaromaticity

Dependencies
------------
    pip install biopython numpy pandas

Usage
-----
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
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from Bio.PDB import PDBParser, MMCIFParser, Superimposer, PPBuilder

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PLDDT_THRESHOLD  = 70.0
SHELL_RADII      = [5.0, 8.0, 12.0]
DISPLACEMENT_THR = 1.0        # Å
LOCAL_WINDOW     = 2          # ±residues for dihedral extraction
BACKBONE_ATOMS   = {"N", "CA", "C", "O"}

# Clark et al. 2012 BRCA1 functional domains (UniProt 1-based, inclusive)
DOMAINS = {
    "RING":    (1,    109),
    "central": (758,  1064),
    "c_term":  (1650, 1863),
}

# ═══════════════════════════════════════════════════════════════════════════
# BIOCHEMICAL LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════

HYDROPHOBICITY: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

RESIDUE_SIZE: dict[str, float] = {
    "G":  57.0, "A":  71.0, "S":  87.0, "T": 101.0, "C": 103.0,
    "V":  99.0, "L": 113.0, "I": 113.0, "P":  97.0, "F": 147.0,
    "Y": 163.0, "W": 186.0, "D": 115.0, "E": 129.0, "N": 114.0,
    "Q": 128.0, "H": 137.0, "K": 128.0, "R": 156.0, "M": 131.0,
}

CHARGE: dict[str, int] = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1,"G": 0, "H": 0,  "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0,  "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0,  "V": 0,
}

AROMATICITY: dict[str, int] = {
    aa: int(aa in {"F", "Y", "W", "H"})
    for aa in "ACDEFGHIKLMNPQRSTVWY"
}

_PAM250_RAW = {
    ("A","A"):2, ("A","R"):-2,("A","N"):0, ("A","D"):0, ("A","C"):-2,
    ("A","Q"):0, ("A","E"):0, ("A","G"):1, ("A","H"):-1,("A","I"):-1,
    ("A","L"):-2,("A","K"):-1,("A","M"):1, ("A","F"):-3,("A","P"):1,
    ("A","S"):1, ("A","T"):1, ("A","W"):-6,("A","Y"):-3,("A","V"):0,
    ("R","R"):6, ("R","N"):0, ("R","D"):-1,("R","C"):-4,("R","Q"):1,
    ("R","E"):-1,("R","G"):-3,("R","H"):2, ("R","I"):-2,("R","L"):-3,
    ("R","K"):3, ("R","M"):0, ("R","F"):-4,("R","P"):0, ("R","S"):0,
    ("R","T"):-1,("R","W"):2, ("R","Y"):-4,("R","V"):-2,
    ("N","N"):2, ("N","D"):2, ("N","C"):-4,("N","Q"):1, ("N","E"):1,
    ("N","G"):0, ("N","H"):2, ("N","I"):-2,("N","L"):-3,("N","K"):1,
    ("N","M"):-2,("N","F"):-3,("N","P"):0, ("N","S"):1, ("N","T"):0,
    ("N","W"):-4,("N","Y"):-2,("N","V"):-2,
    ("D","D"):4, ("D","C"):-5,("D","Q"):2, ("D","E"):3, ("D","G"):1,
    ("D","H"):1, ("D","I"):-2,("D","L"):-4,("D","K"):0, ("D","M"):-3,
    ("D","F"):-6,("D","P"):-1,("D","S"):0, ("D","T"):0, ("D","W"):-7,
    ("D","Y"):-4,("D","V"):-2,
    ("C","C"):12,("C","Q"):-5,("C","E"):-5,("C","G"):-3,("C","H"):-3,
    ("C","I"):-2,("C","L"):-6,("C","K"):-5,("C","M"):-5,("C","F"):-4,
    ("C","P"):-3,("C","S"):0, ("C","T"):-2,("C","W"):-8,("C","Y"):0,
    ("C","V"):-2,
    ("Q","Q"):4, ("Q","E"):2, ("Q","G"):-1,("Q","H"):3, ("Q","I"):-2,
    ("Q","L"):-2,("Q","K"):1, ("Q","M"):1, ("Q","F"):-5,("Q","P"):0,
    ("Q","S"):-1,("Q","T"):-1,("Q","W"):-5,("Q","Y"):-4,("Q","V"):-2,
    ("E","E"):4, ("E","G"):0, ("E","H"):1, ("E","I"):-2,("E","L"):-3,
    ("E","K"):0, ("E","M"):-2,("E","F"):-5,("E","P"):-1,("E","S"):0,
    ("E","T"):0, ("E","W"):-7,("E","Y"):-4,("E","V"):-2,
    ("G","G"):5, ("G","H"):-2,("G","I"):-3,("G","L"):-4,("G","K"):-2,
    ("G","M"):-3,("G","F"):-5,("G","P"):0, ("G","S"):1, ("G","T"):0,
    ("G","W"):-7,("G","Y"):-5,("G","V"):-1,
    ("H","H"):6, ("H","I"):-2,("H","L"):-2,("H","K"):0, ("H","M"):-2,
    ("H","F"):-2,("H","P"):0, ("H","S"):-1,("H","T"):-1,("H","W"):-3,
    ("H","Y"):0, ("H","V"):-2,
    ("I","I"):5, ("I","L"):2, ("I","K"):-2,("I","M"):2, ("I","F"):1,
    ("I","P"):-2,("I","S"):-1,("I","T"):0, ("I","W"):-5,("I","Y"):-1,
    ("I","V"):4,
    ("L","L"):6, ("L","K"):-3,("L","M"):4, ("L","F"):2, ("L","P"):-3,
    ("L","S"):-3,("L","T"):-2,("L","W"):-2,("L","Y"):-1,("L","V"):2,
    ("K","K"):5, ("K","M"):0, ("K","F"):-5,("K","P"):-1,("K","S"):0,
    ("K","T"):0, ("K","W"):-3,("K","Y"):-4,("K","V"):-2,
    ("M","M"):6, ("M","F"):0, ("M","P"):-2,("M","S"):-2,("M","T"):-1,
    ("M","W"):-4,("M","Y"):-2,("M","V"):2,
    ("F","F"):9, ("F","P"):-5,("F","S"):-3,("F","T"):-3,("F","W"):0,
    ("F","Y"):7, ("F","V"):-1,
    ("P","P"):6, ("P","S"):1, ("P","T"):0, ("P","W"):-6,("P","Y"):-5,
    ("P","V"):1,
    ("S","S"):2, ("S","T"):1, ("S","W"):-2,("S","Y"):-3,("S","V"):-1,
    ("T","T"):3, ("T","W"):-5,("T","Y"):-3,("T","V"):0,
    ("W","W"):17,("W","Y"):0, ("W","V"):-6,
    ("Y","Y"):10, ("Y","V"):-2,
    ("V","V"):4,
}
PAM250: dict[tuple[str, str], int] = {}
for (_a, _b), _s in _PAM250_RAW.items():
    PAM250[(_a, _b)] = _s
    PAM250[(_b, _a)] = _s


def pam250(aa1: str, aa2: str) -> Optional[int]:
    return PAM250.get((aa1.upper(), aa2.upper()), None)


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURE PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_structure(path: Path, name: str):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(name, str(path))


def extract_global_plddt_from_cif(cif_path: Path) -> Optional[float]:
    """
    Extract global pLDDT from an AF2 mmCIF file.
    Handles both key-value and loop formats.
    """
    try:
        text = cif_path.read_text(errors="replace")
    except Exception:
        return None

    # Strategy 1: key-value format  (_ma_qa_metric_global.metric_value <float>)
    m = re.search(r"_ma_qa_metric_global\.metric_value\s+([\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Strategy 2: loop format — find pLDDT metric id, then look up its value
    # Column order in AF2 CIFs: id, mode, name, software_group_id, type
    # so name is parts[2], NOT parts[1]
    metric_id_for_plddt = None
    metric_block = re.search(
        r"loop_.*?_ma_qa_metric\.id.*?_ma_qa_metric\.type(.*?)(?=loop_|\Z)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if metric_block:
        for line in metric_block.group(1).splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("_"):
                continue
            if len(parts) >= 3 and parts[2].lower() in {"plddt", "\"plddt\"", "'plddt'"}:
                metric_id_for_plddt = parts[0].strip("\"'")
                break

    if metric_id_for_plddt is not None:
        global_block = re.search(
            r"loop_.*?_ma_qa_metric_global\.(.*?)(?=loop_|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if global_block:
            block_text = global_block.group(0)
            headers = re.findall(r"_ma_qa_metric_global\.(\S+)", block_text)
            try:
                id_col  = headers.index("metric_id")
                val_col = headers.index("metric_value")
            except ValueError:
                id_col = val_col = None
            if id_col is not None:
                for line in block_text.splitlines():
                    parts = line.split()
                    if (len(parts) > max(id_col, val_col) and
                            parts[id_col] == metric_id_for_plddt):
                        try:
                            return float(parts[val_col])
                        except ValueError:
                            pass

    # Strategy 3: mean of per-residue CA B-factors
    bfactors = re.findall(
        r"^ATOM\s+\S+\s+CA\s+.*?\s+([\d.]+)\s+\S+\s+\d",
        text, re.MULTILINE,
    )
    if bfactors:
        try:
            return float(np.mean([float(v) for v in bfactors]))
        except ValueError:
            pass

    return None


def extract_plddt_per_residue(model) -> dict[int, float]:
    """Return {seq_num: mean_pLDDT} for all standard residues."""
    result = {}
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            bfactors = [a.bfactor for a in res.get_atoms()]
            if bfactors:
                result[res.id[1]] = float(np.mean(bfactors))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_residues(model, chain_id: Optional[str] = None) -> list:
    residues = []
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[0] == " ":
                residues.append(res)
    return residues


def _get_residues_in_range(model, start: int, end: int,
                           chain_id: Optional[str] = None) -> list:
    return [
        r for r in _get_residues(model, chain_id)
        if start <= r.id[1] <= end
    ]


def _select_residues(model, domain_range: Optional[tuple[int, int]],
                     chain_id: Optional[str] = None) -> list:
    if domain_range is not None:
        return _get_residues_in_range(model, domain_range[0], domain_range[1], chain_id)
    return _get_residues(model, chain_id)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(a) != len(b):
        return np.nan
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _circular_diff(a: float, b: float) -> float:
    """
    Circular difference a − b in degrees, result in [−180, 180].
    Handles the periodicity of dihedral angles correctly so that,
    e.g., _circular_diff(−179, 179) = −2 rather than −358.
    """
    if np.isnan(a) or np.isnan(b):
        return np.nan
    d = a - b
    return float((d + 180.0) % 360.0 - 180.0)


def _renumber(model, offset: int):
    for chain in model:
        for res in chain:
            old = res.id
            res.id = (old[0], old[1] + offset, old[2])


def _res_by_num(model, seq_num: int, chain_id: Optional[str] = None):
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[1] == seq_num and res.id[0] == " ":
                return res
    return None


def _mean_plddt(residue) -> float:
    vals = [a.bfactor for a in residue.get_atoms()]
    return float(np.mean(vals)) if vals else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — STRUCTURAL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def align_to_wildtype(
    wt_model,
    var_model,
    mutant_seq_num: int,
    plddt_threshold: float = PLDDT_THRESHOLD,
    domain_range: Optional[tuple[int, int]] = None,
) -> int:
    """
    Superimpose var_model onto wt_model using high-confidence CA atoms.
    Transformation applied in-place to ALL atoms of var_model.
    Returns the number of anchor atoms used.
    """
    wt_res  = _select_residues(wt_model,  domain_range)
    var_res = _select_residues(var_model, domain_range)

    wt_dict  = {r.id[1]: r for r in wt_res}
    var_dict = {r.id[1]: r for r in var_res}

    wt_atoms, var_atoms = [], []
    for num in sorted(set(wt_dict) & set(var_dict)):
        wt_r, var_r = wt_dict[num], var_dict[num]
        if (_mean_plddt(wt_r) < plddt_threshold or
                _mean_plddt(var_r) < plddt_threshold):
            continue
        if "CA" in wt_r and "CA" in var_r:
            wt_atoms.append(wt_r["CA"])
            var_atoms.append(var_r["CA"])

    if len(wt_atoms) < 3:
        raise ValueError(
            f"Only {len(wt_atoms)} high-confidence CA anchors found "
            f"(threshold={plddt_threshold}, domain_range={domain_range}). "
            f"Try lowering plddt_threshold or widening domain_range."
        )

    sup = Superimposer()
    sup.set_atoms(wt_atoms, var_atoms)
    sup.apply(var_model.get_atoms())
    return len(wt_atoms)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — GLOBAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_global_features(
    wt_model,
    var_model,
    domain_range: Optional[tuple[int, int]] = None,
) -> dict:
    wt_dict  = {r.id[1]: r for r in _select_residues(wt_model,  domain_range)}
    var_dict = {r.id[1]: r for r in _select_residues(var_model, domain_range)}
    common   = sorted(set(wt_dict) & set(var_dict))

    wt_ca, var_ca, wt_bb, var_bb = [], [], [], []
    for n in common:
        wr, vr = wt_dict[n], var_dict[n]
        if "CA" in wr and "CA" in vr:
            wt_ca.append(wr["CA"].get_vector().get_array())
            var_ca.append(vr["CA"].get_vector().get_array())
        if all(a in wr for a in BACKBONE_ATOMS) and all(a in vr for a in BACKBONE_ATOMS):
            for at in ("N", "CA", "C", "O"):
                wt_bb.append(wr[at].get_vector().get_array())
                var_bb.append(vr[at].get_vector().get_array())

    return {
        "ca_rmsd":       _rmsd(np.array(wt_ca), np.array(var_ca)),
        "backbone_rmsd": _rmsd(np.array(wt_bb), np.array(var_bb)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — LOCAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_local_features(
    wt_model,
    var_model,
    mutant_seq_num: int,
    threshold: float = DISPLACEMENT_THR,
    domain_range: Optional[tuple[int, int]] = None,
) -> dict:
    wt_dict  = {r.id[1]: r for r in _select_residues(wt_model,  domain_range)}
    var_dict = {r.id[1]: r for r in _select_residues(var_model, domain_range)}

    mut_disp = np.nan
    wr = _res_by_num(wt_model, mutant_seq_num)
    vr = _res_by_num(var_model, mutant_seq_num)
    if wr and vr and "CA" in wr and "CA" in vr:
        mut_disp = float((wr["CA"].get_vector() - vr["CA"].get_vector()).norm())

    up = down = 0
    for n in sorted(set(wt_dict) & set(var_dict)):
        if n == mutant_seq_num:
            continue
        wr2, vr2 = wt_dict[n], var_dict[n]
        if "CA" not in wr2 or "CA" not in vr2:
            continue
        d = float((wr2["CA"].get_vector() - vr2["CA"].get_vector()).norm())
        if d >= threshold:
            if n < mutant_seq_num:
                up += 1
            else:
                down += 1

    return {
        "mutant_ca_displacement": mut_disp,
        "n_displaced_upstream":   up,
        "n_displaced_downstream": down,
        "n_displaced_total":      up + down,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — SHELL RMSDs  (domain-scoped to prevent disorder contamination)
# ═══════════════════════════════════════════════════════════════════════════

def compute_shell_rmsds(
    wt_model,
    var_model,
    mutant_seq_num: int,
    radii: list = SHELL_RADII,
    domain_range: Optional[tuple[int, int]] = None,
) -> dict:
    """
    RMSD of CA atoms within each shell radius of the mutant CA in WT.

    domain_range restricts which residues are considered when building each
    shell.  Without it, disordered-linker residues whose AF2 coordinates
    happen to fall near the mutant site in the WT prediction (but are placed
    elsewhere in variant predictions) inflate the 8 Å / 12 Å shell RMSDs
    to non-physical values (> 10 Å for a single missense variant).
    """
    wt_dict  = {r.id[1]: r for r in _select_residues(wt_model,  domain_range)}
    var_dict = {r.id[1]: r for r in _select_residues(var_model, domain_range)}

    wt_mut = wt_dict.get(mutant_seq_num)
    if wt_mut is None or "CA" not in wt_mut:
        out = {}
        for r in radii:
            out[f"shell_rmsd_{int(r)}A"] = np.nan
            out[f"shell_n_{int(r)}A"]    = 0
        return out

    ref_vec = wt_mut["CA"].get_vector()
    out = {}
    for radius in radii:
        wt_c, var_c = [], []
        for n, wr in wt_dict.items():
            if "CA" not in wr:
                continue
            if (wr["CA"].get_vector() - ref_vec).norm() <= radius:
                vr = var_dict.get(n)
                if vr and "CA" in vr:
                    wt_c.append(wr["CA"].get_vector().get_array())
                    var_c.append(vr["CA"].get_vector().get_array())
        out[f"shell_rmsd_{int(radius)}A"] = _rmsd(np.array(wt_c), np.array(var_c))
        out[f"shell_n_{int(radius)}A"]    = len(wt_c)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — DIHEDRAL ANGLES + RAMACHANDRAN
# ═══════════════════════════════════════════════════════════════════════════

_RAMA_ALLOWED = [
    (-180, -30,  -80,  50),
    (-180, -30,  100, 180),
    (-180, -30, -180,-130),
    ( -30,  10,  -80,  50),
    (  40, 180,  -10, 180),
]


def _phi_psi_for(pp_list, seq_num: int) -> tuple[float, float]:
    for pp in pp_list:
        for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
            if res.id[1] == seq_num:
                return (
                    float(np.degrees(phi)) if phi is not None else np.nan,
                    float(np.degrees(psi)) if psi is not None else np.nan,
                )
    return np.nan, np.nan


def _rama_disallowed(phi: float, psi: float) -> bool:
    if np.isnan(phi) or np.isnan(psi):
        return False
    return not any(
        p0 <= phi <= p1 and s0 <= psi <= s1
        for p0, p1, s0, s1 in _RAMA_ALLOWED
    )


def compute_dihedral_features(
    var_model,
    mutant_seq_num: int,
    wt_model=None,
    window: int = LOCAL_WINDOW,
) -> dict:
    """
    Dihedral features for residues [mutant_seq_num ± window].

    For each position computes:
      phi_{offset}, psi_{offset}          — variant angles (degrees)
      sin/cos_phi_{offset}, sin/cos_psi_{offset} — sin/cos encoding of variant angles
      wt_phi_{offset}, wt_psi_{offset}    — WT angles (only if wt_model provided)
      delta_phi_{offset}, delta_psi_{offset}  — circular difference var − WT
      sin/cos_delta_phi_{offset}, sin/cos_delta_psi_{offset}  — encoded delta

    The sin/cos encoding is required for linear models because dihedral angles
    are periodic: 179° and −179° are adjacent, not maximally different.

    Δφ/Δψ uses circular arithmetic (_circular_diff) so the result is always
    in [−180, 180] regardless of the raw angle values.
    """
    var_pp = PPBuilder().build_peptides(var_model)
    wt_pp  = PPBuilder().build_peptides(wt_model) if wt_model is not None else None

    out = {}
    any_violation = False

    for offset in range(-window, window + 1):
        res_num = mutant_seq_num + offset
        vphi, vpsi = _phi_psi_for(var_pp, res_num)

        out[f"phi_{offset:+d}"] = vphi
        out[f"psi_{offset:+d}"] = vpsi

        out[f"sin_phi_{offset:+d}"] = np.sin(np.radians(vphi)) if not np.isnan(vphi) else np.nan
        out[f"cos_phi_{offset:+d}"] = np.cos(np.radians(vphi)) if not np.isnan(vphi) else np.nan
        out[f"sin_psi_{offset:+d}"] = np.sin(np.radians(vpsi)) if not np.isnan(vpsi) else np.nan
        out[f"cos_psi_{offset:+d}"] = np.cos(np.radians(vpsi)) if not np.isnan(vpsi) else np.nan

        if wt_pp is not None:
            wphi, wpsi = _phi_psi_for(wt_pp, res_num)
            out[f"wt_phi_{offset:+d}"] = wphi
            out[f"wt_psi_{offset:+d}"] = wpsi

            dphi = _circular_diff(vphi, wphi)
            dpsi = _circular_diff(vpsi, wpsi)
            out[f"delta_phi_{offset:+d}"] = dphi
            out[f"delta_psi_{offset:+d}"] = dpsi

            out[f"sin_delta_phi_{offset:+d}"] = np.sin(np.radians(dphi)) if not np.isnan(dphi) else np.nan
            out[f"cos_delta_phi_{offset:+d}"] = np.cos(np.radians(dphi)) if not np.isnan(dphi) else np.nan
            out[f"sin_delta_psi_{offset:+d}"] = np.sin(np.radians(dpsi)) if not np.isnan(dpsi) else np.nan
            out[f"cos_delta_psi_{offset:+d}"] = np.cos(np.radians(dpsi)) if not np.isnan(dpsi) else np.nan

        if _rama_disallowed(vphi, vpsi):
            any_violation = True

    out["ramachandran_violation"] = int(any_violation)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — DOMAIN ONE-HOT ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def compute_domain_encoding(mutant_seq_num: int) -> dict:
    return {
        f"domain_{name}": int(s <= mutant_seq_num <= e)
        for name, (s, e) in DOMAINS.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — BIOCHEMICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_biochemical_features(wt_aa: str, mut_aa: str) -> dict:
    wt  = wt_aa.upper()
    mut = mut_aa.upper()

    wt_hydro  = HYDROPHOBICITY.get(wt,  np.nan)
    mut_hydro = HYDROPHOBICITY.get(mut, np.nan)
    wt_size   = RESIDUE_SIZE.get(wt,    np.nan)
    mut_size  = RESIDUE_SIZE.get(mut,   np.nan)
    wt_chg    = CHARGE.get(wt,  0)
    mut_chg   = CHARGE.get(mut, 0)
    wt_aro    = AROMATICITY.get(wt,  0)
    mut_aro   = AROMATICITY.get(mut, 0)
    pam_score = pam250(wt, mut)

    def _d(a, b):
        return (b - a) if not (np.isnan(a) or np.isnan(b)) else np.nan

    return {
        "wt_aa":                   wt,
        "mut_aa":                  mut,
        "pam250_score":            float(pam_score) if pam_score is not None else np.nan,
        "delta_hydrophobicity":    _d(wt_hydro, mut_hydro),
        "delta_size":              _d(wt_size,  mut_size),
        "delta_charge":            mut_chg - wt_chg,
        "delta_aromaticity":       mut_aro - wt_aro,
        "wt_hydrophobicity":       wt_hydro,
        "mut_hydrophobicity":      mut_hydro,
        "wt_size":                 wt_size,
        "mut_size":                mut_size,
        "wt_charge":               wt_chg,
        "mut_charge":              mut_chg,
        "wt_aromatic":             wt_aro,
        "mut_aromatic":            mut_aro,
        "is_charge_reversal":      int(np.sign(wt_chg) != np.sign(mut_chg)
                                       and (wt_chg != 0 or mut_chg != 0)),
        "is_size_increase":        int(mut_size > wt_size)
                                   if not (np.isnan(wt_size) or np.isnan(mut_size)) else np.nan,
        "is_hydrophobic_to_polar": int(wt_hydro > 0 and mut_hydro <= 0)
                                   if not (np.isnan(wt_hydro) or np.isnan(mut_hydro)) else np.nan,
        "is_polar_to_hydrophobic": int(wt_hydro <= 0 and mut_hydro > 0)
                                   if not (np.isnan(wt_hydro) or np.isnan(mut_hydro)) else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class VariantFeatureExtractor:
    """
    Full structural + biochemical feature extractor for BRCA1 missense variants.

    Parameters
    ----------
    wt_path         : path to wildtype PDB or CIF (AlphaFold2/3)
    wt_chain_id     : chain in wildtype structure (default 'A')
    plddt_threshold : minimum pLDDT for alignment anchors (default 70)
    domain_range    : (start, end) to restrict alignment, RMSD, shell, and
                      displacement computations to a folded domain.
                      E.g. DOMAINS["c_term"] = (1650, 1863).
    """

    def __init__(
        self,
        wt_path: str,
        wt_chain_id: str = "A",
        plddt_threshold: float = PLDDT_THRESHOLD,
        domain_range: Optional[tuple[int, int]] = None,
    ):
        self.wt_path         = Path(wt_path)
        self.wt_chain_id     = wt_chain_id
        self.plddt_threshold = plddt_threshold
        self.domain_range    = domain_range

        self.wt_global_plddt: Optional[float] = None
        if self.wt_path.suffix.lower() in {".cif", ".mmcif"}:
            self.wt_global_plddt = extract_global_plddt_from_cif(self.wt_path)

        if domain_range:
            print(f"  [INFO] domain_range={domain_range} — alignment and RMSD "
                  f"restricted to residues {domain_range[0]}–{domain_range[1]}")

    def _load_wt(self):
        return _parse_structure(self.wt_path, "wildtype")[0]

    def extract(
        self,
        var_path: str,
        wt_aa: str,
        mut_aa: str,
        mutant_res: int,
        variant_label: Optional[str] = None,
        var_chain_id: str = "A",
        renumber_offset: int = 0,
    ) -> dict:
        var_path = Path(var_path)
        label    = variant_label or var_path.stem

        var_struct = _parse_structure(var_path, label)
        var_model  = var_struct[0]
        wt_model   = self._load_wt()

        # global pLDDT
        var_global_plddt: Optional[float] = None
        if var_path.suffix.lower() in {".cif", ".mmcif"}:
            var_global_plddt = extract_global_plddt_from_cif(var_path)

        # per-residue pLDDT at mutant site
        plddt_map    = extract_plddt_per_residue(var_model)
        mutant_plddt = plddt_map.get(mutant_res + renumber_offset, np.nan)

        if renumber_offset != 0:
            _renumber(var_model, renumber_offset)

        # step 1 — align
        n_anchors = align_to_wildtype(
            wt_model, var_model, mutant_res,
            plddt_threshold=self.plddt_threshold,
            domain_range=self.domain_range,
        )

        # steps 2-3 — global + local structural (domain-scoped)
        global_f = compute_global_features(
            wt_model, var_model, domain_range=self.domain_range,
        )
        local_f = compute_local_features(
            wt_model, var_model, mutant_res, domain_range=self.domain_range,
        )

        # step 4 — shell RMSDs (domain-scoped)
        shell_f = compute_shell_rmsds(
            wt_model, var_model, mutant_res, domain_range=self.domain_range,
        )

        # step 5 — dihedrals with WT comparison, sin/cos encoding, Δφ/Δψ
        dihedral_f = compute_dihedral_features(
            var_model, mutant_res, wt_model=wt_model,
        )

        # step 6 — domain encoding
        domain_f = compute_domain_encoding(mutant_res)

        # step 7 — biochemical
        biochem_f = compute_biochemical_features(wt_aa, mut_aa)

        return {
            "variant":             label,
            "mutant_residue":      mutant_res,
            "var_global_plddt":    var_global_plddt if var_global_plddt is not None else np.nan,
            "wt_global_plddt":     self.wt_global_plddt if self.wt_global_plddt is not None else np.nan,
            "mutant_plddt":        mutant_plddt,
            "n_alignment_anchors": n_anchors,
            **global_f,
            **local_f,
            **shell_f,
            **dihedral_f,
            **domain_f,
            **biochem_f,
        }

    def extract_batch(
        self,
        variants: list[tuple],
        renumber_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Process a list of variants and return a tidy DataFrame.
        Each entry: (var_path, wt_aa, mut_aa, mutant_res[, label])
        """
        records = []
        for entry in variants:
            vpath, wt_aa, mut_aa, mres = entry[:4]
            label = entry[4] if len(entry) > 4 else None
            try:
                row = self.extract(
                    vpath, wt_aa, mut_aa, mres,
                    variant_label=label,
                    renumber_offset=renumber_offset,
                )
                records.append(row)
                print(f"  [OK]  {label or Path(vpath).stem}  res={mres}  "
                      f"pLDDT={row.get('mutant_plddt', float('nan')):.1f}  "
                      f"CA-RMSD={row.get('ca_rmsd', float('nan')):.3f}")
            except Exception as exc:
                lbl = label or Path(vpath).stem
                print(f"  [ERR] {lbl}: {exc}")
                records.append({"variant": lbl, "mutant_residue": mres, "error": str(exc)})

        return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from cif_loader import run_pipeline

    run_pipeline(
        cif_folder      = "pdb_s",
        wt_pdb          = "fold_brca_wt_v2_model_0.cif",
        output_csv      = "brca1_features.csv",
        plddt_threshold = 70.0,
        renumber_offset = 0,
        domain_range    = DOMAINS["c_term"],
    )
