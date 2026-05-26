"""
Steps:
  1. Structural alignment of missense variants to wildtype (pLDDT-filtered CA superimposition)
  2. Global features: CA RMSD and backbone RMSD
  3. Local features: displacement at mutant site + flanking displaced residues
  4. Shell RMSDs: 5, 8, 12 Å shells around mutant residue
  5. Dihedral angles ±2 residues around mutant + Ramachandran violation flag
  6. Domain one-hot encoding (RING, central, C-terminus)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from Bio import PDB
from Bio.PDB import (
    PDBParser,
    Superimposer,
    PDBIO,
    Select,
)
from Bio.PDB.Polypeptide import PPBuilder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLDDT_THRESHOLD = 70.0          # minimum pLDDT (stored in B-factor) for alignment -- can change
SHELL_RADII = [5.0, 8.0, 12.0]  # Å shells for shell RMSD
DISPLACEMENT_THRESHOLD = 1.0    # Å displacement threshold for local counting
LOCAL_WINDOW = 2                 # ±residues around mutant for dihedral extraction
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Domain definitions: (start, end) in UniProt residue numbering (1-based, inclusive)
# Drawn from Clark et al., 2012 for BRCA1 -- may change
DOMAINS = {
    "RING":     (1,    109),
    "central":  (758,  1064),
    "c_term":   (1650, 1863),
}

def _get_residues(model, chain_id: Optional[str] = None) -> list:
    """Return sorted list of residues (hetflag == ' ') from a model."""
    residues = []
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[0] == " ":   # exclude HETATM and water
                residues.append(res)
    return residues


def _ca_coords(residues: list) -> np.ndarray:
    """Extract alpha-carbon coordinates for a list of residues."""
    coords = []
    for res in residues:
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())
    return np.array(coords)


def _rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute RMSD between two equal-length coordinate arrays."""
    if len(coords_a) == 0 or len(coords_a) != len(coords_b):
        return np.nan
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _backbone_coords(residues: list) -> np.ndarray:
    """Extract N, CA, C, O coordinates (4 atoms per residue, flatten)."""
    coords = []
    for res in residues:
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in res:
                coords.append(res[atom_name].get_vector().get_array())
    return np.array(coords)


def _renumber_residues(model, offset: int):
    """
    Shift all residue sequence numbers in a model by `offset` (in-place).
    Used to align variant numbering to wildtype UniProt numbering.
    """
    for chain in model:
        for res in chain:
            old_id = res.id
            new_seq = old_id[1] + offset
            res.id = (old_id[0], new_seq, old_id[2])


def _get_residue_by_seqnum(model, seq_num: int, chain_id: Optional[str] = None):
    """Fetch a residue by its sequence number; returns None if not found."""
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[1] == seq_num and res.id[0] == " ":
                return res
    return None


def _plddt_of_residue(residue) -> float:
    """Return mean pLDDT (B-factor) for all atoms in a residue."""
    bfactors = [atom.bfactor for atom in residue.get_atoms()]
    return float(np.mean(bfactors)) if bfactors else 0.0


# ---------------------------------------------------------------------------
# Step 1 — Structural alignment
# ---------------------------------------------------------------------------
def align_variant_to_wildtype(
    wt_model,
    var_model,
    mutant_seq_num: int,
    plddt_threshold: float = PLDDT_THRESHOLD,
    window: Optional[int] = None,
    exclude_mutation_window: int = 10,  # NEW: exclude ±10 residues from alignment
) -> tuple[list, list, float]:
    """
    Superimpose variant model onto wildtype using high-confidence CA atoms,
    optionally excluding residues near the mutation site.
    """
    wt_residues  = _get_residues(wt_model)
    var_residues = _get_residues(var_model)

    wt_dict  = {r.id[1]: r for r in wt_residues}
    var_dict = {r.id[1]: r for r in var_residues}

    common_nums = sorted(set(wt_dict) & set(var_dict))

    if window is not None:
        common_nums = [n for n in common_nums
                       if abs(n - mutant_seq_num) <= window]

    wt_atoms, var_atoms = [], []
    for num in common_nums:
        # NEW: Skip residues near mutation site for alignment
        if abs(num - mutant_seq_num) <= exclude_mutation_window:
            continue
            
        wt_res  = wt_dict[num]
        var_res = var_dict[num]

        if (_plddt_of_residue(wt_res)  < plddt_threshold or
                _plddt_of_residue(var_res) < plddt_threshold):
            continue

        if "CA" in wt_res and "CA" in var_res:
            wt_atoms.append(wt_res["CA"])
            var_atoms.append(var_res["CA"])

    if len(wt_atoms) < 3:
        raise ValueError(
            f"Too few high-confidence CA atoms for alignment "
            f"({len(wt_atoms)}). Lower plddt_threshold or inspect structures."
        )

    sup = Superimposer()
    sup.set_atoms(wt_atoms, var_atoms)
    sup.apply(var_model.get_atoms())

    return wt_atoms, var_atoms, sup.rms


# ---------------------------------------------------------------------------
# Step 2 — Global features: CA RMSD + backbone RMSD
# ---------------------------------------------------------------------------
def compute_global_features(wt_model, var_model, alignment_rmsd: float = None) -> dict:
    """
    Compute global CA RMSD and backbone (N/CA/C/O) RMSD after alignment.
    
    **Important**: This computes RMSD over ALL residues, not just those
    used for alignment. The variant has already been superimposed.

    Parameters
    ----------
    wt_model, var_model : Bio.PDB Models (variant already aligned to wt)
    alignment_rmsd : optional RMSD from the superposition step

    Returns
    -------
    dict with keys: ca_rmsd, backbone_rmsd, ca_rmsd_all, alignment_rmsd
    """
    wt_residues  = _get_residues(wt_model)
    var_residues = _get_residues(var_model)

    wt_dict  = {r.id[1]: r for r in wt_residues}
    var_dict = {r.id[1]: r for r in var_residues}
    common   = sorted(set(wt_dict) & set(var_dict))

    wt_ca, var_ca = [], []
    wt_bb, var_bb = [], []

    for num in common:
        wt_r  = wt_dict[num]
        var_r = var_dict[num]

        if "CA" in wt_r and "CA" in var_r:
            wt_ca.append(wt_r["CA"].get_vector().get_array())
            var_ca.append(var_r["CA"].get_vector().get_array())

        # Backbone: only if ALL four atoms present in both
        if all(a in wt_r for a in BACKBONE_ATOMS) and \
           all(a in var_r for a in BACKBONE_ATOMS):
            for atom_name in ("N", "CA", "C", "O"):
                wt_bb.append(wt_r[atom_name].get_vector().get_array())
                var_bb.append(var_r[atom_name].get_vector().get_array())

    ca_rmsd_all = _rmsd(np.array(wt_ca), np.array(var_ca))
    bb_rmsd_all = _rmsd(np.array(wt_bb), np.array(var_bb))

    result = {
        "ca_rmsd":       ca_rmsd_all,
        "backbone_rmsd": bb_rmsd_all,
        "n_ca_atoms":    len(wt_ca),
        "n_bb_atoms":    len(wt_bb),
    }
    
    if alignment_rmsd is not None:
        result["alignment_rmsd"] = alignment_rmsd
    
    return result


# ---------------------------------------------------------------------------
# Step 3 — Local features: mutant-site displacement + flanking displacement count
# ---------------------------------------------------------------------------

def compute_local_features(
    wt_model,
    var_model,
    mutant_seq_num: int,
    displacement_threshold: float = DISPLACEMENT_THRESHOLD,
) -> dict:
    """
    Local structural features around the mutation site.

    Parameters
    ----------
    wt_model, var_model : aligned Bio.PDB Models
    mutant_seq_num      : residue number of the mutation
    displacement_threshold : Å threshold to count a flanking residue as displaced

    Returns
    -------
    dict with keys:
        mutant_ca_displacement   : Euclidean CA displacement at mutant residue (Å)
        n_displaced_upstream     : # residues displaced ≥ threshold upstream of mutant
        n_displaced_downstream   : # residues displaced ≥ threshold downstream of mutant
        n_displaced_total        : total flanking displaced residues
    """
    wt_residues  = _get_residues(wt_model)
    var_residues = _get_residues(var_model)
    wt_dict  = {r.id[1]: r for r in wt_residues}
    var_dict = {r.id[1]: r for r in var_residues}

    # Displacement at mutant site
    mut_disp = np.nan
    wt_mut  = wt_dict.get(mutant_seq_num)
    var_mut = var_dict.get(mutant_seq_num)
    if wt_mut and var_mut and "CA" in wt_mut and "CA" in var_mut:
        diff = (wt_mut["CA"].get_vector() - var_mut["CA"].get_vector())
        mut_disp = float(diff.norm())

    # Flanking displacement counts
    common = sorted(set(wt_dict) & set(var_dict))
    upstream_displaced   = 0
    downstream_displaced = 0

    for num in common:
        if num == mutant_seq_num:
            continue
        wt_r  = wt_dict[num]
        var_r = var_dict[num]
        if "CA" not in wt_r or "CA" not in var_r:
            continue
        diff = (wt_r["CA"].get_vector() - var_r["CA"].get_vector())
        dist = float(diff.norm())
        if dist >= displacement_threshold:
            if num < mutant_seq_num:
                upstream_displaced += 1
            else:
                downstream_displaced += 1

    return {
        "mutant_ca_displacement":  mut_disp,
        "n_displaced_upstream":    upstream_displaced,
        "n_displaced_downstream":  downstream_displaced,
        "n_displaced_total":       upstream_displaced + downstream_displaced,
    }


# ---------------------------------------------------------------------------
# Step 4 — Shell RMSDs (5, 8, 12 Å) around mutant residue
# ---------------------------------------------------------------------------

def compute_shell_rmsds(
    wt_model,
    var_model,
    mutant_seq_num: int,
    radii: list = SHELL_RADII,
) -> dict:
    """
    Compute shell RMSDs using CA atoms within each radius of the mutant CA
    in the **wildtype** structure.

    Parameters
    ----------
    wt_model, var_model : aligned Bio.PDB Models
    mutant_seq_num      : residue number of the mutation
    radii               : list of shell radii in Å (default [5, 8, 12])

    Returns
    -------
    dict with keys: shell_rmsd_5A, shell_rmsd_8A, shell_rmsd_12A
                    (and shell_n_5A etc. for atom counts per shell)
    """
    wt_residues  = _get_residues(wt_model)
    var_residues = _get_residues(var_model)
    wt_dict  = {r.id[1]: r for r in wt_residues}
    var_dict = {r.id[1]: r for r in var_residues}

    wt_mut = wt_dict.get(mutant_seq_num)
    if wt_mut is None or "CA" not in wt_mut:
        return {f"shell_rmsd_{int(r)}A": np.nan for r in radii} | \
               {f"shell_n_{int(r)}A":    0       for r in radii}

    mut_ca_vec = wt_mut["CA"].get_vector()

    results = {}
    for radius in radii:
        wt_shell_coords, var_shell_coords = [], []
        for num, wt_r in wt_dict.items():
            if "CA" not in wt_r:
                continue
            dist = (wt_r["CA"].get_vector() - mut_ca_vec).norm()
            if dist <= radius:
                var_r = var_dict.get(num)
                if var_r and "CA" in var_r:
                    wt_shell_coords.append(wt_r["CA"].get_vector().get_array())
                    var_shell_coords.append(var_r["CA"].get_vector().get_array())

        key_r = f"shell_rmsd_{int(radius)}A"
        key_n = f"shell_n_{int(radius)}A"
        results[key_r] = _rmsd(np.array(wt_shell_coords), np.array(var_shell_coords))
        results[key_n] = len(wt_shell_coords)

    return results


# ---------------------------------------------------------------------------
# Step 5 — Dihedral angles ± LOCAL_WINDOW around mutant + Ramachandran flag
# ---------------------------------------------------------------------------

# Ramachandran "allowed" regions (approximate; based on PROCHECK / Lovell 2003)
# Glycine and proline have distinct regions; this covers non-Gly/Pro.
# Format: (phi_min, phi_max, psi_min, psi_max) in degrees
_RAMA_ALLOWED = [
    (-180, -30,  -80,  50),   # alpha-R helix
    (-180, -30,  100, 180),   # beta-strand / extended (upper half)
    ( -30,  10,  -80,  50),   # alpha-R helix (near 0)
    (-180, -30, -180, -130),  # extended (lower)
    ( 30,  180, -180, -180),  # left-handed (Gly only, but coarse allow)
    ( 40,  180,  -10, 180),   # left-handed alpha / polyPro II
]


def _phi_psi(ppb_list: list, seq_num: int) -> tuple[float, float]:
    """
    Return (phi, psi) in degrees for residue seq_num from a PPBuilder result.
    Returns (NaN, NaN) if angles cannot be computed.
    """
    for pp in ppb_list:
        phi_psi = pp.get_phi_psi_list()
        residues = list(pp)
        for res, (phi, psi) in zip(residues, phi_psi):
            if res.id[1] == seq_num:
                phi_deg = np.degrees(phi) if phi is not None else np.nan
                psi_deg = np.degrees(psi) if psi is not None else np.nan
                return phi_deg, psi_deg
    return np.nan, np.nan


def _is_ramachandran_disallowed(phi: float, psi: float) -> bool:
    """Return True if (phi, psi) falls outside all allowed regions."""
    if np.isnan(phi) or np.isnan(psi):
        return False   # cannot assess
    for (phi_min, phi_max, psi_min, psi_max) in _RAMA_ALLOWED:
        if phi_min <= phi <= phi_max and psi_min <= psi <= psi_max:
            return False
    return True


def compute_dihedral_features(
    var_model,
    mutant_seq_num: int,
    window: int = LOCAL_WINDOW,
) -> dict:
    """
    Extract phi/psi dihedral angles for residues in [mutant ± window] of the
    variant structure. Also returns a binary flag for any Ramachandran-
    disallowed angle in the window.

    Parameters
    ----------
    var_model       : aligned Bio.PDB Model of the variant
    mutant_seq_num  : residue number of the mutation
    window          : number of flanking residues on each side (default 2)

    Returns
    -------
    dict with keys:
        phi_{offset}, psi_{offset} for offset in [-window, ..., +window]
        ramachandran_violation : 1 if any angle in window is disallowed, else 0
    """
    ppb = PPBuilder()
    pp_list = ppb.build_peptides(var_model)

    results = {}
    any_violation = False

    for offset in range(-window, window + 1):
        seq_num = mutant_seq_num + offset
        phi, psi = _phi_psi(pp_list, seq_num)
        results[f"phi_{offset:+d}"] = phi
        results[f"psi_{offset:+d}"] = psi
        if _is_ramachandran_disallowed(phi, psi):
            any_violation = True

    results["ramachandran_violation"] = int(any_violation)
    return results


# ---------------------------------------------------------------------------
# Step 6 — Domain one-hot encoding
# ---------------------------------------------------------------------------

def compute_domain_encoding(mutant_seq_num: int) -> dict:
    """
    One-hot encode whether the mutant residue falls within defined BRCA1
    functional domains (Clark et al., 2012).

    Parameters
    ----------
    mutant_seq_num : UniProt residue number of the mutation

    Returns
    -------
    dict with binary keys: domain_RING, domain_central, domain_c_term
    """
    return {
        f"domain_{name}": int(start <= mutant_seq_num <= end)
        for name, (start, end) in DOMAINS.items()
    }


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class StructuralFeatureExtractor:
    """
    High-level interface for extracting all structural features from a set of
    BRCA variant AlphaFold2 PDB files against a wildtype reference.

    Parameters
    ----------
    wt_pdb_path    : path to the wildtype AlphaFold3 PDB file
    wt_chain_id    : chain identifier in the wildtype PDB (default 'A')
    plddt_threshold: minimum pLDDT score for alignment anchors (default 70)

    Example
    -------
    extractor = StructuralFeatureExtractor("wt_brca1_af3.pdb")
    features  = extractor.extract_features("variant_R1699W.pdb", 1699)
    df = extractor.extract_batch([
        ("variant_R1699W.pdb", 1699),
        ("variant_C61G.pdb",    61),
    ])
    """

    def __init__(
        self,
        wt_pdb_path: str,
        wt_chain_id: str = "A",
        plddt_threshold: float = PLDDT_THRESHOLD,
    ):
        self.wt_pdb_path     = Path(wt_pdb_path)
        self.wt_chain_id     = wt_chain_id
        self.plddt_threshold = plddt_threshold
        self._parser         = PDBParser(QUIET=True)
        self._wt_structure   = self._load(self.wt_pdb_path, "wildtype")

    # ------------------------------------------------------------------
    def _load(self, path: Path, name: str):
        return self._parser.get_structure(name, str(path))

    def _wt_model(self):
        """Return a fresh copy of wildtype model (model index 0)."""
        # Re-parse each time to avoid mutating the reference
        structure = self._load(self.wt_pdb_path, "wildtype")
        return structure[0]

    # ------------------------------------------------------------------
    def extract_features(
        self,
        variant_pdb_path: str,
        mutant_seq_num: int,
        variant_name: Optional[str] = None,
        var_chain_id: str = "A",
        renumber_offset: int = 0,
    ) -> dict:
        """
        Extract all features for a single variant PDB.

        Parameters
        ----------
        variant_pdb_path : path to variant AlphaFold2 PDB
        mutant_seq_num   : UniProt residue number of the missense mutation
        variant_name     : label for this variant (defaults to filename stem)
        var_chain_id     : chain in variant PDB (default 'A')
        renumber_offset  : integer offset to re-number variant residues to
                           match wildtype UniProt numbering (default 0 = no shift)

        Returns
        -------
        dict of all extracted features
        """
        var_path = Path(variant_pdb_path)
        name     = variant_name or var_path.stem

        var_structure = self._load(var_path, name)
        var_model     = var_structure[0]
        wt_model      = self._wt_model()

        # --- Step 1: renumber + align ---
        if renumber_offset != 0:
            _renumber_residues(var_model, renumber_offset)

        wt_atoms, var_atoms, alignment_rmsd = align_variant_to_wildtype(  # Updated
            wt_model, var_model, mutant_seq_num,
            plddt_threshold=self.plddt_threshold,
        )

        # --- Step 2: global ---
        global_feats = compute_global_features(wt_model, var_model, alignment_rmsd)  # Updated

        # --- Step 3: local ---
        local_feats = compute_local_features(wt_model, var_model, mutant_seq_num)

        # --- Step 4: shell RMSDs ---
        shell_feats = compute_shell_rmsds(wt_model, var_model, mutant_seq_num)

        # --- Step 5: dihedrals ---
        dihedral_feats = compute_dihedral_features(var_model, mutant_seq_num)

        # --- Step 6: domain encoding ---
        domain_feats = compute_domain_encoding(mutant_seq_num)

        return {
            "variant":           name,
            "mutant_residue":    mutant_seq_num,
            **global_feats,
            **local_feats,
            **shell_feats,
            **dihedral_feats,
            **domain_feats,
        }

    # ------------------------------------------------------------------
    def extract_batch(
        self,
        variant_list: list[tuple],
        renumber_offset: int = 0,
    ) -> pd.DataFrame:
        """
        Process multiple variants and return a tidy DataFrame.

        Parameters
        ----------
        variant_list : list of (pdb_path, mutant_seq_num) tuples, or
                       (pdb_path, mutant_seq_num, variant_name) tuples
        renumber_offset : applied uniformly to all variants

        Returns
        -------
        pd.DataFrame, one row per variant
        """
        records = []
        for entry in variant_list:
            pdb_path, mut_num = entry[0], entry[1]
            v_name = entry[2] if len(entry) > 2 else None
            try:
                feats = self.extract_features(
                    pdb_path, mut_num,
                    variant_name=v_name,
                    renumber_offset=renumber_offset,
                )
                records.append(feats)
                print(f"  [OK]  {v_name or Path(pdb_path).stem} (res {mut_num})")
            except Exception as exc:
                print(f"  [ERR] {v_name or Path(pdb_path).stem}: {exc}")
                records.append({"variant": v_name or Path(pdb_path).stem,
                                 "mutant_residue": mut_num,
                                 "error": str(exc)})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI / quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    WT_PDB   = "wt_brca1_af3.pdb"           # AlphaFold3 wildtype PDB
    VARIANTS = [
        # (variant_pdb_path, mutant_residue_number, optional_label)
        ("a102c.pdb",   102, "A102C"),
        ("e1829.pdb", 1829, "E1829P"),
    ]

    PLDDT_THRESHOLD = 70.0

    print(f"Loading wildtype from: {WT_PDB}")
    extractor = StructuralFeatureExtractor(WT_PDB, plddt_threshold=PLDDT_THRESHOLD)

    print("Extracting features for each variant…")
    df = extractor.extract_batch(VARIANTS)

    out_csv = "brca1_structural_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFeature matrix saved to: {out_csv}\n")
    with pd.option_context("display.max_columns", None,
                           "display.width",       200,
                           "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))