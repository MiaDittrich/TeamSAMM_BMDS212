# SAMM — BRCA1 Missense Variant Feature Pipeline

End-to-end pipeline that turns AlphaFold variant structures into a model-ready
feature matrix for predicting BRCA1 missense pathogenicity.

Focused on the BRCT C-terminal domain (residues 1650–1863). For each variant
the pipeline computes structural deviations from wildtype, dihedral
Ramachandran summaries, biochemical substitution scores, and an EvoEF2 ΔΔG
stability estimate.

---

## Quickstart

```bash
# 1. Install Python dependencies
pip install biopython numpy pandas

# 2. Build EvoEF2 (one-time)
git clone https://github.com/tommyhuangthu/EvoEF2.git
cd EvoEF2 && g++ -O3 -ffast-math -o EvoEF2 src/*.cpp && cd ..

# 3. Put AlphaFold variant CIFs in pdb_s/ following the naming convention
#    fold_brca1_<wt3>{resnum}<mut3>_model_*.cif
#    e.g. fold_brca1_met1775arg_model_0.cif
#
#    Put the wildtype prediction at fold_brca_wt_v2_model_0.cif
#    (top-level of this folder).

# 4. Run the whole pipeline
python3 run_all.py
```

Final output: `brca1_model_features_lean.csv` (22 columns, one row per variant).

---

## Pipeline stages

| Stage | Script                 | Output                              |
|-------|------------------------|-------------------------------------|
| 1     | `brca1_pipeline.py`    | `brca1_features.csv`                |
| 2     | `evoef2_stability.py`  | `brca1_features_with_evoef2.csv`    |
| 3     | `feature_engineering.py` | `brca1_model_features.csv`        |
| 4     | `prune_features.py`    | `brca1_model_features_lean.csv`     |

`run_all.py` chains all four stages.

- **Stage 1** parses each CIF, superimposes it on the wildtype using
  pLDDT-filtered Cα anchors (threshold = 70), and computes Cα/backbone RMSDs,
  per-shell RMSDs at 5/8/12 Å, ±2 residue φ/ψ angles + Ramachandran
  violation, domain one-hot encoding, and biochemical substitution features
  (PAM250, Δhydrophobicity, Δsize, Δcharge, Δaromaticity, plus binary flags).
- **Stage 2** converts every CIF to PDB and runs EvoEF2 `ComputeStability`
  on each variant and the wildtype, then writes per-term energies and a
  ΔΔG = E(variant) − E(WT) for every term.
- **Stage 3** runs EvoEF2 `BuildMutant` + `ComputeStability` per variant for a
  single-structure summary ΔΔG, drops zero-variance/redundant columns, and
  encodes any remaining raw dihedrals as sin/cos.
- **Stage 4** aggressively prunes to the 22 most-informative columns.

---

## Lean feature set (22 columns)

| Group         | Columns |
|---------------|---------|
| identity      | `variant`, `mutant_residue` |
| confidence    | `mutant_plddt` |
| structural    | `ca_rmsd`, `backbone_rmsd`, `mutant_ca_displacement` |
| shell RMSDs   | `shell_rmsd_5A`, `shell_rmsd_8A`, `shell_rmsd_12A` |
| dihedral      | `ramachandran_violation` |
| biochemical   | `pam250_score`, `delta_hydrophobicity`, `delta_size`, `delta_charge`, `delta_aromaticity`, `is_charge_reversal`, `is_size_increase`, `is_hydrophobic_to_polar`, `is_polar_to_hydrophobic` |
| ΔΔG (EvoEF2)  | `evoef2_ddg_Total`, `ddg_evoef2` |
| flags         | `is_disordered_variant` |

`is_disordered_variant = 1` flags variants outside the BRCT folded domain;
their structural features reflect AlphaFold linker noise and should be
filtered before modelling.

---

## File layout

```
SAMM /
├── run_all.py              # one-command driver
├── brca1_pipeline.py       # Stage 1: structural + biochemical features
├── cif_loader.py           # variant filename parser
├── evoef2_stability.py     # Stage 2: per-term EvoEF2 energies + ΔΔG
├── feature_engineering.py  # Stage 3: clean + summary ΔΔG
├── prune_features.py       # Stage 4: drop to lean feature set
├── brca1.txt               # BRCA1 reference amino-acid sequence
├── fold_brca_wt_v2_model_0.cif   # wildtype AlphaFold structure
├── pdb_s/                  # (gitignored) variant CIFs go here
└── EvoEF2/                 # (gitignored) clone + build separately
```

---

## Notes & caveats

- EvoEF2 requires PDB format, not mmCIF. Stage 2 handles the conversion
  automatically and caches results in `pdb_s_converted/`.
- The variant filename regex expects exactly the AlphaFold-server naming
  convention shown above; non-matching files are silently skipped.
- Structural features are *domain-scoped* to BRCT (1650–1863) so that
  AlphaFold's high-confidence predictions inside the folded domain are not
  contaminated by disordered-linker noise elsewhere in BRCA1.
