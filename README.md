# TeamSAMM_BMDS212

# TeamSAMM — BRCA1 Missense Variant Pathogenicity Prediction

Predict the clinical impact of **BRCA1 missense variants** by learning the
relationship between protein-structure / biochemistry features and
experimental (MAVE) functional scores, then validating those predictions
against independent **ClinVar** pathogenic/benign labels.

The work targets the **BRCT C-terminal domain (residues ~1650–1863)**, where
BRCA1 missense variants are most clinically consequential and where AlphaFold
structures are high-confidence.

---

## The big picture

The project runs as a four-stage pipeline. Each stage has its own folder and
its own detailed README.

| Stage | Folder | What it does | Detailed docs |
|-------|--------|--------------|---------------|
| **1. Feature extraction** | `feature_extraction/` (and `SAMM/`) | Turn AlphaFold variant structures into a model-ready feature matrix (structural deviations, Ramachandran dihedrals, biochemical substitution scores, EvoEF2 ΔΔG stability). | [feature_extraction/README.md](feature_extraction/README.md) |
| **2. Model training & prediction** | `aim3/` | Train regressors (Random Forest et al.) on MAVE functional scores, then predict and classify held-out ClinVar variants. | [aim3/README.md](aim3/README.md) |
| **3. External validation** | `aim4/` | Statistically validate that predicted scores separate ClinVar pathogenic from benign variants (distribution tests + logistic regression). | [aim4/README.md](aim4/README.md) |
| **4. Severity scoring** | `pipeline_4.py` | Convert raw MAVE predictions into calibrated severity scores ∈ [0, 1] with 95% CIs and Benign/VUS/Pathogenic tiers. | inline docstring |

**Core idea:** lower predicted score ⇒ more loss-of-function ⇒ more likely pathogenic.

---

## Quick start

Each stage is run independently from the repository root.

```bash
# Stage 1 — build the feature matrix from AlphaFold structures
#   (requires EvoEF2 built locally + variant CIFs; see feature_extraction/README.md)
cd feature_extraction && python3 run_all.py && cd ..

# Stage 2 — train the model and predict/classify ClinVar variants
pip install -r aim3/requirements.txt
python3 aim3/brca1_predict.py                 # default: Random Forest, 'both' label

# Stage 3 — validate predictions against ClinVar ground truth
python3 aim4/01_distribution_analysis.py
python3 aim4/02_logistic_regression.py

# Stage 4 — calibrated severity scores with confidence intervals
python3 pipeline_4.py
```

### Dependencies
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`,
`biopython` (Stage 1), and `xgboost` / `lightgbm` for the full research
pipelines. Stage 1 also needs a local [EvoEF2](https://github.com/tommyhuangthu/EvoEF2) build.

---

## Key results

**Stage 2 — model performance** (Random Forest, `both` label):

| Metric | Value |
|--------|-------|
| CV Spearman ρ | 0.564 ± 0.189 |
| CV Pearson r | 0.682 |
| ClinVar AUROC | 1.000 |

**Stage 3 — external validation** on **48 ClinVar variants** (24 pathogenic / 24 benign):

| Test | Result |
|------|--------|
| Mann–Whitney U (distribution) | p ≈ 3.5 × 10⁻⁹ |
| Kolmogorov–Smirnov (distribution) | p ≈ 3.0 × 10⁻¹² |
| Logistic regression AUROC (LOO-CV) | 0.997 |
| Average Precision | 0.997 |
| Accuracy @ Youden-optimal cutoff | 47/48 (97.9%) |

> **Reading the metrics:** AUROC / Average Precision are *threshold-free* and
> measure how well predicted scores **rank** pathogenic above benign.
> Accuracy/sensitivity/specificity depend on the chosen probability cutoff and
> are reported at both the naive 0.5 cutoff and the Youden-optimal cutoff. See
> [aim4/README.md](aim4/README.md) for the full explanation.

---

## Repository layout

```
.
├── feature_extraction/   # Stage 1 — AlphaFold structures → feature matrix
├── SAMM/                 # Stage 1 (earlier/parallel copy of the feature pipeline)
├── aim3/                 # Stage 2 — model training + ClinVar prediction
│   ├── brca1_predict.py              # main train/predict/save script
│   ├── brca1_mave_research_pipeline_v2.py  # full research pipeline
│   └── results/                      # predictions, metrics, figures, saved model
├── aim4/                 # Stage 3 — external validation
│   ├── 01_distribution_analysis.py
│   ├── 02_logistic_regression.py
│   └── results/                      # KDE, ROC, PR, confusion matrices, stats
├── pipeline_4.py         # Stage 4 — calibrated severity scores + CIs + tiers
├── brca1_mave_pipeline_v2.py  # ablation-pruned variant of the Stage 2 pipeline
├── results/              # outputs from the top-level pipelines (SHAP, ablation, etc.)
├── samm_sample_pdbs/     # example PDB structures
└── prior_versions/       # earlier/intermediate datasets (AlphaMissense, MaveDB, merges)
```

### Notes on auxiliary files
- `SAMM/` and `feature_extraction/` are two iterations of the Stage 1 feature
  pipeline; `feature_extraction/` is the cleaned-up version.
- `brca1_mave_pipeline_v2.py` is a feature-pruned variant of the Stage 2
  pipeline (drops features with negative leave-one-out ablation impact).
- Loose top-level CSVs (`brca1_final.csv`, `brca1_features_2.csv`, etc.) and
  `prior_versions/` hold intermediate and historical datasets used during
  development.

---

## Data flow

```
AlphaFold CIFs ──▶ Stage 1 ──▶ feature matrix ──┐
                                                ├──▶ Stage 2 ──▶ pred_score per variant
MAVE scores + AlphaMissense ────────────────────┘                 │
                                                                  ├──▶ Stage 3 (validate vs ClinVar)
                                                                  └──▶ Stage 4 (severity tiers + CIs)
```


 Ariana Lotfi — Dataset Curation, MAVE Processing & Pipeline Development
  
  - Identified and sourced the two MAVE datasets used for functional labels (urn:mavedb:00001222-a-2,
  urn:mavedb:00001222-b-2; Adamovich et al. 2022), covering HDR activity and cisplatin resistance assays across
  BRCA1 BRCT domain variants
  - Built the MAVE data processing pipeline: filtering to single missense variants, z-score normalization,
  directional alignment of scores, and outer join to produce a unified mave_score and per-variant coverage metric
  - Curated and maintained the master variant dataset (brca1_final.csv), including merging AlphaMissense
  pathogenicity scores with MAVE functional data via inner join on protein position and amino acid identity
  - Organized the repository data structure, separating the model-ready dataset from intermediate/prior versions
