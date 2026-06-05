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
| **2. Model training & prediction** | `aim3/`, `code_repo/` | Train regressors (Random Forest et al.) on MAVE functional scores, then predict and classify held-out ClinVar variants. `code_repo/` holds the consolidated, publication-grade training + analysis pipeline (model comparison, SHAP importance, feature/group ablations, paper figures). | [aim3/README.md](aim3/README.md) |
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
cd aim3 && python3 brca1_predict.py && cd ..   # default: Random Forest, 'both' label
#   (brca1_predict.py reads BRCA1_FEATS.csv / BRCA1_MAVE.csv from aim3/, so run it
#    from inside aim3/. The consolidated pipeline lives in code_repo/ — run the
#    numbered scripts in order: python3 code_repo/01_model_comparison.py … 05_make_figures.py)

# Stage 3 — validate predictions against ClinVar ground truth
#   (reads clinvar_test_predictions_annotated.csv from the repo root; cwd does not matter)
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

**Stage 2 — model performance** (Random Forest, `both` label; `aim3/brca1_predict.py`
default run, 20-variant ClinVar hold-out):

| Metric | Value |
|--------|-------|
| CV Spearman ρ | 0.564 ± 0.189 |
| CV Pearson r | 0.682 |
| ClinVar AUROC | 1.000 |
| ClinVar accuracy | 19/20 (95%) |

**Stage 3 — external validation** on **28 ClinVar variants** (14 pathogenic / 14 benign):

| Test | Result |
|------|--------|
| Mann–Whitney U (distribution) | U = 0.0, p ≈ 7.5 × 10⁻⁶ |
| Kolmogorov–Smirnov (distribution) | D = 1.0, p ≈ 5.0 × 10⁻⁸ |
| Logistic regression AUROC (LOO-CV) | 1.000 |
| Average Precision | 1.000 |
| Accuracy @ fixed 0.5 cutoff | 27/28 (96.4%) |
| Sensitivity / Specificity @ 0.5 | 14/14 (100%) / 13/14 (92.9%) |

> **Reading the metrics:** AUROC / Average Precision are *threshold-free* and
> measure how well predicted scores **rank** pathogenic above benign
> (AUROC = 1.0 ⇒ some cutoff separates all 28 correctly).
> Accuracy/sensitivity/specificity depend on the probability cutoff and are
> reported only at the fixed, prespecified 0.5 cutoff — a data-driven "optimal"
> cutoff is deliberately *not* used, as tuning it on the labels would overstate
> performance. See [aim4/README.md](aim4/README.md) for the full explanation.

---

## Repository layout

```
.
├── feature_extraction/   # Stage 1 — AlphaFold structures → feature matrix
├── SAMM/                 # Stage 1 (earlier/parallel copy of the feature pipeline)
├── SAMM /                # Stage 1 (stray duplicate folder — note the trailing space)
├── aim3/                 # Stage 2 — model training + ClinVar prediction
│   ├── brca1_predict.py                    # main train/predict/save script
│   ├── brca1_mave_research_pipeline_v2.py  # full research pipeline
│   ├── BRCA1_FEATS.csv / BRCA1_MAVE.csv    # feature matrix + MAVE/AlphaMissense scores
│   ├── trained_model.joblib                # serialised sklearn pipeline
│   └── results/                            # predictions, metrics, figures
├── code_repo/            # Stage 2 — consolidated, publication-grade pipeline
│   ├── 01_model_comparison.py        # all models × label strategies
│   ├── 02_main_pipeline.py           # final Random Forest train/predict
│   ├── 03_ablations.py               # leave-one-feature/group-out ablations
│   ├── 04_combinatorial_ablations.py # all 31 feature-group subsets
│   ├── 05_make_figures.py            # paper figures (fig1–fig6)
│   ├── data/                         # feats, MAVE, clinvar_test (48), clinvar_explore (24)
│   ├── figures/                      # generated PNG/PDF figures
│   └── results/                      # CV metrics, SHAP, ablations, predictions
├── aim4/                 # Stage 3 — external validation
│   ├── 01_distribution_analysis.py
│   ├── 02_logistic_regression.py
│   └── results/                      # KDE, ROC, PR, confusion matrix, stats
├── pipeline_4.py         # Stage 4 — calibrated severity scores + CIs + tiers
├── brca1_mave_pipeline_v2.py  # ablation-pruned variant of the Stage 2 pipeline
├── clinvar_test_predictions_annotated.csv  # Stage 3 input (28 annotated ClinVar variants)
├── results/              # outputs from the top-level pipelines (SHAP, ablation, etc.)
├── validation/           # standalone ClinVar validation notes (clinvar_result.txt)
├── samm_sample_pdbs/     # example PDB structures
└── prior_versions/       # earlier/intermediate datasets (AlphaMissense, MaveDB, merges)
```

### Notes on auxiliary files
- `SAMM/` and `feature_extraction/` are two iterations of the Stage 1 feature
  pipeline; `feature_extraction/` is the cleaned-up version. `SAMM /` (with a
  trailing space in the name) is a stray duplicate left over from development.
- `code_repo/` is the consolidated Stage 2 deliverable (numbered scripts 01–05
  run in order). `aim3/` and `brca1_mave_pipeline_v2.py` are the working
  iterations it was distilled from; `02_main_pipeline.py` is documented as
  architecturally identical to `brca1_mave_pipeline_v2.py`. Note the different
  ClinVar splits in play: `aim3/brca1_predict.py` checks a 20-variant hold-out,
  `code_repo/` uses a 48-variant test set, and the committed Stage 3 validation
  (`aim4/`) runs on the 28-variant annotated set.
- `brca1_mave_pipeline_v2.py` is a feature-pruned variant of the Stage 2
  pipeline (drops features with negative leave-one-out ablation impact).
- Loose top-level CSVs (`brca1_final.csv`, `brca1_features_2.csv`,
  `brca1_model_features_lean.csv`, etc.) and `prior_versions/` hold intermediate
  and historical datasets used during development.

---

## Data flow

```
AlphaFold CIFs ──▶ Stage 1 ──▶ feature matrix ──┐
                                                ├──▶ Stage 2 ──▶ pred_score per variant
MAVE scores + AlphaMissense ────────────────────┘                 │
                                                                  ├──▶ Stage 3 (validate vs ClinVar)
                                                                  └──▶ Stage 4 (severity tiers + CIs)
```
---

## Teammate Contributions

** Ariana Lotfi — Dataset Curation, MAVE Processing & Pipeline Development**
  
  - Identified and sourced the two MAVE datasets used for functional labels (urn:mavedb:00001222-a-2,
  urn:mavedb:00001222-b-2; Adamovich et al. 2022), covering HDR activity and cisplatin resistance assays across
  BRCA1 BRCT domain variants
  - Built the MAVE data processing pipeline: filtering to single missense variants, z-score normalization,
  directional alignment of scores, and outer join to produce a unified mave_score and per-variant coverage metric
  - Curated and maintained the master variant dataset (brca1_final.csv), including merging AlphaMissense
  pathogenicity scores with MAVE functional data via inner join on protein position and amino acid identity
  - Organized the repository data structure, separating the model-ready dataset from intermediate/prior versions

** Margot Hutchins - Dataset Generation for Structural and Biochemical data 
  - Processed AF3 structure files and aligned to wildtype structure using high-confidence BRCT domain residues
  - Computed local and global structural features, e.g. RMSDs and Ramachandran angles
  - Imputed biochemical features of mutant residue
  - Joined structural and biochemical features with two energy stability metrics from EvoEF2
  - Dropped low-variance columns for data pruning

** Mia Dittrich

- Created mutagenesis script to insert selected missense mutations into the WT BRCA1 sequence to construct variants needed for 3D structure production (FASTA file output)
- Folding variant sequences in AlphaFold 3 to prepare for structural feature extraction
- External validation of predicted pathogenicity scores using held-out ClinVar dataset 

** Spencer Cha — Machine Learning Model training, evaluation, and analysis

- Folded some variant sequences in AlphaFold 3 to prepare for structure feature extraction
- Created machine learning model scripts to test different model architectures and label strategies
- Evaluated machine learning model with cross validation metrics and optimized model performance through hyperparameter tuning
- Validated feature importance by calculating SHAP values and performing feature ablation and feature group combinatorial ablation
- Created VUS / CIP variant sets and analyzed according to external ClinVar validation

