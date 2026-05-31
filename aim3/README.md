# BRCA1 ClinVar Variant Predictor

Predicts BRCA1 MAVE functional scores from structural, biochemical,
evolutionary, and AlphaMissense features, then classifies 20 held-out
ClinVar variants as **pathogenic** or **benign**.

## Files

| File | Description |
|------|-------------|
| `brca1_predict.py` | Main script — train, predict, save results |
| `brca1_mave_pipeline_v2.py` | Full research pipeline (all models, ablation, comparison plots) |
| `BRCA1_FEATS.csv` | 178 BRCA1 variants with structural/biochemical/evo features |
| `BRCA1_MAVE.csv` | MAVE assay scores + AlphaMissense pathogenicity |
| `requirements.txt` | Python dependencies |
| `results/` | Generated outputs (CSV, plot, saved model) |

## Quick start

```bash
pip install -r requirements.txt
python brca1_predict.py                    # RF model, 'both' label (defaults)
python brca1_predict.py --model ridge      # Ridge regression
python brca1_predict.py --labels cisplatin # use cisplatin scores only
python brca1_predict.py --save-model       # write trained_model.joblib
python brca1_predict.py --load-model       # skip training, reload saved model
```

## Outputs

All files are written to `results/` (configurable with `--outdir`):

| File | Contents |
|------|----------|
| `clinvar_predictions.csv` | Predicted scores, 95 % bootstrap CIs, and ClinVar labels for all 20 test variants |
| `metrics_summary.csv` | CV regression metrics (Spearman, R², Pearson, RMSE) + ClinVar classification metrics (AUROC, accuracy, sensitivity, specificity) |
| `clinvar_plot.png` | Predicted score plot with error bars, coloured by ClinVar label |
| `trained_model.joblib` | Serialised sklearn pipeline (written with `--save-model`) |

## Model

The default model is **Random Forest** trained on the `both` label
(average of cisplatin and HDR activity scores) with:

- **KNN imputation** (k = 5 neighbours) for missing feature values
- **3 engineered features**: am_pathogenicity², AlphaMissense × log-EvoEF2
  interaction, pLDDT-weighted CA displacement
- **5-fold × 5-repeat cross-validation** for performance estimation

All hyperparameters are documented in the `PARAMETERS` block at the top of
`brca1_predict.py` and can be edited directly or overridden via CLI flags.

## ClinVar test set (20 variants)

All variants are confirmed present in **both** `BRCA1_FEATS.csv` and
`BRCA1_MAVE.csv`, guaranteeing complete AlphaMissense scores.

| Pathogenic (10) | Benign (10) |
|-----------------|-------------|
| S1841R | D1733G |
| L1839S | I1766V |
| V1838E | T1773S |
| M1775R | V1736I |
| Y1703S | K1793Q |
| W1718L | E1794G |
| G1706R | S1797C |
| G1738E | H1862L |
| S1715R | P1831S |
| I1760S | E1829T |

## Results

**Default run** (RF, `both` label, n = 70 training variants):

| Metric | Value |
|--------|-------|
| CV Spearman ρ | 0.564 ± 0.189 |
| CV R² | 0.320 ± 0.643 |
| CV Pearson r | 0.682 |
| CV RMSE | 0.362 |
| ClinVar AUROC | **1.000** |
| ClinVar Accuracy | **19/20 (95 %)** |
| Sensitivity | **10/10 (100 %)** |
| Specificity | 9/10 (90 %) |

The single misclassification is **H1862L** (benign → predicted pathogenic),
a structurally unusual substitution whose 95 % bootstrap CI crosses the
classification threshold, indicating genuine model uncertainty.

## Full research pipeline

`brca1_mave_pipeline_v2.py` runs all 5 models (RF, Ridge, XGBoost, LightGBM,
Stacking), all 4 label strategies, ablation studies, feature importance
analysis, and a v1 vs v2 improvement comparison. See its docstring for usage.

```bash
python brca1_mave_pipeline_v2.py --no-ablation    # faster run
python brca1_mave_pipeline_v2.py --model rf --labels both
```
