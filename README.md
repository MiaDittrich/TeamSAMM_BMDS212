# TeamSAMM_BMDS212


 Ariana Lotfi — Dataset Curation, MAVE Processing & Pipeline Development
  
  - Identified and sourced the two MAVE datasets used for functional labels (urn:mavedb:00001222-a-2,
  urn:mavedb:00001222-b-2; Adamovich et al. 2022), covering HDR activity and cisplatin resistance assays across
  BRCA1 BRCT domain variants
  - Built the MAVE data processing pipeline: filtering to single missense variants, z-score normalization,
  directional alignment of scores, and outer join to produce a unified mave_score and per-variant coverage metric
  - Curated and maintained the master variant dataset (brca1_final.csv), including merging AlphaMissense
  pathogenicity scores with MAVE functional data via inner join on protein position and amino acid identity
  - Organized the repository data structure, separating the model-ready dataset from intermediate/prior versions
