# Reproducing Every Figure and Table

This covers cross-validation benchmarking, the VarChAMP blind test, variant-repository inference/classification/charts, and all supplementary analyses. For training the model from scratch, see [`docs/TRAINING.md`](TRAINING.md). External path prerequisites are listed there too.

## Phase 3: Grouped Cross-Validation (Fig 3, S1)

```bash
# MutPred-PPI GCV (30 seeds, Sahni+Fragoza dataset)
conda run -n ppi python src/evaluation/mutpred_ppi_cv.py \
    --dataset sahni_fragoza --ablation megascale_all \
    --device cuda:0

# Comparator methods (require external installations)
conda run -n ppi python src/evaluation/swing_cv.py    --dataset sahni_fragoza
conda run -n ppi python src/evaluation/esignet_cv.py  --dataset sahni_fragoza
conda run -n ppi python src/evaluation/mint_cv.py     --dataset sahni_fragoza
conda run -n ppi python src/evaluation/pplm_cv.py     --dataset sahni_fragoza
conda run -n ppi python src/evaluation/saambe3d_cv.py --dataset sahni_fragoza
```

Intermediate MINT/PPLM/eSIG-Net sequence embeddings must be precomputed first:

```bash
conda run -n ppi python src/evaluation/precompute_mint_embeddings.py
conda run -n ppi python src/evaluation/precompute_pplm_embeddings.py
# eSIG-Net: use eSIG-Net's own precompute script (see its repository)
```

### Biclass SF GCV (S-biclass)

Restricts Fig 3's cross-validation to "biclass" ordered protein pairs — pairs (A, B) where mutations in interactor A include both disruptive and non-disruptive labels.

```bash
conda run -n ppi python src/analysis/biclass_sf_gcv.py
```

Output:
- `results_revisions/biclass_gcv/roc_sahni_fragoza_biclass_with_variance.png` → **S-biclass**
- `results_revisions/biclass_gcv/biclass_c3_aurocs.tsv`

## Phase 4: VarChAMP Blind Test (Fig 4, S2)

The SFVCFP test set contains 22,321 entries from four VarChAMP sub-cohorts.
Two of these (VC1p: 1,377 entries and CAVA: 1,581 entries) use gene-name/Entrez
protein IDs rather than UniProt IDs. The main blind test scripts handle
VC2026+pooled entries (UniProt-keyed, from `training_data.csv`). The VC1p and
CAVA entries are processed separately via supplement scripts that remap gene
names to UniProt IDs using `gene_symbol_to_uniprot.pkl` (the same mapping used
in GCV construction), then classify each entry as C1/C2/C3.

```bash
# Step 4a: Run main VCFP blind test (VC2026 + pooled entries)
conda run -n ppi python src/evaluation/mutpred_ppi_cv.py \
    --dataset sahni_fragoza_varchamp_full_pooled ...
# (repeat for eSIG-Net, SWING, MINT, PPLM — see their respective cv scripts)

# Step 4b: Supplement VC1p+CAVA entries for each method
conda run -n ppi python src/evaluation/supplement_mutpredppi_vc1pcava.py --device cuda:0
conda run -n ppi python src/evaluation/supplement_esignet_vc1pcava.py --device cuda:0
conda run -n ppi python src/evaluation/supplement_mint_vc1pcava.py --predictor seq_diff
conda run -n ppi python src/evaluation/supplement_mint_vc1pcava.py --predictor site_diff
conda run -n ppi python src/evaluation/supplement_pplm_vc1pcava.py --predictor seq_diff
conda run -n ppi python src/evaluation/supplement_pplm_vc1pcava.py --predictor site_diff
conda run -n ppi python src/evaluation/supplement_swing_vc1pcava.py
conda run -n ppi python src/evaluation/supplement_swing_vc1pcava.py --test-pretrain

# Step 4c: Merge supplements into main per-class arrays
conda run -n ppi python src/analysis/merge_vc1pcava_into_main.py

# Step 4d: Enforce consistent test-set membership across methods
conda run -n ppi python src/analysis/restratify_vcfp_blind_test.py

# Step 4e: Generate Fig 4 + S2 from per-class arrays
conda run -n ppi python src/analysis/varchamp_blind_test.py
```

**Note:** MINT and PPLM caches must include vc1pcava protein embeddings. Run
`precompute_mint_embeddings.py --dataset sahni_fragoza_varchamp1p_cava` before
Step 4b. Methods that require PDB structures (SAAMBE-3D, MutPPI, MutPPI+,
DDMutPPI) cannot predict vc1pcava entries and are evaluated on the 14,116
VC2026+pooled entries only. MutPred2 does not require structures and covers all 17,052 entries.

## Phase 5: Variant Database Inference (Fig 5–7)

ProtT5 embeddings must be precomputed before inference (large, resume-safe):

```bash
nohup conda run -n ppi python src/variant_db_inference/precompute_prott5.py \
    --fasta /path/to/gnomad_wt_and_vt.fasta --out prott5_embeddings.h5 \
    --device cuda:0 > precompute.log 2>&1 &

conda run -n ppi python src/variant_db_inference/run_variant_db_inference.py \
    --dataset gnomad --device cuda:0
```

**Note:** HGMD and COSMIC datasets require licensed access. COSMIC can be enabled with `--include-cosmic`. HGMD is excluded from all distributed files.

## Phase 6: Variant DB Classification and Chart Generation

```bash
conda run -n ppi python src/analysis/classify_variant_dbs.py \
    --output-dir results_revisions/variant_dbs_sfvfp

conda run -n ppi python src/analysis/variant_db_charts.py \
    --data-dir results_revisions/variant_dbs_sfvfp \
    --edgotype-bootstrap --controlled-bootstrap --k3-only
```

Output:
- `enrichment_bootstrap_sufficient_partners.png` → **Fig 5** (ClinVar row includes Rare Benign/Benign/Pathogenic/VUS/Pathogenic AR/Pathogenic AD; HGMD row includes HGMD/AR/AD)
- `enrichment_bootstrap_sufficient_partners_k3.png` → **S4** (same grouping, partner-controlled)

### Gene inheritance-mode (AR/AD) mapping — prerequisite for Fig 5, S4, S-stability

The ClinVar/HGMD AR-only vs. AD-only comparisons require a gene→UniProt inheritance-mode mapping, built once:

```bash
conda run -n ppi python src/analysis/build_ar_ad_gene_sets.py
```

Output: `/data/ross/ppi_lossgain/interaction_loss/clingen_ar_ad_uniprot_sets.pkl` — `{"AR": set[uniprot], "AD": set[uniprot]}`, mutually exclusive, from `ClinGen_MOI.csv`. See [`docs/DATA_SOURCES.md`](DATA_SOURCES.md) for details.

This produces `results_revisions/variant_dbs_sfvfp/clinvar/{ar,ad}_pathogenic_{edgotype_classes.npy,posterior_ls.pkl}` and `results_revisions/variant_dbs_sfvfp/hgmd/{ar,ad}_hgmd_{edgotype_classes.npy,posterior_ls.pkl}`, consumed directly by the Phase 6 chart command above (no separate re-run step needed — `classify_variant_dbs.py` produces these alongside the base subsets in one pass).

### COSMIC Onco/TSG QN vs. Edgetic stat test (S-cosmic-stat)

```bash
conda run -n ppi python src/analysis/cosmic_onco_tsg_stat_test.py
```

Output: `results_revisions/cosmic_stat_test/cosmic_onco_tsg_qn_vs_edgetic.tex` → **S-cosmic-stat**

### Protein class enrichment (S-protclass)

```bash
conda run -n ppi python src/analysis/protein_class_enrichment.py
```

Output: `results_revisions/protein_class_enrichment/pathogenic_by_class.png` → **S-protclass**

### Stability vs. interaction per-variant scatter (S-stability, 6 panels)

Requires the AR/AD gene-set mapping above (used internally for data loading, even though AR/AD panels are not shown in the final figure).

```bash
conda run -n ppi python src/analysis/stability_interaction_scatter.py --cosmic-min-recurrence 32
```

Output: `results_revisions/stability_interaction/scatter_per_variant_kde.png` → **S-stability**
(6 panels: ClinVar Pathogenic, ClinVar Benign, ClinVar VUS, gnomAD, HGMD, COSMIC recurrence≥32)

### Robustness analyses

```bash
# Interface vs. non-interface AUROC stratification
conda run -n ppi python src/analysis/interface_analysis.py

# pLDDT quality stratification
conda run -n ppi python src/analysis/plddt_stratification.py

# Threshold sensitivity
conda run -n ppi python src/analysis/threshold_sensitivity.py

# Protein class stratification (single- vs. multi-domain)
conda run -n ppi python src/analysis/protein_class_stratification.py

# Combined 3-row figure (interface / pLDDT / protein-class in one image, panels A/B/C)
# reuses compute_curves()/plot_on_axes() from the three scripts above — run them
# first (or let this script call compute_curves() itself, which it does).
conda run -n ppi python src/analysis/combined_robustness_figure.py
```

Output: each standalone script still produces its own `*_auroc_by_class.png` + `.tsv`;
`combined_robustness_figure.py` additionally produces `combined_robustness_by_class.png`
(all three comparisons stacked as rows A/B/C in one figure).

Output in `results_revisions/robustness_analyses/`.

## Phase 7: Table Generation

```bash
# Training data statistics table (figures/training_data_table.tex)
conda run -n ppi python src/analysis/generate_training_table.py

# Variant repository statistics table (figures/variant_db_stats_table.tex)
conda run -n ppi python src/analysis/extract_variant_db_stats.py
```

## Phase 8: ROC/AUC Figures

```bash
conda run -n ppi python src/analysis/roc_plots.py
```

## Ablation figure (S-abl)

```bash
conda run -n ppi python src/analysis/run_roc_ablation.py
```

Output: `results_revisions/macro_aucs/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` → **S-abl**
