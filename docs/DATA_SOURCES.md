# Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| Sahni (PNAS 2015) | Supplementary data from paper | RefSeq IDs; 562 unique proteins |
| Fragoza (Nat Commun 2019) | Supplementary data | UniProt IDs |
| VarChAMP | Unpublished (IGVF consortium) | Combined dataset (VC1p + CAVA + 2026-batch + pooled cohorts, merged as "VCFP"/`varchamp_full_pooled`); cross-referenced via IGVF special issue; excluded from public release |
| Tsuboyama (Science 2023) | [doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6) | MegaScale stability pretraining data; train/val/test splits (`mega_splits.pkl`) from [SPURS](https://github.com/Tsuboyama-lab/SPURS) |
| ClinVar | clinvar.ncbi.nlm.nih.gov | January 2, 2025 release. Pathogenic (P/LP) and benign (B/LB) with ≥1 review star; all missense VUS |
| gnomAD | gnomad.broadinstitute.org | v4.1.0, all chromosomes; AFs assigned via GroupMax flag |
| COSMIC | cancer.sanger.ac.uk | v101. License required; retained only genes assigned to a single oncogene/TSG class |
| HGMD | hgmd.cf.ac.uk | Professional 2025. License required; "DM" variants only ("DM?" excluded) |
| ASD | Fu et al. 2022 | Case variants specifically associated with autism spectrum disorder; labels in `variant_label_dict.pkl` |
| NDD | Pejaver et al. 2020 (MutPred2 paper), *Nature Communications* | Case/control variants across four neurodevelopmental disorders (ASD, intellectual disability, schizophrenia, epileptic encephalopathy); labels in `variant_label_dict.pkl` |
| BioGRID | thebiogrid.org | Release 4.4.244; physical binding evidence only, used to build the interactome for variant-repository partner selection |
| MINT (comparator method) | Ullanat et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_mint_embeddings.py` |
| PPLM (comparator method) | Liu et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_pplm_embeddings.py` |
| AlphaFold3 structures (this study) | Zenodo: [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) | Subject to AlphaFold Server Output Terms of Use. Covers both training/evaluation complexes (`datasets/af3_structures.tar.gz`) and variant-repository complexes for ClinVar/gnomAD/autism-NDD (`datasets/af3_structures_variant_dbs.tar` — individually gzipped `.cif.gz` files in an uncompressed tar, so a single structure can be extracted without decompressing the whole archive) — COSMIC/HGMD variant-DB structures excluded (licensing). |
| Trained models + Sahni/Fragoza training data | Zenodo: [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) | Post-AF3-structure-filtering; used for Fig 3 GCV |

## VarChAMP VC1p/CAVA gene→UniProt mapping

`home/varchamp1p/gene_symbol_to_uniprot.pkl` and `home/cava/gene_symbol_to_uniprot.pkl` map
gene-symbol-keyed VC1p/CAVA proteins to UniProt IDs. Generator (ported from a previously
uncontrolled legacy notebook into the repo): `src/data_processing/variant_databases/map_varchamp_gene_ids.py`
— uses the HGNC REST API (same approach as `map_cosmic.py`), preferring whichever UniProt ID
(for genes with multiple) has a precomputed AlphaFold structure available. Verified to give
100% coverage of every gene symbol actually used by the pipeline (567/567 for VC1p, 178/178
for CAVA, derived directly from the `af3_graphs/` contact-graph filenames rather than the
original raw scoring CSVs, which no longer exist on disk).

## Gene inheritance-mode (AR/AD) stratification

The ClinVar/HGMD AR-only vs. AD-only disease-gene comparisons (Fig 5, S4, S-stability) use:
- **ClinGen Gene-Disease Validity curations** (`ClinGen_MOI.csv`) — mode-of-inheritance per gene, as curated in: Chen Y, Fayer S, Jain S, Benazouz M, Sverchkov Y, Stone J, Sharma H, Bergquist T, Stewart R, Mooney SD, Craven M, Radivojac P, Starita LM, Fowler DM, Pejaver V. Gene- and domain-aware calibration increases the clinical utility of variant effect predictors. bioRxiv [Preprint]. 2026 Mar 31:2026.02.17.706269. doi: [10.64898/2026.02.17.706269](https://doi.org/10.64898/2026.02.17.706269). PMID: 41756877; PMCID: PMC12934735. A gene counts as "AR-only" only if its full curated MOI set is exactly `{AR}` (similarly for AD); genes with any X-linked, mixed, or other MOI are excluded from both groups.
- **Gene→UniProt mapping**: primary via `gene_symbol_to_uniprot.pkl`, with a `gnomad_uniprot_to_gene.tsv`-based fallback for HGNC-renamed synonyms.

See [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md) for the exact commands.

## Licensing / exclusions

VarChAMP raw data is unpublished IGVF consortium data and is excluded from all public releases (git, Zenodo). COSMIC and HGMD variant-partner interaction data are excluded due to commercial/academic licensing restrictions — obtain directly from their respective sources using the versions above. gnomAD and ClinVar must also be downloaded directly (public, no redistribution restriction, but not bundled here for size reasons).
