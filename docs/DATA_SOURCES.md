# Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| Sahni (PNAS 2015) | Supplementary data from paper | RefSeq IDs; 562 unique proteins |
| Fragoza (Nat Commun 2019) | Supplementary data | UniProt IDs |
| VarChAMP 2026 | Unpublished (IGVF consortium) | Cross-referenced via IGVF special issue; excluded from public release |
| Tsuboyama (Science 2023) | [doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6) | MegaScale stability pretraining data |
| ClinVar | clinvar.ncbi.nlm.nih.gov | January 2, 2025 release. Pathogenic (P/LP) and benign (B/LB) with ≥1 review star; all missense VUS |
| gnomAD | gnomad.broadinstitute.org | v4.1.0, all chromosomes; AFs assigned via GroupMax flag |
| COSMIC | cancer.sanger.ac.uk | v101. License required; retained only genes assigned to a single oncogene/TSG class |
| HGMD | hgmd.cf.ac.uk | Professional 2025. License required; "DM" variants only ("DM?" excluded) |
| ASD / NDD | Fu et al. 2022; Tulika et al. | Case/control labels in `variant_label_dict.pkl` |
| BioGRID | thebiogrid.org | Release 4.4.244; physical binding evidence only, used to build the interactome for variant-repository partner selection |
| AlphaFold3 structures (this study) | Zenodo: [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) | Subject to AlphaFold Server Output Terms of Use |
| Trained models + Sahni/Fragoza training data | Zenodo: [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) | Post-AF3-structure-filtering; used for Fig 3 GCV |

## Gene inheritance-mode (AR/AD) stratification

The ClinVar/HGMD AR-only vs. AD-only disease-gene comparisons (Fig 5, S4, S-stability) use:
- **ClinGen Gene-Disease Validity curations** (`ClinGen_MOI.csv`) — mode-of-inheritance per gene. A gene counts as "AR-only" only if its full curated MOI set is exactly `{AR}` (similarly for AD); genes with any X-linked, mixed, or other MOI are excluded from both groups.
- **Gene→UniProt mapping**: primary via `gene_symbol_to_uniprot.pkl`, with a `gnomad_uniprot_to_gene.tsv`-based fallback for HGNC-renamed synonyms.

See [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md) for the exact commands.

## Licensing / exclusions

VarChAMP raw data is unpublished IGVF consortium data and is excluded from all public releases (git, Zenodo). COSMIC and HGMD variant-partner interaction data are excluded due to commercial/academic licensing restrictions — obtain directly from their respective sources using the versions above. gnomAD and ClinVar must also be downloaded directly (public, no redistribution restriction, but not bundled here for size reasons).
