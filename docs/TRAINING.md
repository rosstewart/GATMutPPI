# Training MutPred-PPI From Scratch

This covers pretraining and fine-tuning the model. For running inference with the pre-trained model, see [`docs/INFERENCE.md`](INFERENCE.md). For reproducing paper figures/tables, see [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md).

## Downloads

Download model weights and training data from Zenodo before proceeding (see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md) for links):
- **Model weights**: `model_weights/` directory
- **Training data**: `datasets/sahni_fragoza_training_data.csv`, `datasets/sahni_training_data.csv`
- **AF3 structures** (for training graph regeneration): AF3 structures Zenodo entry

## Which model directory is used for what

There are three model-checkpoint directories in this repo, each with a distinct purpose —
using the wrong one for the wrong task produces a subtly wrong result rather than an error,
so this distinction matters:

| Directory | Model | Used for | Why |
|---|---|---|---|
| `weights/` (git-tracked) | Sahni+Fragoza+VarChAMP (SFVCFP) | Public-facing inference (`src/inference/02_run_mutpred-ppi_inference.py`), variant-repository inference (`run_variant_db_inference.py` → Fig 5, S4, S-stability, master CSV) | Not a blind test — using the model trained on the most data gives the best real-world predictions. |
| `models_sahni_fragoza/` (internal, gitignored) | Sahni+Fragoza only (no VarChAMP) | SF GCV (Fig 3), VCFP blind test (Fig 4) — including the VC1p/CAVA supplement (`supplement_mutpredppi_vc1pcava.py`) | Both are testing generalization to data the model never trained on. VCFP blind test specifically requires the SF-only model — using the SFVCFP model there would defeat the purpose of a *blind* test (the model would already have seen the VarChAMP data during training). |
| `model_weights/` (Zenodo staging, gitignored) | Copies of the above, renamed for public distribution | Zenodo upload only | Not read by any script directly — see `model_weights/README.md`. |

## External Path Prerequisites

The training and evaluation scripts contain hardcoded absolute paths that must exist on your machine. Update these constants if running on a different system:

| Constant | File | Default path |
|----------|------|-------------|
| `_CV_DIR` | `src/evaluation/mutpred_ppi_cv.py` | `/home/rcstewart/gnn/ppi_interaction_loss/cv_splits`. **Regenerable from scratch**: `src/training/generate_cv_splits.py --dataset sahni_fragoza --out-dir <path>` reproduces the exact fold-assignment splits (verified content-identical to the existing splits for seeds 0/15/29 — same train/test indices per fold, byte-identical `all_vt_ids` files). Reuses the already-in-repo `cluster_sequences()`/`load_sahni_fragoza()`. Note: row order depends on `glob.glob()` over the graph directory, which is stable on an unmodified filesystem but not a portable cross-machine guarantee — see the script's docstring. |
| `_MEGASCALE_PRETRAINED_PATH` | `src/evaluation/mutpred_ppi_cv.py` (imported by `train_final_model.py`) | **In-repo now**: `model_weights/gnn_prott5_megascale_pretrain.pt`. If missing, either download from Zenodo or run Phase 0 below to regenerate. Loading now fails with a clear, actionable error (not a bare traceback) if the file isn't present. |
| `_MEGASCALE_SCALER_PATH` | `src/evaluation/mutpred_ppi_cv.py` (imported by `train_final_model.py`) | **In-repo now**: `model_weights/mutation_diff_scaler.pkl` (verified byte-identical to the scaler produced by Phase 0 preprocessing). |
| MINT cache | `src/evaluation/precompute_mint_embeddings.py`, `src/evaluation/predictors/mint_mlp.py` | **In-repo now**: `data_caches/mint_cache.pkl` (both the precompute script's output default and the predictor's read default now point here — previously mismatched: `nm_revisions/mint_cache.pkl` vs `2026/mint_cache/mint_cache_v2.pkl`). Regenerate with `precompute_mint_embeddings.py`. |
| PPLM cache | `src/evaluation/precompute_pplm_embeddings.py`, `src/evaluation/predictors/pplm_mlp.py` | **In-repo now**: `data_caches/pplm_cache.pkl` (same path-mismatch fix as MINT above). Regenerate with `precompute_pplm_embeddings.py`. |
| MINT/PPLM/eSIG-Net predictor code | `src/evaluation/predictors/{mint_mlp,pplm_mlp,esignet,nn_base}.py`, `src/evaluation/esignet_gcv_iter_legacy.py` | **Vendored in-repo now** — previously imported via `sys.path.insert` from `/home/rcstewart/ppi_lossgain/2026/mutppi/benchmark/predictors/` and `esignet_scripts/`, entirely outside the repo. `mint_cv.py`/`pplm_cv.py`/`esignet_cv.py` and the 3 corresponding `supplement_*_vc1pcava.py` scripts now import from this in-repo location; verified all 6 still import and run cleanly. |
| `_TRAINING_CSV`/`_CSV`/`_TRAIN`/`TRAINING_CSV` (same file, single constant name per script) | `mutpred_ppi_cv.py`, `generate_training_table.py`, `restratify_vcfp_blind_test.py`, `varchamp_dataset_comparison.py`, `generate_pooled_t5_embs.py`, all 5 `supplement_*_vc1pcava.py` scripts | `/home/rcstewart/mutppi/benchmark/training_data.csv` — the master labeled Sahni+Fragoza+VarChAMP table. Internal-only (contains unpublished VarChAMP rows); not a Zenodo deposition target. Only the SF-only subset (`datasets/sahni_fragoza_training_data.csv`) is distributed. |
| `_PRETRAINED_PATH` / `_SCALER_PATH` | `src/evaluation/mutpred_ppi_cv.py` | Legacy pre-MegaScale (FoldX/RaSP-based) checkpoint + scaler, used only by the `full`/`full_all` ablation — this produces the "Prior Best" (MutPred-PPI v1.0) comparison bar in the S-abl ablation figure. Not regenerable via the current `pretrain_stability.py` pipeline (that produces the MegaScale checkpoint instead); this is a preserved artifact from the prior RECOMB v1.0 submission. If unavailable, skip the `full`/`full_all` ablation runs — all other ablations use the MegaScale checkpoint above. |

## Phase 0: MegaScale Stability Pretraining

This is required once before any model training. Source data from [Tsuboyama et al. 2023](https://doi.org/10.1038/s41586-023-06328-6).

```bash
# Preprocess MegaScale CSV + AlphaFold PDBs into training tensors
conda run -n ppi python src/training/preprocess_stability_data.py \
    --csv     /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --pdb-dir /path/to/AlphaFold_model_PDBs/ \
    --splits  /path/to/mega_splits.pkl \
    --outdir  /path/to/megascale_preprocessed/ \
    --device  cuda:0 --n-jobs 16
# Outputs: preprocessed.pkl, mutation_diff_scaler.pkl

# Pretrain GNN on stability data
conda run -n ppi python src/training/pretrain_stability.py \
    --data     /path/to/megascale_preprocessed/preprocessed.pkl \
    --scaler   /path/to/megascale_preprocessed/mutation_diff_scaler.pkl \
    --outmodel /path/to/gnn_prott5_megascale_pretrain.pt \
    --device   cuda:0
```

Update `_MEGASCALE_PRETRAINED_PATH` in `src/evaluation/mutpred_ppi_cv.py` to point to the output checkpoint.

## Phase 1: Training Data Preparation

Contact graphs (`.mat` files) are derived from AF3 structures and are not distributed directly. Generate them using:

```bash
# For each protein pair, generate contact graph from AF3 mmCIF
conda run -n ppi python src/inference/01_make_contact_graphs_and_fasta.py \
    working_dir/ af3_structures/ variants.tsv
```

CV splits (fold assignments) are pre-computed and stored in `_CV_DIR`. These define the 30-seed grouped cross-validation used for all Fig 3 benchmarking.

## Phase 2: Model Training

```bash
# Train Sahni+Fragoza model (primary, Fig 3)
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza --ablation megascale_all \
    --device cuda:0

# Train Sahni+Fragoza+VarChAMP model (full-data, Fig 4 / VarChAMP blind test)
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza_varchamp_full_pooled --ablation megascale_all \
    --device cuda:0
```
