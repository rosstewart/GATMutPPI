# Training MutPred-PPI From Scratch

This covers pretraining and fine-tuning the model. For running inference with the pre-trained model, see [`docs/INFERENCE.md`](INFERENCE.md). For reproducing paper figures/tables, see [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md).

## Downloads

Download model weights and training data from Zenodo before proceeding (see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md) for links):
- **Model weights**: `model_weights/` directory
- **Training data**: `datasets/sahni_fragoza_training_data.csv`, `datasets/sahni_training_data.csv`
- **AF3 structures** (for training graph regeneration): AF3 structures Zenodo entry

## External Path Prerequisites

The training and evaluation scripts contain hardcoded absolute paths that must exist on your machine. Update these constants if running on a different system:

| Constant | File | Default path |
|----------|------|-------------|
| `_CV_DIR` | `src/evaluation/mutpred_ppi_cv.py` | `/home/rcstewart/gnn/ppi_interaction_loss/cv_splits` |
| `_MEGASCALE_PRETRAINED_PATH` | `src/evaluation/mutpred_ppi_cv.py` | `/data/ross/gnn/jose_2016_lossgain_models/gnn_prott5_megascale_pretrain.pt` |
| `_MEGASCALE_SCALER_PATH` | `src/evaluation/mutpred_ppi_cv.py` | `/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/mutation_diff_scaler.pkl` |
| MINT cache | `src/evaluation/precompute_mint_embeddings.py` | `/data/ross/ppi_lossgain/interaction_loss/nm_revisions/mint_cache.pkl` |
| PPLM cache | `src/evaluation/precompute_pplm_embeddings.py` | `/data/ross/ppi_lossgain/interaction_loss/nm_revisions/pplm_cache.pkl` |

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
