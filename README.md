# MutPred-PPI

Official repository for "Predicting interaction-specific protein–protein interaction perturbations by missense variants with MutPred-PPI", published in RECOMB 2026.

## Paper Links

- **RECOMB 2026 Proceedings**  
  https://recomb.org/proceedings/proceedings/2030-2026/2026/

- **PDF link**  
  https://doi.org/10.64898/2025.12.20.695738

## Overview

MutPred-PPI is a deep learning framework that predicts whether missense mutations disrupt protein-protein interactions. It combines structural information from protein complexes with sequence embeddings from protein language models to achieve high-accuracy predictions.

**Key Features:**
- Graph neural networks with attention mechanisms for structural analysis
- ProtT5 protein language model embeddings for sequence representation
- Binary classification: probabilistic score ranging from 0-1, where 1 indicates high probability of interaction disruption and 0 indicates preserved interaction
- Parallel processing support for large-scale analysis

## Installation

```bash
# Clone repository
git clone https://github.com/rosstewart/mutpred-ppi.git
cd mutpred-ppi

# Create conda environment
conda create -n mutpred-ppi python=3.9 -y
conda activate mutpred-ppi

# Install sentencepiece (required for ProtT5)
conda install sentencepiece -c conda-forge -y

# Install PyTorch
# For GPU with CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r src/inference/requirements.txt
```

### System Requirements

- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended
- ~4GB disk space for models and data


## Quick Start

```bash
# Step 1: Prepare AlphaFold3 inputs (if using AlphaFold3)
python src/inference/00_make_af3_json_input.py proteins.fasta variants.tsv af3_inputs/

# Step 2: Obtain protein complex structures (see Step 1.5 below)

# Step 3: Generate contact graphs
python src/inference/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# Step 4: Run predictions
python src/inference/02_run_mutpred-ppi_inference.py working_dir/
```

## Data Availability

Pre-trained model weights, Sahni+Fragoza training data (post-AF3 structure filtering, as used in Fig 3 GCV), and AF3 complex structures are available on Zenodo:
- **Models + training data**: https://doi.org/10.5281/zenodo.17645488
- **AF3 structures**: https://doi.org/10.5281/zenodo.18701748

VarChAMP data was unpublished IGVF consortium data at time of release and is excluded. COSMIC and HGMD require licensed access and are not distributed. gnomAD and ClinVar data must be downloaded from their respective public portals.

## Detailed Usage

### Step 1: Prepare AlphaFold3 Input Files (Optional)

If using AlphaFold3 for structure generation, prepare JSON files for submission:

```bash
python src/inference/00_make_af3_json_input.py \
    <fasta_file> \
    <triplet_tsv> \
    <output_directory> \
    [--seeds N]
```

**Arguments:**
- `fasta_file`: FASTA file containing all protein sequences
- `triplet_tsv`: TSV file with columns (protein_a, variant, protein_b)
- `output_directory`: Where to save JSON files
- `--seeds`: Number of AlphaFold3 model seeds (1-5, default: 1)

**Input Format:**

FASTA file:
```
>PROT1
MKTLLILAVVAAALA...
>PROT2
MSEQNNTEMTFQIQR...
```

TSV file:
```
PROT1	V123A	PROT2
PROT1	G456D	PROT3
PROT2	W89R	PROT3
```

### Step 1.5: Obtain Protein Complex Structures

MutPred-PPI requires protein complex structures in mmCIF format. You can use structures from any source:

#### Option A: AlphaFold3 Structures
Submit protein complex queries to the [AlphaFold3 Server](https://alphafoldserver.com/) using the JSON files from Step 1, or generate structures locally if you have access to AlphaFold3. 

**Note:** AlphaFold3 structures are subject to AlphaFold3's Terms of Use (non-commercial use only). See their [terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) for details.

#### Option B: Experimental Structures
Download experimental structures from the [Protein Data Bank](https://www.rcsb.org/). Convert to mmCIF format if needed.

#### Option C: Other Structure Prediction Tools
Use any other structure prediction tool that outputs mmCIF or PDB format (e.g., AlphaFold-Multimer, RoseTTAFold).

Save all mmCIF structure files to `<mmcif_dir>` for use in Step 2. **Important:** The sequences in the structure file must exactly match the sequences in the FASTA files.

### Step 2: Generate Contact Graphs

Process structures to create residue contact graphs:

```bash
python src/inference/01_make_contact_graphs_and_fasta.py \
    <working_dir> \
    <mmcif_dir> \
    <variants_file> \
    [n_jobs]
```

**Arguments:**
- `working_dir`: Output directory for graphs and sequences
- `mmcif_dir`: Directory containing structure files (.cif or .mmcif)
- `variants_file`: Same TSV file from Step 1
- `n_jobs`: Number of parallel jobs (default: 1). Parallel processing is recommended for large-scale analysis. Large protein complexes may take several minutes to process.

**Outputs:**
- `working_dir/af3_graphs/`: Contact graph matrices (.mat and helper files)
- `working_dir/wt_and_vt.fasta`: Combined wild-type and variant sequences for ProtT5 embedding generation

### Step 3: Run MutPred-PPI Inference

Predict interaction disruption for all variants:

```bash
python src/inference/02_run_mutpred-ppi_inference.py \
    <working_dir> \
    [--device DEVICE]
```

**Arguments:**
- `working_dir`: Directory from Step 2 containing graphs and FASTA
- `--device`: Compute device (default: cuda:0, use 'cpu' if no GPU available)

**Output:**
- `working_dir/results/MutPred-PPI_preds.tsv`: Prediction scores for each input variant
  - Tab-separated format with headers: `complex_id`, `variant`, `score`

## File Formats

### Variant Notation

Variants use standard notation: `[WT_residue][position][MT_residue]`
- Example: `V123A` (Valine at position 123 to Alanine)
- Position numbering starts at 1
- Use single-letter amino acid codes

### Structure Files

The pipeline accepts mmCIF files with flexible naming:
- `PROT1_PROT2.cif`
- `prefix_PROT1_PROT2_suffix.mmcif`
- Case-insensitive matching supported

## Example Workflow

A complete minimal example using 10 test variants is provided in `example/`.

**Note:** Example AlphaFold3 structures are subject to AlphaFold 3 Output Terms of Use and provided for non-commercial research only. See: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

```bash
# Activate conda environment (if using conda)
conda activate mutpred-ppi

# 1. Generate AlphaFold3 inputs (if using AlphaFold3)
python src/inference/00_make_af3_json_input.py \
    example/test_proteins.fasta \
    example/test_variants.tsv \
    example/

# NOTE: For this example, structure generation is already done
# Example structure: example/af3_models/fold_o00548_p46531_model_0.cif

# 2. Process structures to generate contact graphs
python src/inference/01_make_contact_graphs_and_fasta.py \
    example/ \
    example/af3_models/ \
    example/test_variants.tsv \
    1  # Limit to 1 parallel job

# 3. Run predictions
python src/inference/02_run_mutpred-ppi_inference.py \
    example/ \
    --device cuda:0

# View results
cat example/results/MutPred-PPI_preds.tsv
```

**Expected output:**
```
complex_id	variant	score
O00548_P46531	A653T	0.08524392545223236
O00548_P46531	R661S	0.06803165376186371
O00548_P46531	N34I	0.35405662655830383
...
```

## Project Structure

```
MutPred-PPI/
├── src/
│   ├── inference/                       # Public 3-step inference pipeline
│   │   ├── 00_make_af3_json_input.py
│   │   ├── 01_make_contact_graphs_and_fasta.py
│   │   ├── 02_run_mutpred-ppi_inference.py
│   │   ├── requirements.txt
│   │   └── utils/
│   ├── training/                        # Model training scripts
│   │   ├── preprocess_stability_data.py # Preprocess MegaScale data for pretraining
│   │   ├── pretrain_stability.py        # MegaScale stability pretraining
│   │   └── train_final_model.py         # Final PPI model training
│   ├── evaluation/                      # Cross-validation benchmarking
│   │   ├── mutpred_ppi_cv.py            # MutPred-PPI grouped cross-validation
│   │   ├── esignet_cv.py
│   │   ├── mint_cv.py
│   │   ├── pplm_cv.py
│   │   ├── swing_cv.py
│   │   └── saambe3d_cv.py
│   ├── data_processing/                 # Dataset preparation
│   │   ├── training_sets/
│   │   └── variant_databases/           # map_gnomad.py, map_cosmic.py, etc.
│   ├── variant_db_inference/            # Large-scale variant DB scoring
│   │   ├── precompute_prott5.py
│   │   └── run_variant_db_inference.py
│   └── analysis/                        # Figure generation + paper analyses
│       ├── roc_plots.py
│       ├── variant_db_charts.py
│       ├── varchamp_blind_test.py
│       └── ...
├── figures/                             # LaTeX table outputs (generated)
│   ├── training_data_table.tex
│   └── variant_db_stats_table.tex
├── example/                             # Minimal worked example inputs
│   ├── test_proteins.fasta
│   └── test_variants.tsv
├── LICENSE
└── README.md
```

Model weights and training data are distributed via Zenodo (see [Data Availability](#data-availability) above).

## Reproducing Paper Results

### Downloads

Download model weights and training data from Zenodo before proceeding:
- **Model weights**: `model_weights/` directory (see Zenodo)
- **Training data**: `datasets/sahni_fragoza_training_data.csv`, `datasets/sahni_training_data.csv`
- **AF3 structures** (for training graph regeneration): see the AF3 structures Zenodo entry

### External Path Prerequisites

The training and evaluation scripts contain hardcoded absolute paths that must exist on your machine. Update these constants if running on a different system:

| Constant | File | Default path |
|----------|------|-------------|
| `_CV_DIR` | `src/evaluation/mutpred_ppi_cv.py` | `/home/rcstewart/gnn/ppi_interaction_loss/cv_splits` |
| `_MEGASCALE_PRETRAINED_PATH` | `src/evaluation/mutpred_ppi_cv.py` | `/data/ross/gnn/jose_2016_lossgain_models/gnn_prott5_megascale_pretrain.pt` |
| `_MEGASCALE_SCALER_PATH` | `src/evaluation/mutpred_ppi_cv.py` | `/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/mutation_diff_scaler.pkl` |
| MINT cache | `src/evaluation/precompute_mint_embeddings.py` | `/data/ross/ppi_lossgain/interaction_loss/nm_revisions/mint_cache.pkl` |
| PPLM cache | `src/evaluation/precompute_pplm_embeddings.py` | `/data/ross/ppi_lossgain/interaction_loss/nm_revisions/pplm_cache.pkl` |

### Phase 0: MegaScale Stability Pretraining

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

### Phase 1: Training Data Preparation

Contact graphs (`.mat` files) are derived from AF3 structures and are not distributed directly. Generate them using:

```bash
# For each protein pair, generate contact graph from AF3 mmCIF
conda run -n ppi python src/inference/01_make_contact_graphs_and_fasta.py \
    working_dir/ af3_structures/ variants.tsv
```

CV splits (fold assignments) are pre-computed and stored in `_CV_DIR`. These define the 30-seed grouped cross-validation used for all Fig 3 benchmarking.

### Phase 2: Model Training

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

### Phase 3: Grouped Cross-Validation (Fig 3, S1)

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

### Phase 4: VarChAMP Blind Test (Fig 4)

```bash
# Run blind test for each method (saves per-class npy arrays)
conda run -n ppi python src/evaluation/mutpred_ppi_cv.py \
    --dataset sahni_fragoza_varchamp_full_pooled ...

# Enforce consistent test-set membership across methods
conda run -n ppi python src/analysis/restratify_vcfp_blind_test.py

# Generate Fig 4 from per-class arrays
conda run -n ppi python src/analysis/varchamp_blind_test.py
```

### Phase 5: Variant Database Inference (Fig 5–7)

ProtT5 embeddings must be precomputed before inference (large, resume-safe):

```bash
nohup conda run -n ppi python src/variant_db_inference/precompute_prott5.py \
    --fasta /path/to/gnomad_wt_and_vt.fasta --out prott5_embeddings.h5 \
    --device cuda:0 > precompute.log 2>&1 &

conda run -n ppi python src/variant_db_inference/run_variant_db_inference.py \
    --dataset gnomad --device cuda:0
```

**Note:** HGMD and COSMIC datasets require licensed access. COSMIC can be enabled with `--include-cosmic`. HGMD is excluded from all distributed files.

### Phase 6: Variant DB Classification and Chart Generation

```bash
conda run -n ppi python src/analysis/classify_variant_dbs.py
conda run -n ppi python src/analysis/variant_db_charts.py
```

### Phase 7: Table Generation

```bash
# Training data statistics table (figures/training_data_table.tex)
conda run -n ppi python src/analysis/generate_training_table.py

# Variant repository statistics table (figures/variant_db_stats_table.tex)
conda run -n ppi python src/analysis/extract_variant_db_stats.py
```

### Phase 8: ROC/AUC Figures

```bash
conda run -n ppi python src/analysis/roc_plots.py
conda run -n ppi python src/analysis/protein_class_stratification.py
```

## Performance

- **Inference speed**: ~100 variant-partner combinations/minute on GPU (V100) with precomputed structures
- **Memory usage**: ~4GB GPU memory for typical complexes (ProtT5 usage)

## Troubleshooting

### Common Issues

**CUDA out of memory error:**
```bash
# Use CPU instead
python src/inference/02_run_mutpred-ppi_inference.py working_dir/ --device cpu

# Or use a different GPU
python src/inference/02_run_mutpred-ppi_inference.py working_dir/ --device cuda:1
```

**Missing structures:**
- Ensure structure files contain both protein IDs in filename
- Check that files have .cif or .mmcif extension
- Verify protein IDs match between FASTA and TSV files

**Sequence mismatch errors:**
- Ensure sequences in structure files exactly match FASTA sequences
- Check for missing or extra residues in structure files
- Verify correct protein pairing in filenames

**Invalid amino acids in sequences:**
- Only standard 20 amino acids supported (ACDEFGHIKLMNPQRSTVWY)
- Remove non-standard residues or replace with closest standard amino acid

**Module import errors:**
```bash
# Ensure you're in the correct directory
cd mutpred-ppi/

# Reinstall dependencies
pip install -r src/requirements.txt --upgrade
```

**Conda environment issues:**
```bash
# If having package conflicts, create fresh environment
conda deactivate
conda env remove -n mutpred-ppi
conda create -n mutpred-ppi python=3.9 -y
conda activate mutpred-ppi
# Then reinstall following Installation steps
```

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Important Note:** While MutPred-PPI itself is open source, users must comply with the licensing terms of any structural data they use as input:
- **AlphaFold3 structures**: Subject to [AlphaFold3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) (non-commercial only)
- **PDB structures**: Check individual structure licenses
- **Other sources**: Comply with respective terms

## Citation

If you use MutPred-PPI in your research, please cite:

```bibtex
@inproceedings{stewart2026mutpred-ppi,
  title={Predicting interaction-specific protein--protein interaction perturbations by missense variants with MutPred-PPI},
  author={Stewart, Ross and Laval, Florent and Coppin, Georges and Spirohn-Fitzgerald, Kerstin and Tixhon, Maxime and Hao, Tong and Calderwood, Michael A and Mort, Matthew and Cooper, David N and Vidal, Marc and Radivojac, Predrag},
  booktitle={Proceedings of the 30th Annual International Conference on Research in Computational Molecular Biology (RECOMB)},
  year={2026},
  note={Also available as bioRxiv preprint (2025.12.20.695738)},
  doi={10.64898/2025.12.20.695738}
}
```

## Contact

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Email**: stewart.ro@northeastern.edu
