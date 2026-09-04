# Running Inference at Scale

This covers the public 3-step inference pipeline (`src/inference/`) for scoring your own variant/partner sets. For a minimal working example, see the [Quick Start](../README.md#quick-start) in the main README.

## Step 1: Prepare AlphaFold3 Input Files (Optional)

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

## Step 1.5: Obtain Protein Complex Structures

MutPred-PPI requires protein complex structures in mmCIF format. You can use structures from any source:

### Option A: AlphaFold3 Structures
Submit protein complex queries to the [AlphaFold3 Server](https://alphafoldserver.com/) using the JSON files from Step 1, or generate structures locally if you have access to AlphaFold3.

**Note:** AlphaFold3 structures are subject to AlphaFold3's Terms of Use (non-commercial use only). See their [terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) for details.

### Option B: Experimental Structures
Download experimental structures from the [Protein Data Bank](https://www.rcsb.org/). Convert to mmCIF format if needed.

### Option C: Other Structure Prediction Tools
Use any other structure prediction tool that outputs mmCIF or PDB format (e.g., AlphaFold-Multimer, RoseTTAFold).

Save all mmCIF structure files to `<mmcif_dir>` for use in Step 2. **Important:** The sequences in the structure file must exactly match the sequences in the FASTA files.

## Step 2: Generate Contact Graphs

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

## Step 3: Run MutPred-PPI Inference

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

## Full Example Workflow

A complete minimal example using 10 test variants is provided in `example/`.

**Note:** Example AlphaFold3 structures are subject to AlphaFold 3 Output Terms of Use and provided for non-commercial research only. See: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

```bash
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
pip install -r src/inference/requirements.txt --upgrade
```

**Conda environment issues:**
```bash
# If having package conflicts, create fresh environment
conda deactivate
conda env remove -n mutpred-ppi
conda create -n mutpred-ppi python=3.9 -y
conda activate mutpred-ppi
# Then reinstall following Installation steps in the main README
```
