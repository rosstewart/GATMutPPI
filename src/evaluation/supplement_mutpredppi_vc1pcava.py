#!/usr/bin/env python
"""MutPred-PPI supplemental blind test on VC1p + CAVA entries.

These entries are in the SFVCFP pkl but were excluded from the main VCFP blind
test because their graphs use gene-name/Entrez IDs rather than UniProt IDs.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl (same mapping
used in load_sahni_fragoza_varchamp1p_cava() for GCV construction). C1/C2/C3
classification uses the remapped UniProt IDs against the SF protein set.

22 entries are shared between VC1p and CAVA — they are deduplicated here.

Usage:
    conda run -n ppi OPENBLAS_NUM_THREADS=1 python supplement_mutpredppi_vc1pcava.py [--device cuda:1]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_CV_MOD  = Path("/data/ross/ppi_lossgain/interaction_loss/publication/src/evaluation")
_INF_MOD = Path("/data/ross/ppi_lossgain/interaction_loss/publication/src/inference")

# Import model_loader BEFORE cv.py to prevent src/evaluation/utils/ shadowing.
sys.path.insert(0, str(_INF_MOD))
from utils.model_loader import MutPred_PPI, model_predict  # noqa: E402

sys.path.insert(0, str(_CV_MOD))
from mutpred_ppi_cv import (  # noqa: E402
    _load_varchamp1p_raw, _load_cava_raw,
    get_gene_name, split_wt_id_underscore,
)

_MODEL_PATH = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/models_sahni_fragoza/"
    "MutPred-PPI_sahni_fragoza_megascale_all_all.pt"
)
_SCALER_PATH = Path(
    "/data/ross/ppi_lossgain/interaction_loss/megascale_preprocessed/"
    "mutation_diff_scaler.pkl"
)
_TRAINING_CSV = Path(
    "/home/rcstewart/mutppi/benchmark/training_data.csv"
)
_VC1P_MAP  = Path("/data/ross/ppi_lossgain/interaction_loss/home/varchamp1p/gene_symbol_to_uniprot.pkl")
_CAVA_MAP  = Path("/data/ross/ppi_lossgain/interaction_loss/home/cava/gene_symbol_to_uniprot.pkl")
_OUT_DIR   = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)
_DESCRIPTION = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"


def build_sf_proteins() -> set[str]:
    df = pd.read_csv(_TRAINING_CSV)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return set(df.loc[sf_mask, "interactor"]) | set(df.loc[sf_mask, "partner"])


def classify(u1: str, u2: str, sf_proteins: set[str]) -> int:
    a_in = u1 in sf_proteins
    b_in = u2 in sf_proteins
    if a_in and b_in:
        return 1
    if a_in or b_in:
        return 2
    return 3


def load_model(device: torch.device):
    m = MutPred_PPI(input_dim=1024).to(device)
    m.load_state_dict(torch.load(str(_MODEL_PATH), map_location=device, weights_only=True))
    m.eval()
    print(f"Loaded: {_MODEL_PATH}")
    return [m]


def run_inference(sources: list[tuple[dict, dict]], sf_proteins: set, models, scaler, device):
    """Run inference over vc1p and cava sources, dedup on UniProt vt_ids.

    sources: list of (data_dict, gs2u_dict) pairs
    Returns arrays: scores, labels, vt_ids_out (UniProt format), class_labels
    """
    seen: set[str] = set()
    scores, labels, vt_ids_out, classes = [], [], [], []
    n_ok = n_skip_dup = n_skip_missing_map = n_invalid = 0

    for data, gs2u in sources:
        for i, vt_id in enumerate(data["all_vt_ids"]):
            wt_id = data["all_wt_ids"][i]
            variant = vt_id.split(" ", 1)[1]
            mut_idx = int(variant[1:-1])  # 0-based (vc1pcava convention)

            # Remap to UniProt
            try:
                a, b = split_wt_id_underscore(wt_id)
                u1 = gs2u[get_gene_name(a)]
                u2 = gs2u[get_gene_name(b)]
            except (KeyError, ValueError):
                n_skip_missing_map += 1
                continue

            uniprot_vt_id = f"{u1} {u2} {variant}"

            if uniprot_vt_id in seen:
                n_skip_dup += 1
                continue
            seen.add(uniprot_vt_id)

            combined = data["prott5_embeddings"][i]
            em       = data["edge_mats"][i]
            raw_diff = data["mutation_site_diffs"][i]
            diff_scaled = scaler.transform(raw_diff.reshape(1, -1)).squeeze()

            score = model_predict(combined, em, models, mut_idx, diff_scaled, device)
            if score is None:
                n_invalid += 1
                continue

            label = 1 if len(data["pos_labels"][i]) > 0 else 0
            c = classify(u1, u2, sf_proteins)

            scores.append(float(score))
            labels.append(label)
            vt_ids_out.append(uniprot_vt_id)
            classes.append(c)
            n_ok += 1

    print(f"  ok={n_ok}  dup_skip={n_skip_dup}  map_miss={n_skip_missing_map}  invalid={n_invalid}")
    return (
        np.array(scores, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        np.array(vt_ids_out),
        np.array(classes, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Building SF protein set…")
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins")

    print("Loading UniProt maps…")
    vc1p_gs2u = pickle.load(open(_VC1P_MAP, "rb"))
    cava_gs2u = pickle.load(open(_CAVA_MAP, "rb"))

    print("Loading VC1p data…")
    vc1p = _load_varchamp1p_raw()
    print(f"  {len(vc1p['all_vt_ids'])} entries")

    print("Loading CAVA data…")
    cava = _load_cava_raw()
    print(f"  {len(cava['all_vt_ids'])} entries")

    print("Loading model…")
    models = load_model(device)

    print("Loading scaler…")
    scaler = joblib.load(str(_SCALER_PATH))

    print("\nRunning inference (UniProt remapping + dedup)…")
    scores, labels, vt_ids_out, classes = run_inference(
        [(vc1p, vc1p_gs2u), (cava, cava_gs2u)],
        sf_proteins, models, scaler, device,
    )

    print(f"\nTotal unique entries: {len(scores)}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{_DESCRIPTION}_vc1pcava"

    for c in [1, 2, 3]:
        mask = classes == c
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        if mask.sum() == 0:
            auc_str = "n/a"
        elif n_pos < 2 or n_neg < 2:
            auc_str = "too few"
        else:
            auc_str = f"{roc_auc_score(labels[mask], scores[mask]):.4f}"
        print(f"  C{c}: n={mask.sum()} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(_OUT_DIR / f"{suffix}_c{c}_preds.npy",  scores[mask])
        np.save(_OUT_DIR / f"{suffix}_c{c}_labels.npy", labels[mask])
        np.save(_OUT_DIR / f"{suffix}_c{c}_vt_ids.npy", vt_ids_out[mask])

    print(f"\nSaved → {_OUT_DIR}/{suffix}_c{{1,2,3}}_*.npy")


if __name__ == "__main__":
    main()
