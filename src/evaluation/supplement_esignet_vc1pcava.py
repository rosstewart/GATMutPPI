#!/usr/bin/env python
"""eSIG-Net supplemental blind test on VC1p + CAVA entries.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl. The ESM-2
cache is extended with precomputed supplement PKLs that cover vc1pcava mutant
and WT embeddings not already in esm2_residue_embeddings_with_pooled.pkl.

22 entries shared between VC1p and CAVA are deduplicated (by UniProt vt_id).

Usage:
    conda run -n ppi python supplement_esignet_vc1pcava.py [--device cuda:1] [--seed 42]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CV_MOD    = Path("/data/ross/ppi_lossgain/interaction_loss/publication/src/evaluation")
_BENCH_DIR = Path("/home/rcstewart/ppi_lossgain/2026/mutppi/benchmark")
sys.path.insert(0, str(_CV_MOD))
sys.path.insert(0, str(_BENCH_DIR))

from mutpred_ppi_cv import (  # noqa: E402
    _load_varchamp1p_raw, _load_cava_raw,
    get_gene_name, split_wt_id_underscore,
)
import predictors.esignet as _esignet_mod          # noqa: E402
from predictors.esignet import ESigNetPredictor    # noqa: E402

_TRAINING_CSV = Path(
    "/home/rcstewart/ppi_lossgain/2026/mutppi/benchmark/training_data.csv"
)
_ESM_CACHE = "/data/ross/ppi_lossgain/interaction_loss/2026/esm2_residue_embeddings_with_pooled.pkl"
_VC1P_SUPP = Path(
    "/data/ross/ppi_lossgain/interaction_loss/home/eSIG-Net/"
    "esm2_varchamp1p_blind_test_supplement_sahni_fragoza.pkl"
)
_CAVA_SUPP = Path(
    "/data/ross/ppi_lossgain/interaction_loss/home/eSIG-Net/"
    "esm2_cava_blind_test_supplement_sahni_fragoza.pkl"
)
_VC1P_MAP = Path("/data/ross/ppi_lossgain/interaction_loss/home/varchamp1p/gene_symbol_to_uniprot.pkl")
_CAVA_MAP = Path("/data/ross/ppi_lossgain/interaction_loss/home/cava/gene_symbol_to_uniprot.pkl")
_OUT_DIR  = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)
_DESCRIPTION = "eSIG-Net (Sahni+Fragoza train) (varchamp_full_pooled)"


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


def build_test_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build test DataFrame (UniProt + 1-based mutations for eSIG-Net lookup).

    Returns:
        df: rows with interactor/partner/mutation/sequences/perturbed
        uniprot_vt_ids: "UNIPROT1 UNIPROT2 MUT_0based" (canonical format for npy)
        class_labels: C1/C2/C3 integers
    """
    seen: set[str] = set()
    rows, uniprot_vt_ids, class_labels = [], [], []
    n_skip_dup = n_skip_map = 0

    for data, gs2u in sources:
        for i, vt_id in enumerate(data["all_vt_ids"]):
            wt_id = data["all_wt_ids"][i]
            variant_0based = vt_id.split(" ", 1)[1]   # e.g. "V131I" (0-based)

            try:
                a, b = split_wt_id_underscore(wt_id)
                u1 = gs2u[get_gene_name(a)]
                u2 = gs2u[get_gene_name(b)]
            except (KeyError, ValueError):
                n_skip_map += 1
                continue

            canonical_id = f"{u1} {u2} {variant_0based}"
            if canonical_id in seen:
                n_skip_dup += 1
                continue
            seen.add(canonical_id)

            # eSIG-Net expects 1-based mutation positions in the DataFrame
            pos_0 = int(variant_0based[1:-1])
            mut_1based = f"{variant_0based[0]}{pos_0 + 1}{variant_0based[-1]}"

            full_seq = data["wt_seqs"][i]
            L_inter  = data["seq_lengths"][i][0]
            label    = 1 if len(data["pos_labels"][i]) > 0 else 0

            rows.append({
                "interactor":          u1,
                "partner":             u2,
                "mutation":            mut_1based,
                "interactor_sequence": full_seq[:L_inter],
                "partner_sequence":    full_seq[L_inter:],
                "perturbed":           label,
            })
            uniprot_vt_ids.append(canonical_id)   # 0-based for output vt_id
            class_labels.append(classify(u1, u2, sf_proteins))

    print(f"  Unique entries: {len(rows)}  dup_skip={n_skip_dup}  map_miss={n_skip_map}")
    return pd.DataFrame(rows), uniprot_vt_ids, class_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()

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

    print("\nBuilding test DataFrame (UniProt remap + dedup)…")
    test_df, uniprot_vt_ids, class_labels = build_test_df(
        [(vc1p, vc1p_gs2u), (cava, cava_gs2u)],
        sf_proteins,
    )

    print("\nLoading main ESM-2 cache…")
    main_cache = _esignet_mod.load_cache(_ESM_CACHE)
    n_main = len(main_cache) if main_cache else 0
    print(f"  Main cache: {n_main} entries")

    print("Loading vc1p+cava supplement caches…")
    vc1p_supp = pickle.load(open(_VC1P_SUPP, "rb"))
    cava_supp = pickle.load(open(_CAVA_SUPP, "rb"))
    print(f"  VC1p supplement: {len(vc1p_supp)} entries")
    print(f"  CAVA supplement: {len(cava_supp)} entries")

    # Merge: supplement fills in vc1pcava entries missing from main cache
    merged_cache: dict = {} if main_cache is None else dict(main_cache)
    merged_cache.update(vc1p_supp)
    merged_cache.update(cava_supp)
    print(f"  Merged cache: {len(merged_cache)} entries")

    # Monkeypatch load_cache to return merged dict for any path
    _esignet_mod.load_cache = lambda path: merged_cache
    _esignet_mod._ESM_CACHE_PATH = _ESM_CACHE
    ESigNetPredictor.cache_paths = [_ESM_CACHE]

    print("\nLoading SF training data…")
    df_train = pd.read_csv(_TRAINING_CSV)
    sf_mask = df_train["dataset"].str.contains("Sahni") | df_train["dataset"].str.contains("Fragoza")
    train_df = df_train[sf_mask].copy().reset_index(drop=True)
    # Drop NaN sequences
    train_df = train_df[
        train_df["interactor_sequence"].notna() & train_df["partner_sequence"].notna()
    ].reset_index(drop=True)
    print(f"  SF train: {len(train_df)} entries")

    print(f"\nTraining eSIG-Net (seed={args.seed})…")
    predictor = ESigNetPredictor(seed=args.seed, device=args.device)
    predictor.fit(train_df)

    print(f"\nPredicting on {len(test_df)} vc1pcava entries…")
    scores = predictor.predict(test_df).astype(np.float32)

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{len(scores)} NaN scores (missing cache entries)")

    # Drop NaN scores
    valid_mask = ~np.isnan(scores)
    scores_v       = scores[valid_mask]
    labels_v       = np.array([r["perturbed"] for r in test_df.to_dict("records")], dtype=np.int32)[valid_mask]
    vt_ids_v       = np.array(uniprot_vt_ids)[valid_mask]
    classes_v      = np.array(class_labels, dtype=np.int32)[valid_mask]

    print(f"\nTotal valid: {len(scores_v)} / {len(scores)}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{_DESCRIPTION}_vc1pcava"

    for c in [1, 2, 3]:
        mask = classes_v == c
        n_pos = int((labels_v[mask] == 1).sum())
        n_neg = int((labels_v[mask] == 0).sum())
        if mask.sum() == 0:
            auc_str = "n/a"
        elif n_pos < 2 or n_neg < 2:
            auc_str = "too few"
        else:
            auc_str = f"{roc_auc_score(labels_v[mask], scores_v[mask]):.4f}"
        print(f"  C{c}: n={mask.sum()} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(_OUT_DIR / f"{suffix}_c{c}_preds.npy",  scores_v[mask])
        np.save(_OUT_DIR / f"{suffix}_c{c}_labels.npy", labels_v[mask])
        np.save(_OUT_DIR / f"{suffix}_c{c}_vt_ids.npy", vt_ids_v[mask])

    print(f"Saved → {_OUT_DIR}/{suffix}_c{{1,2,3}}_*.npy")
    print(f"\nDone in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
