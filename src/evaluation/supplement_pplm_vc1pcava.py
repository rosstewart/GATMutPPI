#!/usr/bin/env python
"""PPLM supplemental blind test on VC1p + CAVA entries.

These entries are in the SFVCFP pkl but were excluded from the main VCFP blind
test because their graphs use gene-name/Entrez IDs rather than UniProt IDs.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl (same mapping
used in load_sahni_fragoza_varchamp1p_cava() for GCV construction). C1/C2/C3
classification uses the remapped UniProt IDs against the SF protein set.

The PPLM GCV cache already covers all vc1pcava proteins (confirmed by precompute
showing 0 new embeddings needed), so no supplemental precompute is required.

Output vt_ids use canonical 0-based format: "UNIPROT1 UNIPROT2 MUT_0based"
(matching the MutPred-PPI vc1pcava supplement, so restratify intersection works).

22 entries shared between VC1p and CAVA are deduplicated.

Usage:
    conda run -n ppi python supplement_pplm_vc1pcava.py [--predictor seq_diff|site_diff]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CV_MOD = Path(__file__).resolve().parent  # src/evaluation — also holds predictors/ now
sys.path.insert(0, str(_CV_MOD))

import predictors.pplm_mlp as _pplm_mod  # noqa: E402
from predictors.pplm_mlp import PPLMSeqDiff, PPLMSiteDiff  # noqa: E402
from predictors import nn_base  # noqa: E402

from mutpred_ppi_cv import (  # noqa: E402
    _load_varchamp1p_raw, _load_cava_raw,
    get_gene_name, split_wt_id_underscore,
)

_TRAINING_CSV = Path("/home/rcstewart/mutppi/benchmark/training_data.csv")
_GCV_CACHE    = Path(__file__).resolve().parents[2] / "data_caches" / "pplm_cache.pkl"
_VC1P_MAP     = Path("/data/ross/ppi_lossgain/interaction_loss/home/varchamp1p/gene_symbol_to_uniprot.pkl")
_CAVA_MAP     = Path("/data/ross/ppi_lossgain/interaction_loss/home/cava/gene_symbol_to_uniprot.pkl")
_OUT_DIR      = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)

_DESCRIPTIONS = {
    "seq_diff":  "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "site_diff": "PPLM_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
}
_PREDICTOR_MAP = {"seq_diff": PPLMSeqDiff, "site_diff": PPLMSiteDiff}


def classify(u1: str, u2: str, sf_proteins: set[str]) -> int:
    a_in = u1 in sf_proteins
    b_in = u2 in sf_proteins
    if a_in and b_in:
        return 1
    if a_in or b_in:
        return 2
    return 3


def build_sf_proteins() -> set[str]:
    df = pd.read_csv(_TRAINING_CSV)
    sf = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return set(df.loc[sf, "interactor"]) | set(df.loc[sf, "partner"])


def load_sf_train_df() -> pd.DataFrame:
    df = pd.read_csv(_TRAINING_CSV)
    sf = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return df[sf].copy().reset_index(drop=True)


def install_pplm_cache() -> None:
    print(f"Loading PPLM GCV cache: {_GCV_CACHE}", flush=True)
    with open(_GCV_CACHE, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache)} entries", flush=True)
    cache_key = str(_GCV_CACHE)
    _pplm_mod.CACHE_PATH = cache_key
    PPLMSeqDiff._cache_path  = cache_key
    PPLMSiteDiff._cache_path = cache_key
    nn_base._CACHE_STORE[cache_key] = cache


def build_vc1pcava_test_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build PPLM-format DataFrame and canonical 0-based vt_ids from vc1pcava raw data.

    Returns (df, canonical_vt_ids) where df has 1-based mutations (for PPLM cache
    lookup) and canonical_vt_ids has 0-based mutations (for output npy files).
    """
    seen: set[str] = set()
    rows = []
    canonical_vt_ids = []
    n_dup = n_miss = 0

    for data, gs2u in sources:
        for i, vt_id in enumerate(data["all_vt_ids"]):
            wt_id   = data["all_wt_ids"][i]
            variant = vt_id.split(" ", 1)[1]   # e.g. "D220V" (0-based)
            pos_0   = int(variant[1:-1])

            try:
                a, b = split_wt_id_underscore(wt_id)
                u1 = gs2u[get_gene_name(a)]
                u2 = gs2u[get_gene_name(b)]
            except (KeyError, ValueError):
                n_miss += 1
                continue

            canon_id = f"{u1} {u2} {variant}"   # 0-based, canonical output format
            if canon_id in seen:
                n_dup += 1
                continue
            seen.add(canon_id)

            mut_1based = f"{variant[0]}{pos_0 + 1}{variant[-1]}"  # for PPLM cache lookup
            label = 1 if len(data["pos_labels"][i]) > 0 else 0

            rows.append({
                "interactor": u1,
                "partner":    u2,
                "mutation":   mut_1based,
                "perturbed":  label,
                "class":      classify(u1, u2, sf_proteins),
            })
            canonical_vt_ids.append(canon_id)

    print(f"  Unique entries: {len(rows)}  dup_skip={n_dup}  map_miss={n_miss}", flush=True)
    df = pd.DataFrame(rows)
    return df, canonical_vt_ids


def save_stratified(
    scores: np.ndarray,
    df: pd.DataFrame,
    canonical_vt_ids: list[str],
    description: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels   = df["perturbed"].values.astype(int)
    classes  = df["class"].values
    vt_ids   = np.array(canonical_vt_ids)
    nan_mask = ~np.isnan(scores)

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} NaN scores (cache misses)", flush=True)

    print(f"Writing stratified outputs → {out_dir}", flush=True)
    for c_idx, c_label in enumerate([1, 2, 3]):
        mask  = (classes == c_label) & nan_mask
        p_sub = scores[mask].astype(np.float32)
        l_sub = labels[mask]
        v_sub = vt_ids[mask]
        np.save(out_dir / f"{description}_vc1pcava_c{c_label}_preds.npy",  p_sub)
        np.save(out_dir / f"{description}_vc1pcava_c{c_label}_labels.npy", l_sub)
        np.save(out_dir / f"{description}_vc1pcava_c{c_label}_vt_ids.npy", v_sub)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(l_sub, p_sub) if len(np.unique(l_sub)) > 1 else float("nan")
        print(f"  C{c_label}: n={len(v_sub)} (pos={int((l_sub==1).sum())}, neg={int((l_sub==0).sum())}) AUC={auc:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="seq_diff", choices=("seq_diff", "site_diff"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    description = _DESCRIPTIONS[args.predictor]

    install_pplm_cache()

    print("Building SF protein set…", flush=True)
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins", flush=True)

    print("Loading UniProt maps…", flush=True)
    vc1p_gs2u = pickle.load(open(_VC1P_MAP, "rb"))
    cava_gs2u = pickle.load(open(_CAVA_MAP, "rb"))

    print("Loading VC1p data…", flush=True)
    vc1p = _load_varchamp1p_raw()
    print(f"  {len(vc1p['all_vt_ids'])} entries", flush=True)

    print("Loading CAVA data…", flush=True)
    cava = _load_cava_raw()
    print(f"  {len(cava['all_vt_ids'])} entries", flush=True)

    print("\nBuilding test DataFrame (UniProt remap + dedup)…", flush=True)
    test_df, canonical_vt_ids = build_vc1pcava_test_df(
        [(vc1p, vc1p_gs2u), (cava, cava_gs2u)],
        sf_proteins,
    )

    print("\nLoading SF training data…", flush=True)
    train_df = load_sf_train_df()
    print(f"  SF train: {len(train_df)} entries", flush=True)

    PredClass = _PREDICTOR_MAP[args.predictor]
    print(f"\nTraining {PredClass().name} on {len(train_df)} rows (seed={args.seed})…", flush=True)
    predictor = PredClass(seed=args.seed)
    predictor.fit(train_df)

    print(f"\nPredicting on {len(test_df)} vc1pcava entries…", flush=True)
    scores = predictor.predict(test_df).astype(np.float32)

    save_stratified(scores, test_df, canonical_vt_ids, description, _OUT_DIR)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
