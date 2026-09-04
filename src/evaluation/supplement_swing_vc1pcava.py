#!/usr/bin/env python
"""SWING supplemental blind test on VC1p + CAVA entries.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl (matching the
GCV construction in load_sahni_fragoza_varchamp1p_cava()). C1/C2/C3 are
classified using the remapped UniProt IDs against the SF protein set.

22 entries shared between VC1p and CAVA are deduplicated (by UniProt vt_id).

Position note: VC1p/CAVA labels store positions 0-based. SWING's
_build_swing_df() expects 1-based positions — we shift +1 for SWING's
internal processing, but the OUTPUT vt_id uses the original 0-based position
to match MutPred-PPI and all other methods.

Usage:
    conda run -n ppi python supplement_swing_vc1pcava.py [--test-pretrain] [--seed 42]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from xgboost import XGBClassifier

_CV_MOD    = Path("/data/ross/ppi_lossgain/interaction_loss/publication/src/evaluation")
_SWING_DIR = Path("/home/rcstewart/ppi_lossgain/SWING_scripts/blind_test")
sys.path.insert(0, str(_CV_MOD))
sys.path.insert(0, str(_SWING_DIR))

from mutpred_ppi_cv import (  # noqa: E402
    _load_varchamp1p_raw, _load_cava_raw,
    get_gene_name, split_wt_id_underscore,
)
from run_swing_blind_test_vcfp import (  # noqa: E402
    _get_window_encodings, _get_kmers, _get_corpus, _build_swing_df, load_benchmark,
    _D2V_DIM, _D2V_DM, _D2V_ALPHA, _D2V_WINDOW, _D2V_EPOCHS,
    _XGB_N_EST, _XGB_DEPTH, _XGB_LR,
)

_TRAINING_CSV = Path(
    "/home/rcstewart/mutppi/benchmark/training_data.csv"
)
_VC1P_MAP  = Path("/data/ross/ppi_lossgain/interaction_loss/home/varchamp1p/gene_symbol_to_uniprot.pkl")
_CAVA_MAP  = Path("/data/ross/ppi_lossgain/interaction_loss/home/cava/gene_symbol_to_uniprot.pkl")
_OUT_DIR = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)
_DESCRIPTION    = "SWING (Sahni+Fragoza train) (varchamp_full_pooled)"
_DESCRIPTION_TP = "SWING (test pretrain, Sahni+Fragoza train) (varchamp_full_pooled)"


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


def build_vc1pcava_raw_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build SWING test DataFrame + parallel UniProt vt_id + class lists.

    Returns:
        raw_df:  rows ready for _build_swing_df() (interactor/partner/mutation/seqs/label)
        uniprot_vt_ids: parallel list of "UNIPROT1 UNIPROT2 MUT_0based" identifiers
        class_labels:   parallel C1/C2/C3 integers
    """
    seen: set[str] = set()
    rows, uniprot_vt_ids, class_labels = [], [], []
    n_skip_dup = n_skip_map = 0

    for data, gs2u in sources:
        for i, vt_id in enumerate(data["all_vt_ids"]):
            wt_id = data["all_wt_ids"][i]
            variant_0based = vt_id.split(" ", 1)[1]  # e.g. "V131I" (0-based)

            # Remap to UniProt
            try:
                a, b = split_wt_id_underscore(wt_id)
                u1 = gs2u[get_gene_name(a)]
                u2 = gs2u[get_gene_name(b)]
            except (KeyError, ValueError):
                n_skip_map += 1
                continue

            uniprot_id = f"{u1} {u2} {variant_0based}"
            if uniprot_id in seen:
                n_skip_dup += 1
                continue
            seen.add(uniprot_id)

            # Convert 0-based position to 1-based for SWING internal processing
            pos_0 = int(variant_0based[1:-1])
            variant_1based = f"{variant_0based[0]}{pos_0 + 1}{variant_0based[-1]}"

            full_seq = data["wt_seqs"][i]
            L_inter  = data["seq_lengths"][i][0]
            inter_seq = full_seq[:L_inter]
            part_seq  = full_seq[L_inter:]

            # Use gene-name tokens as interactor/partner keys (SWING uses these for Doc2Vec)
            interactor = a  # e.g. "ACSF3_71337"
            partner    = b  # e.g. "PPP1R13B_70264"

            label = 1 if len(data["pos_labels"][i]) > 0 else 0

            rows.append({
                "interactor":          interactor,
                "partner":             partner,
                "mutation":            variant_1based,
                "interactor_sequence": inter_seq,
                "partner_sequence":    part_seq,
                "perturbed":           label,
            })
            uniprot_vt_ids.append(uniprot_id)
            class_labels.append(classify(u1, u2, sf_proteins))

    print(f"  Unique entries: {len(rows)}  dup_skip={n_skip_dup}  map_miss={n_skip_map}")
    return pd.DataFrame(rows), uniprot_vt_ids, class_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pretrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    description = _DESCRIPTION_TP if args.test_pretrain else _DESCRIPTION

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

    print("\nBuilding VC1p+CAVA test DataFrame (with UniProt remap + dedup)…")
    raw_test, uniprot_vt_ids, class_labels = build_vc1pcava_raw_df(
        [(vc1p, vc1p_gs2u), (cava, cava_gs2u)],
        sf_proteins,
    )

    print(f"Before SWING validation: {len(raw_test)}")
    test_df = _build_swing_df(raw_test)
    print(f"After SWING validation:  {len(test_df)}")

    if len(test_df) == 0:
        print("No valid entries after SWING position validation. Exiting.")
        return

    # Align uniprot_vt_ids and class_labels to the validated test_df rows
    # _build_swing_df may drop rows; use index alignment
    valid_indices = list(test_df.index)
    uniprot_vt_ids = [uniprot_vt_ids[i] for i in valid_indices]
    class_labels   = [class_labels[i]   for i in valid_indices]
    test_df = test_df.reset_index(drop=True)

    print(f"\nLoading SF training data…")
    train_df, _, _ = load_benchmark()
    print(f"SF train: {len(train_df)} entries")

    print("\nAdding WT sequences to training set (WT augmentation)…")
    train_wts = train_df.copy()
    train_wts["Mutated_Seq"] = train_df["Target_Seq"]
    train_wts["Type"] = "WildType"
    train_wts["Y2H_score"] = 0
    train_aug = pd.concat([train_df, train_wts], ignore_index=True).sample(
        frac=1, random_state=args.seed
    ).reset_index(drop=True)
    train_aug.index = range(len(train_aug))

    print("Computing window encodings for training data…")
    train_encodings = _get_window_encodings(train_aug)
    train_kmers     = _get_kmers(train_encodings)

    print("Computing window encodings for test data…")
    test_encodings = _get_window_encodings(test_df)
    test_kmers     = _get_kmers(test_encodings)

    n_train = len(train_aug)

    if args.test_pretrain:
        combined_corpus = list(_get_corpus(train_kmers + test_kmers))
        print(f"\nTraining Doc2Vec on combined corpus ({n_train}+{len(test_df)})…")
        d2v = Doc2Vec(
            vector_size=_D2V_DIM, min_count=1, alpha=_D2V_ALPHA,
            dm=_D2V_DM, window=_D2V_WINDOW,
        )
        d2v.build_vocab(combined_corpus)
        d2v.train(combined_corpus, total_examples=d2v.corpus_count, epochs=_D2V_EPOCHS)
        train_aug_mut = train_aug[train_aug["Type"] == "Mutant"].copy()
        train_aug_wt  = train_aug[train_aug["Type"] == "WildType"].copy()
        X_train_mut   = np.array([d2v.dv.vectors[i] for i in train_aug_mut.index])
        X_train_wt    = np.array([d2v.dv.vectors[i] for i in train_aug_wt.index])
        X_test        = np.array([d2v.dv.vectors[n_train + i] for i in range(len(test_df))])
    else:
        train_corpus = list(_get_corpus(train_kmers))
        print(f"\nTraining Doc2Vec (dim={_D2V_DIM}, epochs={_D2V_EPOCHS})…")
        d2v = Doc2Vec(
            vector_size=_D2V_DIM, min_count=1, alpha=_D2V_ALPHA,
            dm=_D2V_DM, window=_D2V_WINDOW,
        )
        d2v.build_vocab(train_corpus)
        d2v.train(train_corpus, total_examples=d2v.corpus_count, epochs=_D2V_EPOCHS)
        train_aug_mut = train_aug[train_aug["Type"] == "Mutant"].copy()
        train_aug_wt  = train_aug[train_aug["Type"] == "WildType"].copy()
        X_train_mut   = np.array([d2v.dv.vectors[i] for i in train_aug_mut.index])
        X_train_wt    = np.array([d2v.dv.vectors[i] for i in train_aug_wt.index])
        test_corpus   = list(_get_corpus(test_kmers))
        print("Inferring Doc2Vec vectors for test data…")
        X_test = np.array([d2v.infer_vector(doc.words) for doc in tqdm(test_corpus)])

    X_train = np.concatenate([X_train_mut, X_train_wt])
    y_train = np.concatenate([
        train_aug_mut["Y2H_score"].values, np.zeros(len(X_train_wt), dtype=int)
    ])

    print(f"\nTraining XGBoost on {len(X_train)} rows…")
    xgb = XGBClassifier(
        n_estimators=_XGB_N_EST, max_depth=_XGB_DEPTH, learning_rate=_XGB_LR,
        use_label_encoder=False, eval_metric="logloss", random_state=args.seed,
    )
    xgb.fit(X_train, y_train)
    scores = xgb.predict_proba(X_test)[:, 1].astype(np.float32)
    labels_arr = test_df["Y2H_score"].values
    classes_arr = np.array(class_labels, dtype=np.int32)
    vt_ids_arr  = np.array(uniprot_vt_ids)

    print(f"\nTotal: {len(scores)} entries")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{description}_vc1pcava"

    for c in [1, 2, 3]:
        mask = classes_arr == c
        n_pos = int((labels_arr[mask] == 1).sum())
        n_neg = int((labels_arr[mask] == 0).sum())
        if mask.sum() == 0:
            auc_str = "n/a"
        elif n_pos < 2 or n_neg < 2:
            auc_str = "too few"
        else:
            auc_str = f"{roc_auc_score(labels_arr[mask], scores[mask]):.4f}"
        print(f"  C{c}: n={mask.sum()} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(_OUT_DIR / f"{suffix}_c{c}_preds.npy",  scores[mask])
        np.save(_OUT_DIR / f"{suffix}_c{c}_labels.npy", labels_arr[mask].astype(np.int32))
        np.save(_OUT_DIR / f"{suffix}_c{c}_vt_ids.npy", vt_ids_arr[mask])

    print(f"Saved → {_OUT_DIR}/{suffix}_c{{1,2,3}}_*.npy")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
