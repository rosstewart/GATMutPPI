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

`run()` is importable (e.g. from src/evaluation/run_vcfp_blind_test.py) and
returns the method description string used for the saved npy files.

Usage:
    conda run -n ppi python supplement_swing_vc1pcava.py [--test-pretrain] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
from xgboost import XGBClassifier

_CV_MOD = Path(__file__).resolve().parent  # src/evaluation
sys.path.insert(0, str(_CV_MOD))

from mutpred_ppi_cv import split_wt_id_underscore  # noqa: E402
from vcfp_common import (  # noqa: E402
    build_sf_proteins, load_vc1pcava_sources, iter_vc1pcava_entries,
    save_vc1pcava_supplement,
)
from swing_common import (  # noqa: E402
    _get_window_encodings, _get_kmers, _get_corpus, _build_swing_df, load_benchmark,
    _D2V_DIM, _D2V_DM, _D2V_ALPHA, _D2V_WINDOW, _D2V_EPOCHS,
    _XGB_N_EST, _XGB_DEPTH, _XGB_LR,
)

_DESCRIPTION    = "SWING (Sahni+Fragoza train) (varchamp_full_pooled)"
_DESCRIPTION_TP = "SWING (test pretrain, Sahni+Fragoza train) (varchamp_full_pooled)"


def build_vc1pcava_raw_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set,
):
    """Build SWING test DataFrame + parallel UniProt vt_id + class lists.

    Returns:
        raw_df:  rows ready for _build_swing_df() (interactor/partner/mutation/seqs/label)
        uniprot_vt_ids: parallel list of "UNIPROT1 UNIPROT2 MUT_0based" identifiers
        class_labels:   parallel C1/C2/C3 integers
    """
    rows, uniprot_vt_ids, class_labels = [], [], []

    for data, i, wt_id, variant_0based, u1, u2, uniprot_id, label, c in iter_vc1pcava_entries(
        sources, sf_proteins
    ):
        # Convert 0-based position to 1-based for SWING internal processing
        pos_0 = int(variant_0based[1:-1])
        variant_1based = f"{variant_0based[0]}{pos_0 + 1}{variant_0based[-1]}"

        full_seq = data["wt_seqs"][i]
        L_inter  = data["seq_lengths"][i][0]
        inter_seq = full_seq[:L_inter]
        part_seq  = full_seq[L_inter:]

        # Use gene-name tokens as interactor/partner keys (SWING uses these for
        # Doc2Vec); recover them the same way iter_vc1pcava_entries did internally.
        interactor, partner = split_wt_id_underscore(wt_id)  # e.g. "ACSF3_71337"

        rows.append({
            "interactor":          interactor,
            "partner":             partner,
            "mutation":            variant_1based,
            "interactor_sequence": inter_seq,
            "partner_sequence":    part_seq,
            "perturbed":           label,
        })
        uniprot_vt_ids.append(uniprot_id)
        class_labels.append(c)

    print(f"  Unique entries: {len(rows)}")
    return pd.DataFrame(rows), uniprot_vt_ids, class_labels


def run(test_pretrain: bool = False, seed: int = 42) -> str:
    """Train/predict SWING on VC1p+CAVA entries and save the vc1pcava supplement.

    Returns the method description string (also the key expected by
    merge_vc1pcava_into_main.merge_method() / restratify_vcfp_blind_test.restratify_one_method()),
    or None if there were no valid entries after SWING position validation.
    """
    description = _DESCRIPTION_TP if test_pretrain else _DESCRIPTION

    t0 = time.time()

    print("Building SF protein set…")
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins")

    sources = load_vc1pcava_sources()

    print("\nBuilding VC1p+CAVA test DataFrame (with UniProt remap + dedup)…")
    raw_test, uniprot_vt_ids, class_labels = build_vc1pcava_raw_df(sources, sf_proteins)

    print(f"Before SWING validation: {len(raw_test)}")
    test_df = _build_swing_df(raw_test)
    print(f"After SWING validation:  {len(test_df)}")

    if len(test_df) == 0:
        print("No valid entries after SWING position validation. Exiting.")
        return None

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
        frac=1, random_state=seed
    ).reset_index(drop=True)
    train_aug.index = range(len(train_aug))

    print("Computing window encodings for training data…")
    train_encodings = _get_window_encodings(train_aug)
    train_kmers     = _get_kmers(train_encodings)

    print("Computing window encodings for test data…")
    test_encodings = _get_window_encodings(test_df)
    test_kmers     = _get_kmers(test_encodings)

    n_train = len(train_aug)

    if test_pretrain:
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
        use_label_encoder=False, eval_metric="logloss", random_state=seed,
    )
    xgb.fit(X_train, y_train)
    scores = xgb.predict_proba(X_test)[:, 1].astype(np.float32)
    labels_arr = test_df["Y2H_score"].values
    classes_arr = np.array(class_labels, dtype=np.int32)
    vt_ids_arr  = np.array(uniprot_vt_ids)

    print(f"\nTotal: {len(scores)} entries")
    result = save_vc1pcava_supplement(scores, labels_arr, vt_ids_arr, classes_arr, description)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pretrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(test_pretrain=args.test_pretrain, seed=args.seed)


if __name__ == "__main__":
    main()
