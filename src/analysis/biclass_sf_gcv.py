#!/usr/bin/env python3
"""Biclass SF GCV: filter Sahni+Fragoza GCV to ordered protein pairs where mutations
in the interactor span BOTH label=1 (disrupting) AND label=0 (non-disrupting).

A biclass pair (A, B) is ordered: mutations in protein A must have both classes; the
reverse pair (B, A) is evaluated independently. This captures pairs where the interaction
can be either maintained or disrupted, depending on which specific residue is mutated.

Output:
    results_revisions/biclass_gcv/roc_sahni_fragoza_biclass_with_variance.png

Usage:
    conda run -n ppi python src/analysis/biclass_sf_gcv.py
"""
from __future__ import annotations

import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PUB = Path("/data/ross/ppi_lossgain/interaction_loss/publication")
_CV  = Path("/home/rcstewart/gnn/ppi_interaction_loss/cv_splits")

sys.path.insert(0, str(_PUB / "src" / "analysis"))
from roc_plots import (
    compute_roc_with_variance,
    plot_roc_with_confidence,
    METHOD_DISPLAY_NAMES,
    colors as METHOD_COLORS,
)

LABEL_FILE   = _CV / "sahni_fragoza_all_vt_ids_and_labels.txt"
IPTM_PKL     = _PUB / "results_revisions" / "macro_aucs" / "iptm_sahni_fragoza_gcv_splits.pkl"
PKL_DIR      = _PUB / "results_revisions" / "macro_aucs"
OUT_DIR      = _PUB / "results_revisions" / "biclass_gcv"

METHODS = [
    ("MutPredPPI_sahni_fragoza_megascale_all",   "MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"),
    ("ESigNet_sahni_fragoza",                    "ESigNet_sahni_fragoza_detailed_results.pkl"),
    ("SWING_sahni_fragoza_test_pretrain",        "SWING_sahni_fragoza_test_pretrain_detailed_results.pkl"),
    ("SWING_sahni_fragoza_no_test_pretrain",     "SWING_sahni_fragoza_no_test_pretrain_detailed_results.pkl"),
    ("MINT_seq_diff_sahni_fragoza",              "MINT_seq_diff_sahni_fragoza_detailed_results.pkl"),
    ("MINT_site_diff_sahni_fragoza",             "MINT_site_diff_sahni_fragoza_detailed_results.pkl"),
    ("PPLM_seq_diff_sahni_fragoza",              "PPLM_seq_diff_sahni_fragoza_detailed_results.pkl"),
    ("PPLM_site_diff_sahni_fragoza",             "PPLM_site_diff_sahni_fragoza_detailed_results.pkl"),
]


def load_biclass_pairs(label_file: Path) -> set[str]:
    """Return set of complex_ids where interactor mutations span both labels 0 and 1."""
    pair_labels: dict[str, set[int]] = defaultdict(set)
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pair_labels[parts[0]].add(int(parts[2]))
    biclass = {cid for cid, lbls in pair_labels.items() if lbls == {0, 1}}
    n_total = len(pair_labels)
    print(f"  Total ordered pairs: {n_total:,}")
    print(f"  Biclass pairs:       {len(biclass):,} ({len(biclass)/n_total:.1%})")
    return biclass


def filter_detailed_results(
    detailed_results: dict,
    iptm_data: dict,
    biclass_pairs: set[str],
) -> dict:
    """Build a new detailed_results dict restricted to biclass-pair entries."""
    filtered: dict = {"iterations": {}}
    n_kept = n_total = 0

    for it, iter_data in detailed_results["iterations"].items():
        filtered["iterations"][it] = {"folds": {}}
        iptm_iter = iptm_data["iterations"][it]["folds"]

        for fd, fold_data in iter_data["folds"].items():
            filtered["iterations"][it]["folds"][fd] = {}
            iptm_fold = iptm_iter[fd]

            for cl in ["class_1", "class_2", "class_3"]:
                orig     = fold_data[cl]
                cids_raw = iptm_fold[cl]["complex_ids"]
                cids     = [c.replace("|", "-") for c in cids_raw]

                n_iptm = len(cids)
                n_pred = len(orig["preds"])

                if n_iptm != n_pred:
                    # Length mismatch: iptm pkl does not align with this method's fold
                    # — skip this fold (empty entry, no contribution to AUC)
                    filtered["iterations"][it]["folds"][fd][cl] = {
                        "preds": np.array([]), "labels": np.array([]), "auc": 0.0
                    }
                    continue

                mask   = np.array([c in biclass_pairs for c in cids])
                preds  = np.array(orig["preds"])[mask]
                labels = np.array(orig["labels"])[mask]

                n_kept  += int(mask.sum())
                n_total += len(mask)

                filtered["iterations"][it]["folds"][fd][cl] = {
                    "preds": preds, "labels": labels, "auc": 0.0
                }

    frac = n_kept / n_total if n_total else 0
    print(f"  Entries kept: {n_kept:,}/{n_total:,} ({frac:.1%})", flush=True)
    return filtered


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading biclass pairs from label file...", flush=True)
    biclass_pairs = load_biclass_pairs(LABEL_FILE)

    print("Loading iptm pkl (complex_id ordering)...", flush=True)
    with open(IPTM_PKL, "rb") as f:
        iptm_data = pickle.load(f)

    results_dict: dict = {}

    for method_key, pkl_name in METHODS:
        pkl_path = PKL_DIR / pkl_name
        if not pkl_path.exists():
            print(f"  [SKIP] {pkl_name} not found", flush=True)
            continue
        print(f"\nProcessing {method_key}...", flush=True)
        with open(pkl_path, "rb") as f:
            detailed_results = pickle.load(f)

        filtered = filter_detailed_results(detailed_results, iptm_data, biclass_pairs)
        roc      = compute_roc_with_variance(filtered)

        n_curves_c3 = len(roc["class_3"]["aucs"])
        if n_curves_c3 == 0:
            print(f"  [SKIP] No valid C3 curves after biclass filtering", flush=True)
            continue

        mean_c3 = float(np.mean(roc["class_3"]["aucs"]))
        std_c3  = float(np.std(roc["class_3"]["aucs"], ddof=1))
        print(f"  C3 AUROC: {mean_c3:.3f} ± {std_c3:.3f}  (n_curves={n_curves_c3})",
              flush=True)
        results_dict[method_key] = roc

    if not results_dict:
        print("ERROR: No methods produced valid results.", flush=True)
        return

    out_png = OUT_DIR / "roc_sahni_fragoza_biclass_with_variance.png"
    print(f"\nPlotting to {out_png}...", flush=True)
    plot_roc_with_confidence(
        results_dict,
        dataset_name="sahni_fragoza (biclass pairs)",
        save_path=str(out_png),
    )
    print(f"Saved → {out_png}", flush=True)

    # TSV summary of C3 AUROCs
    tsv_path = OUT_DIR / "biclass_c3_aurocs.tsv"
    with open(tsv_path, "w") as f:
        f.write("method\tdisplay_name\tmean_auc_c3\tstd_auc_c3\tn_curves\n")
        for mkey, roc in results_dict.items():
            aucs    = np.array(roc["class_3"]["aucs"])
            dname   = METHOD_DISPLAY_NAMES.get(mkey, mkey)
            f.write(f"{mkey}\t{dname}\t{np.mean(aucs):.4f}\t{np.std(aucs, ddof=1):.4f}"
                    f"\t{len(aucs)}\n")
    print(f"AUCs saved → {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
