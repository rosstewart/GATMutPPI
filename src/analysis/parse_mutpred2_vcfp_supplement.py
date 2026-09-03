#!/usr/bin/env python3
"""Parse MutPred2 output CSV for the vc1pcava supplement and save per-class npy arrays.

The supplement covers the 2,936 new VCFP entries (vc1pcava) that were absent from the
original MutPred2 blind test. MutPred2 was run on:
  data/mutpred2_vc1pcava_supplement_input.fasta

The input FASTA uses 1-based positions. The blind test vt_ids use 0-based positions.
This script converts back to match vt_ids when looking up scores.

After running this script, run src/analysis/merge_vc1pcava_into_main.py to merge the
supplement into the main MutPred2 VCFP blind test arrays.

Usage:
    conda run -n ppi python src/analysis/parse_mutpred2_vcfp_supplement.py \\
        --csv mutpred2_vc1pcava_output.csv
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

_EVAL = Path("/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval")
_FULL_METHOD = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"
_MP2_METHOD  = "MutPred2 (varchamp_full_pooled)"
_SUPP_TAG    = "vc1pcava"


def load_mutpred2_csv(csv_path: Path) -> dict[tuple[str, str], float]:
    """Return {(uniprot_id, mutation_1based): score} from MutPred2 output CSV."""
    scores: dict[tuple[str, str], float] = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                prot_id = parts[0].strip()
                mutation = parts[1].strip()
                score = float(parts[2].strip())
                scores[(prot_id, mutation)] = score
            except ValueError:
                continue
    return scores


def mut_0based_to_1based(mut0: str) -> str:
    """Convert 0-based mutation string to 1-based (as used by MutPred2)."""
    wt_aa = mut0[0]
    pos0 = int(mut0[1:-1])
    mut_aa = mut0[-1]
    return f"{wt_aa}{pos0 + 1}{mut_aa}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MutPred2 output CSV file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading MutPred2 output: {csv_path}")
    mp2_scores = load_mutpred2_csv(csv_path)
    print(f"  {len(mp2_scores)} (protein, mutation) scores loaded")

    # Load full-coverage MutPred-PPI arrays to get the canonical set of new vt_ids
    full_vt_ids: set[str] = set()
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        full_vt_ids.update(np.load(vf, allow_pickle=True).tolist())

    # Load existing MutPred2 vt_ids (14,116 entries)
    mp2_vt_ids: set[str] = set()
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_MP2_METHOD}_c{c}_vt_ids.npy"
        mp2_vt_ids.update(np.load(vf, allow_pickle=True).tolist())

    new_vt_ids = sorted(full_vt_ids - mp2_vt_ids)
    print(f"\nNew vc1pcava entries: {len(new_vt_ids)}")

    # Load existing MutPred2 labels for each class to get the canonical C1/C2/C3 classification
    canonical_class: dict[str, int] = {}
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_MP2_METHOD}_c{c}_vt_ids.npy"
        lf = _EVAL / f"{_MP2_METHOD}_c{c}_labels.npy"
        vids = np.load(vf, allow_pickle=True)
        labs = np.load(lf)
        for vid, lab in zip(vids, labs):
            canonical_class[vid] = c

    # Get canonical class for new entries from restratified MutPred-PPI arrays
    mp2_full_class: dict[str, int] = {}
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        for vid in np.load(vf, allow_pickle=True):
            if vid not in mp2_vt_ids:
                mp2_full_class[vid] = c

    # Build per-class predictions
    per_class: dict[int, list] = {c: [] for c in [1, 2, 3]}

    # Also load labels from MutPred-PPI arrays (they share the same ground truth)
    mp2_full_labels: dict[str, int] = {}
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        lf = _EVAL / f"{_FULL_METHOD}_c{c}_labels.npy"
        vids = np.load(vf, allow_pickle=True)
        labs = np.load(lf)
        for vid, lab in zip(vids, labs):
            mp2_full_labels[vid] = int(lab)

    n_found = n_missing = 0
    for vt in new_vt_ids:
        parts = vt.split()
        uid, mut0 = parts[0], parts[2]
        mut1 = mut_0based_to_1based(mut0)
        score = mp2_scores.get((uid, mut1))
        c = mp2_full_class.get(vt, 3)  # default C3 if not classified
        label = mp2_full_labels.get(vt, -1)

        if score is not None:
            per_class[c].append((score, label, vt))
            n_found += 1
        else:
            n_missing += 1

    print(f"\nScored: {n_found}, Missing: {n_missing}")
    for c in [1, 2, 3]:
        entries = per_class[c]
        print(f"  C{c}: {len(entries)} entries")

    if args.dry_run:
        print("\n[dry-run] Not writing files")
        return

    for c in [1, 2, 3]:
        entries = per_class[c]
        if not entries:
            print(f"  C{c}: no entries — skipping")
            continue
        preds  = np.array([e[0] for e in entries], dtype=np.float32)
        labels = np.array([e[1] for e in entries], dtype=np.int8)
        vids   = np.array([e[2] for e in entries])
        pf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_preds.npy"
        lf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_labels.npy"
        vf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_vt_ids.npy"
        np.save(pf, preds)
        np.save(lf, labels)
        np.save(vf, vids)
        print(f"  C{c}: saved {len(preds)} entries → {pf.name}")

    print("\nDone. Next: run src/analysis/merge_vc1pcava_into_main.py")


if __name__ == "__main__":
    main()
