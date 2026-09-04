#!/usr/bin/env python
"""Regenerate grouped-CV fold-assignment splits from scratch.

Ports the exact split-generation recipe from the original (pre-repo,
uncontrolled) notebook `gnn_sahni_fragoza_gcv_iterations.py` so that splits
can be reproduced byte-identically instead of being read from a pre-computed
external cache (`_CV_DIR`). Sequence clustering (CD-HIT) is unchanged from
`src/evaluation/mutpred_ppi_cv.py::cluster_sequences()` — that logic already
lives in this repo and was not duplicated here.

Recipe (exact, do not change without re-verifying against existing cv_splits/):
  1. Load the dataset via its existing `mutpred_ppi_cv.py` loader (this fixes
     row order to whatever `_load_graphs()`'s glob.glob() returns on this
     machine — see note below).
  2. `indices = list(range(n)); random.seed(42); random.shuffle(indices)` —
     apply this one fixed permutation to vt_ids and cluster labels.
  3. For gcv_seed in range(n_iterations):
       GroupKFold(n_splits=10, shuffle=True, random_state=gcv_seed)
         .split(range(n), groups=clusters)
     Save (fold, train_idx, test_idx) tuples — one iteration = one gcv_seed.

Reproducibility caveat: `_load_graphs()` iterates `glob.glob(f"{graph_dir}/*.mat")`,
whose OS-level ordering is not a formally guaranteed constant across machines/
filesystems. On this machine (same directory, unmodified since original
generation) it reproduces the existing splits byte-identically (verified below);
a fresh clone on a different machine should get identical *fold assignments*
(GroupKFold membership depends only on relative cluster/group structure, not
absolute row order) but the raw index arrays may differ in row numbering if
the underlying directory listing order differs.

Usage:
    conda run -n ppi python src/training/generate_cv_splits.py --dataset sahni_fragoza --out-dir cv_splits_regenerated
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import mutpred_ppi_cv as cv  # noqa: E402

N_SPLITS = 10
N_ITERATIONS = 30
SHUFFLE_SEED = 42

LOADERS = {
    "sahni_fragoza": cv.load_sahni_fragoza,
}


def generate_splits(dataset: str, out_dir: str) -> None:
    if dataset not in LOADERS:
        raise ValueError(f"No loader registered for {dataset!r}; add one to LOADERS.")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {dataset}...", flush=True)
    data = LOADERS[dataset]()
    all_vt_ids = data["all_vt_ids"]
    clusters = data["clusters"]
    n = len(all_vt_ids)
    print(f"  {n} vt_ids, {len(set(clusters))} unique clusters", flush=True)

    indices = list(range(n))
    random.seed(SHUFFLE_SEED)
    random.shuffle(indices)
    all_vt_ids = [all_vt_ids[i] for i in indices]
    clusters = [clusters[i] for i in indices]

    for gcv_seed in range(N_ITERATIONS):
        kf = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=gcv_seed)
        fold_splits = [
            (fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(kf.split(range(n), groups=clusters))
        ]
        with open(f"{out_dir}/{dataset}_train_all_vt_ids_{gcv_seed}.pkl", "wb") as f:
            pickle.dump(all_vt_ids, f)
        with open(f"{out_dir}/{dataset}_train_fold_splits_{gcv_seed}.pkl", "wb") as f:
            pickle.dump(fold_splits, f)
        print(f"  gcv_seed={gcv_seed}: {len(fold_splits)} folds written", flush=True)

    print(f"Done. Output: {out_dir}/{dataset}_train_{{all_vt_ids,fold_splits}}_<gcv_seed>.pkl", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="sahni_fragoza", choices=list(LOADERS.keys()))
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    generate_splits(args.dataset, args.out_dir)
