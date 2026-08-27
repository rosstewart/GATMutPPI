#!/usr/bin/env python
"""SAAMBE-3D GCV inference script for sahni_fragoza_varchamp2026.

Mirrors the flat-array format produced by saambe3d_cv.py for the
sahni_fragoza_varchamp1p_cava dataset, adapted for the seeded GCV splits
used by the varchamp2026 combined dataset.

Since SAAMBE-3D is deterministic (no random training component), a single
GCV seed is sufficient. Predictions are saved in fold-traversal order,
aligned to the pair_test_classes array for that seed.

Outputs (in --outdir):
    sahni_fragoza_varchamp2026_SAAMBE-3D_preds.npy       float32, shape (N,)
    sahni_fragoza_varchamp2026_SAAMBE-3D_test_classes.npy int8,    shape (N,)

where N = total variants across all folds (should equal len(all_vt_ids_and_labels)).

Usage:
    conda run -n pytorch_env python saambe3d_varchamp2026_gcv.py \\
        --model-type regression \\
        --gcv-seed 0 \\
        --outdir /data/ross/ppi_lossgain/interaction_loss/publication/results_revisions/macro_aucs/
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── fixed paths ───────────────────────────────────────────────────────────────
_CV_DIR       = Path("/home/rcstewart/gnn/ppi_interaction_loss/cv_splits")
_PDB_DIR      = Path("/data/ross/ppi_lossgain/interaction_loss/2026/af3_out/pdbs")
_SAAMBE3D_PY  = Path(__file__).parent / "saambe-3d.py"

_LABELS_FILE          = "combined_sahni_fragoza_varchamp2026_all_vt_ids_and_labels.txt"
_FOLD_SPLITS_PAT      = "sahni_fragoza_varchamp2026_train_fold_splits_{seed}.pkl"
_PAIR_TEST_CLASSES_PAT = "combined_sahni_fragoza_varchamp2026_pair_test_classes_{seed}.npy"

_DATASET_NAME = "sahni_fragoza_varchamp2026"


# ── PDB path resolution ───────────────────────────────────────────────────────

def _resolve_pdb(interactor: str, partner: str) -> tuple[Path | None, bool]:
    """Return (pdb_path, is_reversed).

    is_reversed=True means the AF3 model was generated with partner as chain A
    and interactor as chain B (filename order is reversed).

    PDB naming: {a}-{b}_model.pdb where a,b are lowercase UniProt IDs.
    Isoform suffixes (e.g. 'P40692-1') are kept as-is in the filename.
    """
    a = interactor.lower()
    b = partner.lower()
    fwd = _PDB_DIR / f"{a}-{b}_model.pdb"
    rev = _PDB_DIR / f"{b}-{a}_model.pdb"
    if fwd.exists():
        return fwd, False
    if rev.exists():
        return rev, True

    # Fallback: try stripping isoform suffix from interactor or partner
    a_base = a.split("-")[0]
    b_base = b.split("-")[0]
    if a_base != a or b_base != b:
        fwd2 = _PDB_DIR / f"{a_base}-{b_base}_model.pdb"
        rev2 = _PDB_DIR / f"{b_base}-{a_base}_model.pdb"
        if fwd2.exists():
            return fwd2, False
        if rev2.exists():
            return rev2, True

    return None, False


def _split_pair(complex_id: str) -> tuple[str, str]:
    """Split 'INTERACTOR_PARTNER' into (interactor, partner).

    UniProt accession IDs do not contain underscores, so we split on the
    first '_'.  Isoform suffixes like 'P40692-1' are handled by
    _resolve_pdb().
    """
    return complex_id.split("_", 1)


# ── SAAMBE-3D subprocess ──────────────────────────────────────────────────────

def _call_saambe(pdb_path: Path, chain: str, pos_1based: str,
                 wt_res: str, mt_res: str, model_flag: str,
                 tmp_out: Path) -> tuple[float, int]:
    """Run SAAMBE-3D and return (score, binary_label).

    Regression (-d 1): output "{ddg} Destabilizing|Stabilizing"
        score = ddG float; binary = 1 if Destabilizing else 0
    Classification (-d 0): output "Disruptive|Nondisruptive"
        score = float(binary); binary = 1 if Disruptive else 0
    """
    subprocess.run(
        [sys.executable, str(_SAAMBE3D_PY),
         "-i", str(pdb_path), "-c", chain,
         "-r", pos_1based, "-w", wt_res, "-m", mt_res,
         "-d", model_flag, "-o", str(tmp_out)],
        check=True, capture_output=True,
        cwd=str(_SAAMBE3D_PY.parent),
    )
    with open(tmp_out) as fh:
        line = fh.readline().strip()
    tmp_out.unlink(missing_ok=True)
    tokens = line.split()
    if model_flag == "1":
        score = float(tokens[0])
        binary = 1 if len(tokens) > 1 and tokens[1] == "Destabilizing" else 0
    else:
        binary = 1 if tokens and tokens[0] == "Disruptive" else 0
        score = float(binary)
    return score, binary


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_flag = "1" if args.model_type == "regression" else "0"
    stem = "SAAMBE-3D" if args.model_type == "regression" else "SAAMBE-3D_dn"
    out_preds   = outdir / f"{_DATASET_NAME}_{stem}_preds.npy"
    out_classes = outdir / f"{_DATASET_NAME}_{stem}_test_classes.npy"

    if out_preds.exists() and not args.overwrite:
        print(f"Output already exists: {out_preds}  (use --overwrite to rerun)",
              flush=True)
        return

    # ── load canonical vt_ids (0-based positions in file) ────────────────────
    vt_ids: list[tuple[str, str]] = []    # (complex_id, variant)
    with open(_CV_DIR / _LABELS_FILE) as fh:
        for line in fh:
            parts = line.strip().split()
            vt_ids.append((parts[0], parts[1]))
    print(f"Loaded {len(vt_ids)} canonical vt_ids", flush=True)

    # ── load GCV fold splits for chosen seed ─────────────────────────────────
    splits_path = _CV_DIR / _FOLD_SPLITS_PAT.format(seed=args.gcv_seed)
    with open(splits_path, "rb") as fh:
        fold_splits = pickle.load(fh)
    n_folds = len(fold_splits)
    total_test = sum(len(test_idx) for _, _, test_idx in fold_splits)
    print(f"GCV seed {args.gcv_seed}: {n_folds} folds, {total_test} total test variants",
          flush=True)

    # ── load pair_test_classes for this seed ─────────────────────────────────
    ptc_path = _CV_DIR / _PAIR_TEST_CLASSES_PAT.format(seed=args.gcv_seed)
    pair_test_classes = np.load(str(ptc_path))
    print(f"Pair test classes shape: {pair_test_classes.shape}", flush=True)

    # ── verify PDB directory ──────────────────────────────────────────────────
    n_pdbs = len(list(_PDB_DIR.glob("*.pdb")))
    if n_pdbs == 0:
        raise RuntimeError(
            f"No PDB files found in {_PDB_DIR}. "
            "Run convert_varchamp_cifs_to_pdb.py first."
        )
    print(f"Found {n_pdbs} PDB files in {_PDB_DIR}", flush=True)

    # ── run SAAMBE-3D per fold ────────────────────────────────────────────────
    all_preds:   list[float] = []
    all_classes: list[int]   = []
    n_ok = n_missing = n_error = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for fold, train_idx, test_idx in fold_splits:
            print(f"\nFold {fold}: {len(test_idx)} test variants", flush=True)
            for idx in test_idx:
                complex_id, variant = vt_ids[idx]

                # variant positions in labels file are 0-based; PDB is 1-based
                wt_res  = variant[0]
                pos_str = str(int(variant[1:-1]) + 1)   # 0-based → 1-based
                mt_res  = variant[-1]

                interactor, partner = _split_pair(complex_id)
                pdb_path, is_rev = _resolve_pdb(interactor, partner)

                if pdb_path is None:
                    print(f"  MISSING PDB: {complex_id} → {interactor}-{partner}_model.pdb",
                          flush=True)
                    all_preds.append(float("nan"))
                    all_classes.append(int(pair_test_classes[idx]))
                    n_missing += 1
                    continue

                chain = "B" if is_rev else "A"
                tmp_out = Path(tmpdir) / f"out_{fold}_{idx}.txt"
                try:
                    score, _ = _call_saambe(pdb_path, chain, pos_str,
                                            wt_res, mt_res, model_flag, tmp_out)
                    all_preds.append(score)
                    all_classes.append(int(pair_test_classes[idx]))
                    n_ok += 1
                except Exception as exc:
                    n_error += 1
                    if n_error <= 5:
                        print(f"  ERROR {complex_id} {variant}: {exc}", flush=True)
                    all_preds.append(float("nan"))
                    all_classes.append(int(pair_test_classes[idx]))
                    if tmp_out.exists():
                        tmp_out.unlink()

                if (n_ok + n_missing + n_error) % 200 == 0:
                    done = n_ok + n_missing + n_error
                    print(f"  {done}/{total_test} done  "
                          f"ok={n_ok} miss={n_missing} err={n_error}", flush=True)

    print(f"\nDone: {n_ok} ok  {n_missing} missing PDB  {n_error} errors", flush=True)
    print(f"Total predictions: {len(all_preds)}", flush=True)

    np.save(out_preds,   np.array(all_preds,   dtype=np.float32))
    np.save(out_classes, np.array(all_classes, dtype=np.int8))
    print(f"Saved: {out_preds}  shape={np.array(all_preds).shape}", flush=True)
    print(f"Saved: {out_classes}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAAMBE-3D GCV inference for sahni_fragoza_varchamp2026"
    )
    p.add_argument(
        "--model-type", default="regression",
        choices=["regression", "classification"],
        help="regression → ddG score (model -d 1); classification → binary (model -d 0)",
    )
    p.add_argument(
        "--gcv-seed", type=int, default=0,
        help="GCV seed index to use for fold splits (default: 0). "
             "Since SAAMBE-3D is deterministic, one seed is sufficient.",
    )
    p.add_argument(
        "--outdir",
        default="/data/ross/ppi_lossgain/interaction_loss/publication/results_revisions/macro_aucs/",
        help="Output directory for pred arrays",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
