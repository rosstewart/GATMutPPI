#!/usr/bin/env python
"""MINT seq-diff / site-diff MLP group cross-validation training script.

Mirrors esignet_gcv_iter.py but uses the MINT embedding-based MLP predictors
(MINTSeqDiff / MINTSiteDiff from predictors/mint_mlp.py) instead of eSIG-Net.

Both predictors are sklearn-style (no GPU required at inference):
  mint_seq_diff  — mean(mut_A - wt_A) | mean_partner  →  small MLP
  mint_site_diff — emb_mut_A[site] - emb_wt_A[site] | mean_partner  →  small MLP

MINT embeddings must be precomputed before running this script; use
precompute_mint_embeddings.py to generate the cache.

Usage:
    conda run -n ppi python mint_gcv_iter.py --dataset sahni_fragoza \\
        --mint-cache /data/ross/ppi_lossgain/interaction_loss/2026/mint_cache/sahni_fragoza.pkl

    conda run -n ppi python mint_gcv_iter.py \\
        --dataset sahni_fragoza_varchamp1p_cava \\
        --predictor site_diff \\
        --mint-cache /path/to/cache.pkl \\
        --n-gcv 30 \\
        --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── shared code (vendored in-repo, see src/evaluation/esignet_gcv_iter_legacy.py
#    and src/evaluation/predictors/) ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esignet_gcv_iter_legacy import (    # noqa: E402
    DATASET_CONFIGS,
    DatasetConfig,
    _CV_DIR,
    load_data,
    align_to_vt_ids,
    swing_to_esignet,
    _compute_class_aucs,
)

import predictors.mint_mlp as _mint_mod   # noqa: E402
from predictors.mint_mlp import MINTSeqDiff, MINTSiteDiff  # noqa: E402
from predictors.nn_base import load_cache, zero_based_variant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PREDICTOR_MAP = {
    "seq_diff":  MINTSeqDiff,
    "site_diff": MINTSiteDiff,
}


# ── MINT cache audit ──────────────────────────────────────────────────────────

def audit_mint_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    predictor: str,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Load the MINT cache and report mean/res key hit rates.

    Aborts when require=True and the mutant mean-key hit rate falls below
    min_hit_rate.  Missing mean keys mean MINTSeqDiff sees all-zero embeddings;
    missing res keys additionally cripple MINTSiteDiff.
    """
    path = _mint_mod.CACHE_PATH
    print(f"\n── MINT cache audit ──", flush=True)
    print(f"path: {path}", flush=True)

    cache = load_cache(path)
    if cache is None:
        msg = (
            f"MINT cache not found at {path}. "
            f"Run precompute_mint_embeddings.py --dataset {cfg.name} "
            f"to generate it, then pass the path via --mint-cache."
        )
        if require:
            raise SystemExit("ABORT: " + msg + "\nPass --no-require-mint to override.")
        print("WARNING: " + msg, flush=True)
        return

    print(f"entries: {len(cache)}", flush=True)

    # Filter out NaN rows (unmatched vt_ids produce NaN mutations)
    valid_df    = ordered_df.dropna(subset=["Mutation"])
    interactors = valid_df["refseq_id"].astype(str)
    partners    = valid_df["partner"].astype(str)
    mutations   = valid_df["Mutation"].astype(str)

    unique_pairs = set(zip(interactors, partners))
    unique_muts  = set(zip(interactors, partners, mutations))

    def _rate(hits, total):
        return f"{hits}/{total} ({hits / max(total, 1):.1%})"

    mean_wt_hits  = sum(1 for a, b    in unique_pairs if f"mean_{a}_{b}" in cache)
    mean_mut_hits = sum(1 for a, b, m in unique_muts
                        if f"mean_{a}_{b}_{zero_based_variant(m)}" in cache)
    print(f"mean WT  keys: {_rate(mean_wt_hits,  len(unique_pairs))}", flush=True)
    print(f"mean MUT keys: {_rate(mean_mut_hits, len(unique_muts))}", flush=True)

    if predictor == "site_diff":
        res_wt_hits  = sum(1 for a, b    in unique_pairs if f"res_wt_pair_{a}_{b}" in cache)
        res_mut_hits = sum(1 for a, b, m in unique_muts
                           if f"res_mut_pair_{zero_based_variant(m)}_{a}_{b}" in cache)
        print(f"res WT  keys: {_rate(res_wt_hits,  len(unique_pairs))}", flush=True)
        print(f"res MUT keys: {_rate(res_mut_hits, len(unique_muts))}", flush=True)

    mut_rate = mean_mut_hits / max(len(unique_muts), 1)
    if mut_rate < min_hit_rate:
        msg = (
            f"MINT mean MUT hit rate {mut_rate:.1%} is below "
            f"--min-mint-hit-rate {min_hit_rate:.1%}. "
            f"Run precompute_mint_embeddings.py --dataset {cfg.name} "
            f"and pass its output via --mint-cache."
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


# ── main GCV loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.mint_cache:
        _mint_mod.CACHE_PATH = args.mint_cache
        MINTSeqDiff._cache_path  = args.mint_cache
        MINTSiteDiff._cache_path = args.mint_cache
        print(f"MINT cache path overridden: {args.mint_cache}", flush=True)

    cfg       = DATASET_CONFIGS[args.dataset]
    data_root = Path(args.data_root)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    PredictorClass = _PREDICTOR_MAP[args.predictor]
    print(f"Predictor: {PredictorClass().name}", flush=True)

    # ── load and align dataset ────────────────────────────────────────────────
    print(f"Loading dataset: {cfg.name}", flush=True)
    df = load_data(cfg, data_root)
    print(f"  {len(df)} rows after filtering", flush=True)

    ordered_df = align_to_vt_ids(df, cfg)
    print(f"  {len(ordered_df)} rows after vt_ids alignment", flush=True)

    audit_mint_cache(ordered_df, cfg, args.predictor, args.min_mint_hit_rate, args.require_mint)

    labels = ordered_df["Y2H_score"].fillna(0).astype(int).values  # NaN rows excluded via valid_mask

    # ── GCV iterations ────────────────────────────────────────────────────────
    macro_aucs: list[np.ndarray] = []
    micro_aucs: list[list]       = []
    detailed_results             = {"iterations": {}}

    for gcv_seed in range(args.n_gcv):
        print(f"\n{'='*60}", flush=True)
        print(f"GCV seed {gcv_seed}/{args.n_gcv - 1}", flush=True)

        with open(_CV_DIR / cfg.fold_splits_pat.format(seed=gcv_seed), "rb") as f:
            fold_splits = pickle.load(f)
        fold_n_test = [len(test_idx) for _, _, test_idx in fold_splits]

        pair_test_classes = np.load(
            str(_CV_DIR / cfg.pair_test_classes_pat.format(seed=gcv_seed))
        )

        all_preds:  list[float] = []
        all_labels: list[int]   = []

        for fold, train_idx, test_idx in fold_splits:
            # Filter NaN rows (unmatched vt_ids) from both train and test;
            # insert NaN predictions for missing test rows so pair_test_classes aligns.
            train_slice = ordered_df.iloc[train_idx]
            train_slice = train_slice[train_slice["Target_Seq"].notna()].reset_index(drop=True)
            train_df = swing_to_esignet(train_slice)

            test_slice  = ordered_df.iloc[test_idx].reset_index(drop=True)
            test_valid  = test_slice["Target_Seq"].notna()
            test_df     = swing_to_esignet(test_slice[test_valid].reset_index(drop=True))

            print(
                f"\nFold {fold}: {len(train_idx)} train / {len(test_idx)} test — "
                f"fitting {args.predictor}...",
                flush=True,
            )

            predictor = PredictorClass(seed=args.seed)
            predictor.fit(train_df)
            pred_valid = predictor.predict(test_df)

            # Re-expand to full test_idx length, NaN for unmatched rows
            pred_proba = np.full(len(test_idx), np.nan)
            pred_proba[test_valid.values] = pred_valid

            all_preds.extend(pred_proba.tolist())
            all_labels.extend(labels[test_idx].tolist())

            print(f"  Fold {fold} done", flush=True)

        all_preds_arr  = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        print(f"\nGCV seed {gcv_seed} — per-class AUROCs:", flush=True)
        micro_auc, macro_auc, fold_results = _compute_class_aucs(
            all_preds_arr, all_labels_arr, pair_test_classes, fold_n_test
        )
        print(f"micro AUC (c1/c2/c3): {micro_auc}", flush=True)
        print(f"macro AUC (c1/c2/c3): {macro_auc}", flush=True)

        micro_aucs.append(micro_auc)
        macro_aucs.append(macro_auc)
        detailed_results["iterations"][gcv_seed] = {
            "folds":     fold_results,
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
        }

    # ── save results ──────────────────────────────────────────────────────────
    stem = f"MINT_{args.predictor}_{cfg.name}"
    np.save(outdir / f"{stem}_micro_aucs.npy", np.array(micro_aucs))
    np.save(outdir / f"{stem}_macro_aucs.npy", np.array(macro_aucs))
    with open(outdir / f"{stem}_detailed_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)

    print(f"\nResults saved to {outdir}/", flush=True)
    print(f"  {stem}_micro_aucs.npy  shape={np.array(micro_aucs).shape}", flush=True)
    print(f"  {stem}_macro_aucs.npy  shape={np.array(macro_aucs).shape}", flush=True)
    print(f"  {stem}_detailed_results.pkl", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MINT MLP GCV training (mirrors SWING / eSIG-Net pipeline)"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset configuration to use",
    )
    p.add_argument(
        "--predictor",
        default="seq_diff",
        choices=list(_PREDICTOR_MAP),
        help="MINT predictor variant (default: seq_diff)",
    )
    p.add_argument(
        "--data-root",
        default="/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss",
        help="Root directory containing pos/neg/extra files",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MLP training (default: 42)",
    )
    p.add_argument(
        "--n-gcv",
        type=int,
        default=30,
        help="Number of GCV iterations (default: 30)",
    )
    p.add_argument(
        "--outdir",
        default=str(_CV_DIR),
        help="Output directory for results (default: CV splits dir)",
    )
    p.add_argument(
        "--mint-cache",
        default="",
        help=(
            "Path to MINT embedding cache .pkl "
            "(default: built-in CACHE_PATH in mint_mlp.py). "
            "Generate with precompute_mint_embeddings.py."
        ),
    )
    p.add_argument(
        "--min-mint-hit-rate",
        type=float,
        default=0.9,
        help="Minimum sep-MUT key hit rate before --require-mint aborts (default: 0.9).",
    )
    p.add_argument(
        "--require-mint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Abort if the MINT cache is missing or has hit rate below "
            "--min-mint-hit-rate (default: True). "
            "Pass --no-require-mint to run with missing-key rows falling back to prior."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
