#!/usr/bin/env python
"""Run the VarChAMP full-pooled (VCFP) blind test for a single comparator method,
end to end, in one command.

Historically this required three hand-invoked steps per method: run a
method-specific "vc1pcava" supplement script, merge its output into the main
VCFP arrays, then restratify C1/C2/C3 classification across all methods. This
entry point chains all three steps for the chosen `--method`:

  1. Train/predict the method on the VC1p+CAVA entries (the subset of the
     VCFP test set whose graphs use gene-name/Entrez IDs rather than UniProt
     IDs, so they need UniProt remapping before they can be scored/classified
     consistently with the rest of the VCFP set).
  2. Merge the resulting predictions into that method's main VCFP
     `_c{1,2,3}_*.npy` arrays (src/analysis/merge_vc1pcava_into_main.py).
  3. Restratify C1/C2/C3 classification for that method using the single
     canonical SF (Sahni+Fragoza) protein set (src/analysis/restratify_vcfp_blind_test.py).

Usage:
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mutpredppi [--device cuda:1]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method esignet [--device cuda:1] [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mint --predictor seq_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method pplm --predictor site_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method swing [--test-pretrain] [--seed 42]

For MutPred2 (no model to train — external tool output is parsed from a CSV),
see src/analysis/import_mutpred2_vcfp_scores.py instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent          # src/evaluation
_ANALYSIS_DIR = _EVAL_DIR.parent / "analysis"        # src/analysis
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_ANALYSIS_DIR))

from merge_vc1pcava_into_main import merge_method            # noqa: E402
from restratify_vcfp_blind_test import restratify_one_method  # noqa: E402

METHODS = ("mutpredppi", "esignet", "mint", "pplm", "swing")


def run_method(
    method: str,
    predictor: str = "seq_diff",
    test_pretrain: bool = False,
    seed: int = 42,
    device: str = "cuda:1",
) -> str | None:
    """Train/predict `method` on VC1p+CAVA and return its description string.

    Returns None if the method produced no valid supplement entries (only
    possible for SWING, if position validation drops every row).
    """
    if method == "mutpredppi":
        import supplement_mutpredppi_vc1pcava as mod
        return mod.run(device=device)
    if method == "esignet":
        import supplement_esignet_vc1pcava as mod
        return mod.run(device=device, seed=seed)
    if method == "mint":
        import supplement_mint_vc1pcava as mod
        return mod.run(predictor=predictor, seed=seed)
    if method == "pplm":
        import supplement_pplm_vc1pcava as mod
        return mod.run(predictor=predictor, seed=seed)
    if method == "swing":
        import supplement_swing_vc1pcava as mod
        return mod.run(test_pretrain=test_pretrain, seed=seed)
    raise ValueError(f"Unknown method: {method!r} (expected one of {METHODS})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--predictor", default="seq_diff", choices=("seq_diff", "site_diff"),
                    help="MINT/PPLM only.")
    ap.add_argument("--test-pretrain", action="store_true", help="SWING only.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1", help="MutPred-PPI/eSIG-Net only.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run training/prediction and write the supplement npy files, "
                         "but skip merging into / restratifying the main VCFP arrays.")
    args = ap.parse_args()

    description = run_method(
        args.method,
        predictor=args.predictor,
        test_pretrain=args.test_pretrain,
        seed=args.seed,
        device=args.device,
    )

    if description is None:
        print("\nNo supplement entries produced — skipping merge/restratify.")
        return

    print(f"\n=== Merging '{description}' vc1pcava supplement into main VCFP arrays ===")
    merge_method(description, dry_run=args.dry_run)

    print(f"\n=== Restratifying '{description}' C1/C2/C3 classification ===")
    restratify_one_method(description, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
