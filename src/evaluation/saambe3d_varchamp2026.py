#!/usr/bin/env python
"""SAAMBE-3D blind-test evaluation on VarChAMP 2026.

Reads training_data.csv (VarChAMP rows), converts AF3 CIF structures to PDB,
runs SAAMBE-3D on each variant, and saves C1/C2/C3 stratified outputs.

Chain assignment: interactor = chain A, partner = chain B.
CIF filename convention: {interactor.lower()}-{partner.lower()}_model.cif
  (or reversed: {partner.lower()}-{interactor.lower()}_model.cif → reversed chains)

Usage:
    conda run -n pytorch_env python saambe3d_varchamp2026.py [--model-type regression]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── fixed paths ────────────────────────────────────────────────────────────────
_TRAINING_DATA_CSV = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/data/sfvc2026_labeled_data.csv"
)
_CIF_DIR  = Path("/data/ross/ppi_lossgain/interaction_loss/2026/af3_out/models")
_PDB_DIR  = Path("/data/ross/ppi_lossgain/interaction_loss/2026/af3_out/pdbs")
_OUT_DIR  = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)
_SAAMBE3D_PY = Path(__file__).parent / "saambe-3d.py"

_DESCRIPTION = "SAAMBE-3D (Sahni+Fragoza train)"


def check_pdbs(pdb_dir: Path) -> None:
    n = len(list(pdb_dir.glob("*.pdb")))
    if n == 0:
        raise RuntimeError(
            f"No PDB files found in {pdb_dir}. "
            "Run convert_varchamp_cifs_to_pdb.py first (ppi env)."
        )
    print(f"Found {n} PDB files in {pdb_dir}", flush=True)


# ── PDB path resolution ───────────────────────────────────────────────────────

def _resolve_pdb(interactor: str, partner: str, pdb_dir: Path) -> tuple[Path | None, bool]:
    """Return (pdb_path, is_reversed).  is_reversed=True → partner is chain A."""
    a, b = interactor.lower(), partner.lower()
    fwd = pdb_dir / f"{a}-{b}_model.pdb"
    rev = pdb_dir / f"{b}-{a}_model.pdb"
    if fwd.exists():
        return fwd, False
    if rev.exists():
        return rev, True
    return None, False


# ── SAAMBE-3D subprocess ──────────────────────────────────────────────────────

def _call_saambe(pdb_path: Path, chain: str, pos_1based: str,
                 wt_res: str, mt_res: str, model_flag: str,
                 tmp_out: Path) -> tuple[float, int]:
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


# ── Marcotte stratification ───────────────────────────────────────────────────

def _compute_marcotte(vc: pd.DataFrame, sf_proteins: set[str]) -> np.ndarray:
    int_in = vc["interactor"].isin(sf_proteins).values
    par_in = vc["partner"].isin(sf_proteins).values
    classes = np.where(int_in & par_in, 1, np.where(int_in | par_in, 2, 3))
    for c in [1, 2, 3]:
        print(f"  C{c}: {(classes == c).sum()} rows", flush=True)
    return classes


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    check_pdbs(_PDB_DIR)

    df = pd.read_csv(_TRAINING_DATA_CSV)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    sf_proteins: set[str] = set(df.loc[sf_mask, "interactor"]) | set(df.loc[sf_mask, "partner"])

    vc_mask = df["dataset"].str.contains("VarChAMP")
    vc = df[vc_mask].copy().reset_index(drop=True)
    print(f"VarChAMP rows: {len(vc)}", flush=True)

    print("Marcotte stratification (vs Sahni+Fragoza train):", flush=True)
    classes = _compute_marcotte(vc, sf_proteins)

    model_flag = "1" if args.model_type == "regression" else "0"

    scores  = np.full(len(vc), np.nan)
    labels  = vc["perturbed"].astype(int).values
    vt_ids  = (vc["interactor"] + " " + vc["mutation"]).values

    n_ok, n_missing, n_err = 0, 0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, row in vc.iterrows():
            pdb_path, is_rev = _resolve_pdb(row["interactor"], row["partner"], _PDB_DIR)
            if pdb_path is None:
                n_missing += 1
                if n_missing <= 3:
                    print(f"  MISSING PDB: {row['interactor']}-{row['partner']}", flush=True)
                continue

            chain = "B" if is_rev else "A"
            mut = row["mutation"]
            pos_1based = str(int(mut[1:-1]))   # training_data.csv: 1-based positions
            wt_res, mt_res = mut[0], mut[-1]

            tmp_out = Path(tmpdir) / f"out_{i}.txt"
            try:
                score, _ = _call_saambe(pdb_path, chain, pos_1based,
                                        wt_res, mt_res, model_flag, tmp_out)
                scores[i] = score
                n_ok += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  ERROR {row['interactor']} {row['mutation']}: {exc}", flush=True)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(vc)} done  ok={n_ok} miss={n_missing} err={n_err}",
                      flush=True)

    print(f"\nDone: {n_ok} ok  {n_missing} missing PDB  {n_err} errors", flush=True)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in [1, 2, 3]:
        mask = (classes == c) & ~np.isnan(scores)
        if not mask.any():
            print(f"  C{c}: no scored rows, skipping", flush=True)
            continue
        np.save(_OUT_DIR / f"{_DESCRIPTION}_c{c}_preds.npy",  scores[mask])
        np.save(_OUT_DIR / f"{_DESCRIPTION}_c{c}_labels.npy", labels[mask])
        np.save(_OUT_DIR / f"{_DESCRIPTION}_c{c}_vt_ids.npy", vt_ids[mask])
        print(f"  C{c}: saved {mask.sum()} rows", flush=True)

    print(f"Results saved → {_OUT_DIR}/", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["regression", "classification"],
                   default="regression")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
