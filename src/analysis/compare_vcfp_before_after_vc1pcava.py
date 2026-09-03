#!/usr/bin/env python
"""Compare VCFP blind test performance before/after adding VC1p+CAVA entries.

Loads existing per-class npy files (14,116 entries) and supplemental VC1p+CAVA
result files (*_vc1pcava_c3_*.npy), then produces a side-by-side C3 ROC plot
showing: existing-only vs combined vs VC1p+CAVA-only.

Usage:
    conda run -n ppi python src/analysis/compare_vcfp_before_after_vc1pcava.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

EVAL_DIR = Path(
    "/data/ross/ppi_lossgain/interaction_loss/publication/results/varchamp_seqcnf_newvar_eval"
)
SAVE_DIR = EVAL_DIR / "roc_plots"
SAVE_DPI = 300

# Methods with supplement files
METHODS = {
    "MutPred-PPI": "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)",
    "SWING":       "SWING (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SWING (TP)":  "SWING (test pretrain, Sahni+Fragoza train) (varchamp_full_pooled)",
}

COLORS = {
    "MutPred-PPI": "#1f77b4",
    "SWING":       "#ff7f0e",
    "SWING (TP)":  "#d62728",
}


def load_c3(method_key: str) -> tuple[np.ndarray, np.ndarray]:
    p = np.load(EVAL_DIR / f"{method_key}_c3_preds.npy",  allow_pickle=True).astype(float)
    l = np.load(EVAL_DIR / f"{method_key}_c3_labels.npy", allow_pickle=True).astype(int)
    return p, l


def load_supplement_c3(method_key: str) -> tuple[np.ndarray, np.ndarray]:
    pf = EVAL_DIR / f"{method_key}_vc1pcava_c3_preds.npy"
    lf = EVAL_DIR / f"{method_key}_vc1pcava_c3_labels.npy"
    if not pf.exists() or not lf.exists():
        return np.array([]), np.array([])
    p = np.load(pf, allow_pickle=True).astype(float)
    l = np.load(lf, allow_pickle=True).astype(int)
    return p, l


def safe_auc(labels, scores) -> float | None:
    valid = ~np.isnan(scores)
    l, s = labels[valid], scores[valid]
    if len(np.unique(l)) < 2 or len(l) < 4:
        return None
    return roc_auc_score(l, s)


def plot_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    views = ["Existing only", "VC1p+CAVA only", "Combined (existing + VC1p+CAVA)"]

    print("\nAUC comparison (C3):")
    print(f"{'Method':<20}  {'Existing':>10}  {'VC1p+CAVA':>12}  {'Combined':>10}")
    print("-" * 58)

    for display, method_key in METHODS.items():
        existing_p, existing_l = load_c3(method_key)
        supp_p,     supp_l     = load_supplement_c3(method_key)

        if len(existing_p) == 0:
            print(f"  {display}: no existing C3 data — skipping")
            continue

        combined_p = np.concatenate([existing_p, supp_p]) if len(supp_p) else existing_p
        combined_l = np.concatenate([existing_l, supp_l]) if len(supp_l) else existing_l

        auc_exist   = safe_auc(existing_l,  existing_p)
        auc_supp    = safe_auc(supp_l,      supp_p)     if len(supp_p) else None
        auc_combined= safe_auc(combined_l,  combined_p)

        auc_str = lambda v: f"{v:.3f}" if v is not None else "N/A"
        print(f"  {display:<20}  {auc_str(auc_exist):>10}  "
              f"{auc_str(auc_supp):>12}  {auc_str(auc_combined):>10}")

        color = COLORS.get(display, "#333333")

        for ax_idx, (p, l, view) in enumerate([
            (existing_p, existing_l,  views[0]),
            (supp_p,     supp_l,      views[1]),
            (combined_p, combined_l,  views[2]),
        ]):
            if len(p) == 0 or len(np.unique(l)) < 2:
                continue
            fpr, tpr, _ = roc_curve(l, p)
            roc_auc = auc(fpr, tpr)
            axes[ax_idx].plot(fpr, tpr, color=color, lw=2, alpha=0.85,
                              label=f"{display} (n={len(p)}, AUC={roc_auc:.3f})")

    for ax_idx, view in enumerate(views):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"C3 ({view})", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("VCFP Blind Test: Before/After Adding VC1p+CAVA (C3 only)", fontsize=14)
    plt.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / "roc_vcfp_before_after_vc1pcava.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
