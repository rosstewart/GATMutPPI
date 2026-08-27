#!/usr/bin/env python
"""Generate LaTeX-ready data table statistics for each training/evaluation dataset.

Prints protein, pair, variant, and variant-partner counts for:
  - Sahni (original Y2H screen)
  - Sahni + Fragoza (combined SWING training set)
  - Fragoza / SWING (nature-filtered Y2H)
  - VarChAMP (CAVA + VarChAMP1p blind test)
  - MegaScale stability (pretraining set)

Output format per dataset: six values separated by ' & ' (LaTeX table row fragment):
  #proteins  #pairs  #variants  #variant-partners  #disrupting  #non-disrupting
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


def _count_stats(
    proteins: set, pairs: set, variants: set,
    variant_partners: set, disrupting: set, non_disrupting: set,
    label: str,
) -> None:
    print(
        f"{label}: "
        + " & ".join(str(x) for x in [
            len(proteins), len(pairs), len(variants),
            len(variant_partners), len(disrupting), len(non_disrupting),
        ])
    )


def sahni_stats(data_dir: str) -> None:
    df_s = pd.read_csv(f"{data_dir}/sahni_train.csv")
    with open(f"{data_dir}/home/sahni/refseq_to_uniprot.pkl", "rb") as f:
        refseq_to_uniprot = pickle.load(f)
    df_s["refseq_id_orig"] = df_s["refseq_id"]
    df_s["refseq_id"] = df_s["refseq_id"].map(refseq_to_uniprot)

    proteins, pairs, variants, variant_partners, disrupting, non_disrupting = (
        set(), set(), set(), set(), set(), set()
    )
    for _, row in df_s.iterrows():
        interactor = row["refseq_id"]
        variant = row["Mutation"]
        vt_id = f"{interactor} {variant}"
        partner = row["partner"]
        triplet = f"{vt_id} {partner}"
        refseq_id_orig = row["refseq_id_orig"]

        cif1 = f"{data_dir}/home/sahni/af3_models/fold_{refseq_id_orig.lower().replace('-','_')}_{partner.lower().replace('-','_')}_model_0.cif"
        cif2 = f"{data_dir}/home/sahni/af3_models/fold_{partner.lower().replace('-','_')}_{refseq_id_orig.lower().replace('-','_')}_model_0.cif"
        if not os.path.exists(cif1) and not os.path.exists(cif2):
            continue

        proteins.add(interactor)
        proteins.add(partner)
        if (partner, interactor) not in pairs:
            pairs.add((interactor, partner))
        variants.add(vt_id)
        variant_partners.add(triplet)

        if row["Y2H_score"] == 1:  # 1 = disrupting (reversed convention)
            disrupting.add(triplet)
        else:
            non_disrupting.add(triplet)

    _count_stats(proteins, pairs, variants, variant_partners, disrupting, non_disrupting,
                 "Sahni")


def sahni_fragoza_stats(data_dir: str) -> None:
    df_sf = pd.read_csv(f"{data_dir}/sahni_fragoza_train.csv")

    proteins, pairs, variants, variant_partners, disrupting, non_disrupting = (
        set(), set(), set(), set(), set(), set()
    )
    for _, row in df_sf.iterrows():
        interactor = row["refseq_id"]
        variant = row["Mutation"]
        vt_id = f"{interactor} {variant}"
        partner = row["partner"]
        triplet = f"{vt_id} {partner}"

        cif1 = f"{data_dir}/swing_train/af3_models/fold_{interactor.lower().replace('-','_')}_{partner.lower().replace('-','_')}_model_0.cif"
        cif2 = f"{data_dir}/swing_train/af3_models/fold_{partner.lower().replace('-','_')}_{interactor.lower().replace('-','_')}_model_0.cif"
        if not os.path.exists(cif1) and not os.path.exists(cif2):
            continue

        proteins.add(interactor)
        proteins.add(partner)
        if (partner, interactor) not in pairs:
            pairs.add((interactor, partner))
        variants.add(vt_id)
        variant_partners.add(triplet)

        if row["Y2H_score"] == 1:  # 1 = disrupting (reversed convention)
            disrupting.add(triplet)
        else:
            non_disrupting.add(triplet)

    _count_stats(proteins, pairs, variants, variant_partners, disrupting, non_disrupting,
                 "Sahni+Fragoza")


def fragoza_stats(data_dir: str) -> None:
    """SWING/Fragoza nature-paper subset."""
    df_f = pd.read_csv(
        f"{data_dir}/home/SWING/Data/MutInt_Model/Mutation_perturbation_model.csv"
    )

    proteins, pairs, variants, variant_partners, disrupting, non_disrupting = (
        set(), set(), set(), set(), set(), set()
    )
    discard_triplets: set = set()

    for _, row in df_f.iterrows():
        if row["Type"] != "Mutant" or row["Data"] != "nature":
            continue

        interactor = row["Target_UPID"]
        variant = row["Mutation"]
        vt_id = f"{interactor} {variant}"
        partner = row["Interactor_UPID"]
        triplet = f"{vt_id} {partner}"

        cif1 = f"{data_dir}/swing_train/af3_models/fold_{interactor.lower().replace('-','_')}_{partner.lower().replace('-','_')}_model_0.cif"
        cif2 = f"{data_dir}/swing_train/af3_models/fold_{partner.lower().replace('-','_')}_{interactor.lower().replace('-','_')}_model_0.cif"
        if not os.path.exists(cif1) and not os.path.exists(cif2):
            continue

        proteins.add(interactor)
        proteins.add(partner)
        if (partner, interactor) not in pairs:
            pairs.add((interactor, partner))
        variants.add(vt_id)
        variant_partners.add(triplet)

        if row["Y2H_score"] == 1 and triplet not in discard_triplets:
            if triplet in non_disrupting:
                discard_triplets.add(triplet)
                disrupting.discard(triplet)
                non_disrupting.discard(triplet)
            else:
                disrupting.add(triplet)
        elif triplet not in discard_triplets:
            if triplet in disrupting:
                discard_triplets.add(triplet)
                disrupting.discard(triplet)
                non_disrupting.discard(triplet)
            else:
                non_disrupting.add(triplet)

    _count_stats(proteins, pairs, variants, variant_partners, disrupting, non_disrupting,
                 "Fragoza/SWING")


def varchamp_stats(data_dir: str) -> None:
    """CAVA + VarChAMP1p combined blind-test set."""
    cava_df = pd.read_csv(
        f"{data_dir}/home/data_interaction_loss/cava_variant_interaction.tsv",
        header=None, sep="\t",
    )
    cava_df.columns = ["refseq_id", "Mutation", "partner", "Y2H_score"]

    varchamp1p_df = pd.read_csv(
        f"{data_dir}/home/data_interaction_loss/varchamp1p_variant_interaction.tsv",
        header=None, sep="\t",
    )
    varchamp1p_df.columns = ["refseq_id", "Mutation", "partner", "Y2H_score"]

    df_vc = pd.concat([cava_df, varchamp1p_df], ignore_index=True)

    with open(f"{data_dir}/varchamp1p/varchamp1p_gene_symbol_to_uniprot.pkl", "rb") as f:
        varchamp1p_gene_symbol_to_uniprot = pickle.load(f)
    with open(f"{data_dir}/varchamp1p/cava_gene_symbol_to_uniprot.pkl", "rb") as f:
        cava_gene_symbol_to_uniprot = pickle.load(f)
    combined_dict = {**varchamp1p_gene_symbol_to_uniprot, **cava_gene_symbol_to_uniprot}

    def _remove_orf_id(value: str) -> str:
        return "_".join(value.split("_")[:-1])

    def _update_id(value: str) -> str:
        gene_name = _remove_orf_id(value)
        return combined_dict.get(gene_name, value)

    df_vc["refseq_id_gene_orf"] = df_vc["refseq_id"]
    df_vc["partner_gene_orf"]   = df_vc["partner"]
    df_vc["refseq_id_gene"]     = df_vc["refseq_id"].apply(_remove_orf_id)
    df_vc["partner_gene"]       = df_vc["partner"].apply(_remove_orf_id)
    df_vc["refseq_id"]          = df_vc["refseq_id"].apply(_update_id)
    df_vc["partner"]            = df_vc["partner"].apply(_update_id)

    with open(f"{data_dir}/home/varchamp1p/seq_confirmed_variants.pkl", "rb") as f:
        seq_confirmed_variants = set(pickle.load(f))
    with open(f"{data_dir}/home/cava/seq_confirmed_variants.pkl", "rb") as f:
        seq_confirmed_variants = seq_confirmed_variants.union(set(pickle.load(f)))

    proteins, pairs, variants, variant_partners, disrupting, non_disrupting = (
        set(), set(), set(), set(), set(), set()
    )
    discard_triplets: set = set()

    for _, row in df_vc.iterrows():
        interactor       = row["refseq_id"]
        variant          = row["Mutation"]
        vt_id            = f"{interactor} {variant}"
        partner          = row["partner"]
        triplet          = f"{vt_id} {partner}"
        refseq_id_gene   = row["refseq_id_gene"]
        partner_gene     = row["partner_gene"]
        refseq_id_gene_orf = row["refseq_id_gene_orf"]
        partner_gene_orf   = row["partner_gene_orf"]

        if (refseq_id_gene_orf, variant, partner_gene_orf, row["Y2H_score"]) not in seq_confirmed_variants:
            continue

        ri_g = refseq_id_gene.lower().replace("-", "_")
        pg   = partner_gene.lower().replace("-", "_")
        has_cif = any(
            os.path.exists(p) for p in [
                f"{data_dir}/varchamp1p/af3_models/fold_{ri_g}_{pg}_model_0.cif",
                f"{data_dir}/varchamp1p/af3_models/fold_{pg}_{ri_g}_model_0.cif",
                f"{data_dir}/cava/af3_models/fold_{ri_g}_{pg}_model_0.cif",
                f"{data_dir}/cava/af3_models/fold_{pg}_{ri_g}_model_0.cif",
            ]
        )
        if not has_cif:
            continue

        proteins.add(interactor)
        proteins.add(partner)
        if (partner, interactor) not in pairs:
            pairs.add((interactor, partner))
        variants.add(vt_id)
        variant_partners.add(triplet)

        if row["Y2H_score"] == 1 and triplet not in discard_triplets:
            if triplet in non_disrupting:
                discard_triplets.add(triplet)
                disrupting.discard(triplet)
                non_disrupting.discard(triplet)
            else:
                disrupting.add(triplet)
        elif triplet not in discard_triplets:
            if triplet in disrupting:
                discard_triplets.add(triplet)
                disrupting.discard(triplet)
                non_disrupting.discard(triplet)
            else:
                non_disrupting.add(triplet)

    _count_stats(proteins, pairs, variants, variant_partners, disrupting, non_disrupting,
                 "VarChAMP")


def megascale_stability_stats() -> None:
    """MegaScale stability pretraining set (single-chain, no pairs)."""
    pos_df = pd.read_csv(
        "/home/rcstewart/jose_2016_lossgain_datasets/rasp/stability_train/Stability.pos",
        header=None, sep="\t",
    )
    pos_df.columns = ["pdb", "chain", "pos", "ref", "alt"]

    neg_df = pd.read_csv(
        "/home/rcstewart/jose_2016_lossgain_datasets/rasp/stability_train/Stability.neg",
        header=None, sep="\t",
    )
    neg_df.columns = ["pdb", "chain", "pos", "ref", "alt"]

    df_stability = pd.concat([pos_df, neg_df], ignore_index=True)
    df_stability["disrupt"] = [1] * len(pos_df) + [0] * len(neg_df)

    proteins: set = set()
    variants: set = set()
    disrupting: set = set()
    non_disrupting: set = set()

    for _, row in df_stability.iterrows():
        interactor = row["pdb"]
        variant = f"{row['ref']}{row['pos']}{row['alt']}"
        vt_id = f"{interactor} {variant}"

        proteins.add(interactor)
        variants.add(vt_id)

        if row["disrupt"] == 1:
            disrupting.add(vt_id)
        else:
            non_disrupting.add(vt_id)

    print(
        f"MegaScale stability: "
        + " & ".join(str(x) for x in [
            len(proteins), "-", len(variants), "-",
            len(disrupting), len(non_disrupting),
        ])
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Print dataset statistics for LaTeX tables")
    p.add_argument(
        "--data-dir",
        default="/data/ross/ppi_lossgain/interaction_loss",
        help="Base data directory (default: /data/ross/ppi_lossgain/interaction_loss)",
    )
    p.add_argument(
        "--dataset",
        choices=["sahni", "sahni_fragoza", "fragoza", "varchamp", "megascale", "all"],
        default="all",
        help="Which dataset to report (default: all)",
    )
    args = p.parse_args()

    run_all = args.dataset == "all"

    if run_all or args.dataset == "sahni":
        sahni_stats(args.data_dir)
    if run_all or args.dataset == "sahni_fragoza":
        sahni_fragoza_stats(args.data_dir)
    if run_all or args.dataset == "fragoza":
        fragoza_stats(args.data_dir)
    if run_all or args.dataset == "varchamp":
        varchamp_stats(args.data_dir)
    if run_all or args.dataset == "megascale":
        megascale_stability_stats()


if __name__ == "__main__":
    main()
