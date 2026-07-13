"""CLI argument parsing and main() orchestration for the TriTanc pipeline."""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def main() -> None:
    """Main entry point: parse args, run pipeline stages, write outputs."""
    import sys
    from Bio import SeqIO

    parser = argparse.ArgumentParser(
        description="Taxonomy-aware metagenomic contig clustering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--bams", nargs="+", default=None,
                        help="Sorted BAMs (required unless --depth supplied)")
    parser.add_argument("--mmseqs-db", default=None,
                        help="MMseqs2 taxonomy DB (required unless --taxonomy supplied)")
    parser.add_argument("--taxonomy", default=None, help="Precomputed taxonomy TSV")
    parser.add_argument("--taxonomy-format", choices=["mmseqs2", "taxometer"],
                        default="mmseqs2")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Taxometer minimum lineage score")
    parser.add_argument("--ani", default=None, help="Precomputed skani TSV")
    parser.add_argument("--prot-sim", default=None,
                        help="Precomputed protein similarity TSV (query, ref, prot_sim). "
                             "If not supplied, Prodigal + MMseqs2 will be run automatically.")
    parser.add_argument("--skip-prot-sim", action="store_true",
                        help="Disable protein similarity entirely (faster, less sensitive).")
    parser.add_argument("--min-prot-sim", type=float, default=50.0,
                        help="Minimum mean amino-acid identity (0–100) to retain a protein pair.")
    parser.add_argument("--depth", default=None, help="Precomputed depth matrix TSV")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--threads", type=int, default=MMSEQS_THREADS)
    parser.add_argument("--min-len", type=int, default=MIN_LEN,
                        help="Minimum contig length for main/secondary clustering")
    parser.add_argument("--ani-threshold", type=float, default=ANI_THRESHOLD,
                        help="Override adaptive ANI threshold")
    parser.add_argument("--cov-threshold", type=float, default=None,
                        help="Override adaptive coverage correlation threshold")
    parser.add_argument("--min-af", type=float, default=ANI_MIN_AF,
                        help="Minimum skani alignment fraction to retain a pair")
    parser.add_argument("--coverage-as-tiebreaker", action="store_true",
                        help="Disable hard coverage gate; use coverage to rank only")
    parser.add_argument("--leiden-res-main", type=float, default=LEIDEN_RES_MAIN,
                        help="Leiden resolution for tier-1 (genus/species) clustering. "
                             "Higher = finer splits. Raise to 3.5 for dense (eg. oral) datasets and reduce to 2 for sparse datasets.")
    parser.add_argument("--leiden-res-secondary", type=float, default=LEIDEN_RES_SECONDARY,
                        help="Leiden resolution for tier-2 (above-genus) clustering.")
    parser.add_argument("--leiden-res-t4", type=float, default=LEIDEN_RES_T4,
                        help="Leiden resolution for tier-4 (unassigned de-novo) clustering.")
    parser.add_argument("--skip-cov-recovery", action="store_true",
                        help="Disable tier-5 coverage-only recovery pass.")
    parser.add_argument("--cov-recovery-r", type=float, default=0.80,
                        help="Minimum Spearman r for tier-5 coverage-only recovery.")
    parser.add_argument("--cov-recovery-tnf-min", type=float, default=0.80,
                        help="Minimum TNF cosine similarity for tier-5 recovery. "
                             "Set to 0 to disable the TNF guard (more permissive).")
    parser.add_argument("--min-checkm2-bp", type=int, default=100_000,
                        help="Minimum total bin size to include in checkm2_bin_list.txt")
    parser.add_argument("--tnf-gate-main", type=float, default=TNF_GATE_MAIN,
                        help="Minimum TNF cosine similarity for hybrid gate in Tiers 1/2. "
                             "Used when cov+TNF together substitute for missing ANI. "
                             "Lower = more permissive; 0 disables the TNF gate entirely.")
    parser.add_argument("--skip-checkm2", action="store_true")
    parser.add_argument("--skip-drep", action="store_true")
    parser.add_argument("--checkm2-db", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if not args.taxonomy and not args.mmseqs_db:
        parser.error("--mmseqs-db required when --taxonomy is not supplied")
    if not args.depth and not args.bams:
        parser.error("--bams required when --depth is not supplied")

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = args.checkpoint_dir or os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    use_cache = not args.no_cache

    def cached(name: str, ext: str) -> bool:
        return use_cache and _ckpt_exists(ckpt_dir, name, ext)

    check_tools(tools_needed_for(args))

    records = parse_fasta(args.fasta)

    taxonomy_path = args.taxonomy or run_mmseqs_taxonomy(
        args.fasta, args.mmseqs_db, args.outdir, args.threads
    )
    ani_path = args.ani or run_skani(args.fasta, args.outdir, args.threads)
    depth_path = args.depth or run_depth(args.bams, args.outdir)

    if cached("tax_df", "parquet") and cached("absent_from_taxonomy", "json"):
        tax_df = load_parquet(ckpt_dir, "tax_df", listcols=["lineage", "scores"])
        absent_from_taxonomy = load_json(ckpt_dir, "absent_from_taxonomy")
        log.info("Loaded taxonomy from checkpoint")
    else:
        if args.taxonomy_format == "taxometer":
            tax_df = parse_taxonomy_taxometer(taxonomy_path, min_score=args.min_score)
        else:
            tax_df = parse_taxonomy_mmseqs2(taxonomy_path)
        tax_df, absent_from_taxonomy = build_taxonomy_df(tax_df, list(records.keys()))
        save_parquet(tax_df, ckpt_dir, "tax_df")
        save_json(absent_from_taxonomy, ckpt_dir, "absent_from_taxonomy")

    if cached("ani_df", "parquet"):
        ani_df = load_parquet(ckpt_dir, "ani_df")
        log.info("Loaded ANI from checkpoint")
    else:
        ani_df = parse_ani(ani_path, min_af=args.min_af)
        save_parquet(ani_df, ckpt_dir, "ani_df")

    if cached("depth_df", "parquet"):
        depth_df = load_parquet(ckpt_dir, "depth_df")
        log.info("Loaded depth from checkpoint")
    else:
        depth_df = parse_depth(depth_path)
        save_parquet(depth_df, ckpt_dir, "depth_df")

    if cached("tnf", "parquet"):
        tnf = load_tnf(ckpt_dir)
        log.info(f"Loaded TNF from checkpoint ({len(tnf):,} contigs)")
    else:
        tnf = compute_tnf(records)
        save_tnf(tnf, ckpt_dir)

    # Protein similarity (optional — skip entirely with --skip-prot-sim)
    prot_sim_df: pd.DataFrame | None = None
    if not args.skip_prot_sim:
        if cached("prot_sim_df", "parquet"):
            prot_sim_df = load_parquet(ckpt_dir, "prot_sim_df")
            log.info(f"Loaded protein similarity from checkpoint ({len(prot_sim_df):,} rows)")
        else:
            prot_sim_path = args.prot_sim or run_mmseqs_protein_similarity(
                args.fasta, args.outdir, args.threads,
            )
            prot_sim_df = parse_protein_similarity(prot_sim_path, min_prot_sim=args.min_prot_sim)
            save_parquet(prot_sim_df, ckpt_dir, "prot_sim_df")
    else:
        log.info("Protein similarity disabled (--skip-prot-sim)")

    n_samples = depth_df.shape[1]
    ani_override = args.ani_threshold if args.ani_threshold != ANI_THRESHOLD else None
    thresholds = get_adaptive_thresholds(
        n_samples=n_samples,
        ani_override=ani_override,
        cov_override=args.cov_threshold,
        coverage_as_tiebreaker=args.coverage_as_tiebreaker,
        tnf_main_override=args.tnf_gate_main if args.tnf_gate_main != TNF_GATE_MAIN else None,
    )

    all_ckpts = ("clusters", "membership", "cluster_reps", "unassigned")
    if all(cached(n, "json") for n in all_ckpts):
        clusters_raw = load_json(ckpt_dir, "clusters")
        clusters = {cid: set(m) for cid, m in clusters_raw.items()}
        membership = load_json(ckpt_dir, "membership")
        cluster_reps = load_json(ckpt_dir, "cluster_reps")
        unassigned = load_json(ckpt_dir, "unassigned")
        log.info("Loaded all clustering outputs from checkpoint")
    else:
        contigs_main = []
        contigs_secondary = []
        contigs_recover = []

        for c in records:
            row = tax_df.loc[c]
            long_enough = len(records[c].seq) >= args.min_len
            if long_enough and row["rank"] in MAIN_RANKS:
                contigs_main.append(c)
            elif long_enough and row["rank"] in SECONDARY_RANKS:
                contigs_secondary.append(c)
            else:
                contigs_recover.append(c)

        log.info(
            f"Partitioned: main={len(contigs_main):,} "
            f"secondary={len(contigs_secondary):,} "
            f"recover={len(contigs_recover):,}"
        )

        clusters, membership = build_main_clusters(
            contigs_main, tax_df, ani_df, depth_df, tnf, thresholds,
            leiden_res=args.leiden_res_main,
            prot_sim_df=prot_sim_df,
        )
        cluster_id_offset = len(clusters)
        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        clusters, membership, contigs_recover_extra, cluster_id_offset = build_secondary_clusters(
            contigs_secondary, clusters, membership,
            tax_df, ani_df, depth_df, tnf,
            cluster_id_offset, thresholds,
            leiden_res=args.leiden_res_secondary,
            prot_sim_df=prot_sim_df,
        )
        contigs_recover = contigs_recover + contigs_recover_extra

        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        clusters, membership, unassigned = recover_contigs(
            contigs_recover, clusters, membership, cluster_reps,
            tax_df, ani_df, depth_df,
            set(absent_from_taxonomy), thresholds, tnf,
            prot_sim_df=prot_sim_df,
        )

        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        # Tier 4 — de-novo clustering of remaining unassigned contigs
        cluster_id_offset = len(clusters)
        clusters, membership, cluster_reps, unassigned, cluster_id_offset = cluster_unassigned(
            unassigned, clusters, membership, cluster_reps,
            records, tax_df, ani_df, depth_df, tnf, thresholds,
            cluster_id_offset=cluster_id_offset,
            leiden_res=args.leiden_res_t4,
            prot_sim_df=prot_sim_df,
        )

        # Tier 5 — coverage-only last-resort recovery
        if not args.skip_cov_recovery:
            clusters, membership, unassigned = coverage_only_recovery(
                unassigned, clusters, membership, cluster_reps,
                records, depth_df, tnf,
                cov_r_min=args.cov_recovery_r,
                tnf_min=args.cov_recovery_tnf_min,
            )

        save_json({cid: list(m) for cid, m in clusters.items()}, ckpt_dir, "clusters")
        save_json(membership, ckpt_dir, "membership")
        save_json(cluster_reps, ckpt_dir, "cluster_reps")
        save_json(unassigned, ckpt_dir, "unassigned")

    write_outputs(
        clusters, cluster_reps, unassigned,
        records, tax_df, depth_df, args.outdir,
        min_checkm2_bp=args.min_checkm2_bp,
    )

    checkm2_df = pd.DataFrame()
    if not args.skip_checkm2:
        if args.checkm2_db:
            os.environ["CHECKM2DB"] = args.checkm2_db
        checkm2_df = run_checkm2(
            os.path.join(args.outdir, "checkm2_bin_list.txt"),
            args.outdir, args.threads,
        )

    else:
        log.info("Skipping CheckM2 (--skip-checkm2)")

    drep_out = None
    if not args.skip_drep:
        if args.skip_checkm2:
            log.warning("Skipping dRep because CheckM2 was skipped")
        else:
            drep_out = run_drep(
                os.path.join(args.outdir, "clusters"),
                checkm2_df, args.outdir, args.threads,
            )
    else:
        log.info("Skipping dRep (--skip-drep)")

    log.info("=" * 60)
    log.info(f"Total clusters:          {len(clusters):,}")
    log.info(f"Total assigned contigs:  {sum(len(v) for v in clusters.values()):,}")
    log.info(f"Unassigned contigs:      {len(unassigned):,}")
    if not args.skip_cov_recovery:
        log.info("  (after tier-5 coverage-only recovery)")
    if not checkm2_df.empty:
        log.info(f"CheckM2 quality:         {checkm2_df['quality'].value_counts().to_dict()}")
    if drep_out:
        log.info(f"Dereplicated bins:       {drep_out}")
    log.info(f"Output:                  {args.outdir}")
    log.info("Done.")


# Import everything needed from other modules
from tritanc.config import (
    ANI_THRESHOLD, TNF_GATE_MAIN, MMSEQS_THREADS, MIN_LEN,
    MAIN_RANKS, SECONDARY_RANKS, ANI_MIN_AF,
    LEIDEN_RES_MAIN, LEIDEN_RES_SECONDARY, LEIDEN_RES_T4,
)
from tritanc.tools import check_tools, tools_needed_for, run_mmseqs_taxonomy, run_skani, run_depth, run_mmseqs_protein_similarity
from tritanc.parsers import (
    parse_fasta, parse_taxonomy_mmseqs2, parse_taxonomy_taxometer,
    build_taxonomy_df, parse_protein_similarity, parse_depth, parse_ani,
)
from tritanc.checkpoint import (
    save_parquet, load_parquet, save_json, load_json, save_tnf, load_tnf, _ckpt_exists,
)
from tritanc.signals import compute_tnf, representative
from tritanc.config import get_adaptive_thresholds
from tritanc.tiers import (
    build_main_clusters, build_secondary_clusters, cluster_unassigned, coverage_only_recovery,
)
from tritanc.recovery import recover_contigs
from tritanc.output import write_outputs, run_checkm2, run_drep

__all__ = ["main"]
