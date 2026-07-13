"""Output writing: FASTA bins, summary tables, CheckM2, and dRep integration."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
from Bio import SeqIO

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(
    clusters: dict,
    cluster_reps: dict,
    unassigned: list,
    records: dict,
    tax_df,
    depth_df,
    outdir: str,
    min_checkm2_bp: int = 100_000,
) -> None:
    """Write cluster bins, representatives, unassigned FASTA, and summary TSV."""
    bins_dir = os.path.join("clusters", "bins")
    reps_dir = os.path.join("clusters", "representatives")
    unassigned_dir = os.path.join("unassigned")
    os.makedirs(os.path.join(outdir, bins_dir), exist_ok=True)
    os.makedirs(os.path.join(outdir, reps_dir), exist_ok=True)
    os.makedirs(os.path.join(outdir, unassigned_dir), exist_ok=True)

    summary_rows: list = []
    eligible_bins = 0

    for cid, members in clusters.items():
        rep = cluster_reps[cid]
        SeqIO.write(
            [records[rep]] if rep in records else [],
            os.path.join(os.path.join(outdir, reps_dir), f"{cid}_representative.fasta"),
            "fasta",
        )
        SeqIO.write(
            [records[c] for c in members if c in records],
            os.path.join(os.path.join(outdir, bins_dir), f"{cid}.fasta"),
            "fasta",
        )
        rep_row = tax_df.loc[rep] if rep in tax_df.index else None
        total_bp = sum(len(records[c].seq) for c in members if c in records)
        for c in members:
            c_row = tax_df.loc[c] if c in tax_df.index else None
            summary_rows.append({
                "contig": c,
                "cluster": cid,
                "is_rep": c == rep,
                "contig_len": len(records[c].seq) if c in records else None,
                "cluster_bp": total_bp,
                "mean_depth": float(depth_df.loc[c].mean()) if c in depth_df.index else None,
                "rank": c_row["rank"] if c_row is not None else None,
                "name": c_row["name"] if c_row is not None else None,
                "lineage": ";".join(c_row["lineage"]) if c_row is not None else None,
                "rep_name": rep_row["name"] if rep_row is not None else None,
                "rep_rank": rep_row["rank"] if rep_row is not None else None,
                "rep_lineage": ";".join(rep_row["lineage"]) if rep_row is not None else None,
            })

    SeqIO.write(
        [records[c] for c in unassigned if c in records],
        os.path.join(outdir, unassigned_dir, "unassigned_contigs.fasta"),
        "fasta",
    )
    for c in unassigned:
        c_row = tax_df.loc[c] if c in tax_df.index else None
        summary_rows.append({
            "contig": c, "cluster": "unassigned", "is_rep": False,
            "contig_len": len(records[c].seq) if c in records else None,
            "cluster_bp": None,
            "mean_depth": float(depth_df.loc[c].mean()) if c in depth_df.index else None,
            "rank": c_row["rank"] if c_row is not None else None,
            "name": c_row["name"] if c_row is not None else None,
            "lineage": ";".join(c_row["lineage"]) if c_row is not None else None,
            "rep_name": None, "rep_rank": None, "rep_lineage": None,
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(outdir, "cluster_summary.tsv"), sep="\t", index=False
    )
    log.info("Wrote cluster_summary.tsv")

    with open(os.path.join(outdir, "checkm2_bin_list.txt"), "w") as fh:
        for cid, members in clusters.items():
            total_bp = sum(len(records[c].seq) for c in members if c in records)
            if total_bp >= min_checkm2_bp:
                fh.write(os.path.join(outdir,bins_dir, f"{cid}.fasta") + "\n")
                eligible_bins += 1
    log.info(f"Wrote checkm2_bin_list.txt with {eligible_bins:,} bins >= {min_checkm2_bp:,} bp")

    thresh_dir = os.path.join("clusters", "thresh_bins")
    os.makedirs(os.path.join(outdir, thresh_dir), exist_ok=True)

    with open(os.path.join(outdir, "checkm2_bin_list.txt")) as fh:
        bin_paths = [line.strip() for line in fh if line.strip()]

    linked = 0
    for src in bin_paths:
        if not os.path.exists(src):
            log.warning(f"Bin FASTA not found, skipping symlink: {src}")
            continue
        dst = os.path.join(os.path.join(outdir, thresh_dir), os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        linked += 1

    log.info(f"thresh_bins/: {linked:,} symlinks created in {thresh_dir}")
    return thresh_dir


# ═════════════════════════════════════════════════════════════════════════════
# CheckM2 + dRep
# ═════════════════════════════════════════════════════════════════════════════

from tritanc.config import CHECKM2_MIN_COMPLETENESS, CHECKM2_MAX_CONTAMINATION, DREP_ANI


def run_checkm2(bin_list_path: str, outdir: str, threads: int) -> pd.DataFrame:
    """Run CheckM2 quality assessment on cluster bins.

    Returns a DataFrame with completeness, contamination, and quality tier
    for each bin. Skips if no bins pass the minimum size threshold.
    """
    checkm2_dir = os.path.join(outdir, "checkm2")
    os.makedirs(checkm2_dir, exist_ok=True)
    quality_tsv = os.path.join(checkm2_dir, "quality_report.tsv")

    if not os.path.exists(quality_tsv):
        with open(bin_list_path) as fh:
            bin_paths = [line.strip() for line in fh if line.strip()]
        if not bin_paths:
            log.warning("No bins in checkm2_bin_list.txt — skipping CheckM2")
            return pd.DataFrame()
        thresh_dir = os.path.join(outdir,"clusters", "thresh_bins")
        run([
            "checkm2", "predict",
            "--input", thresh_dir,
            "--output-directory", checkm2_dir,
            "--extension", "fasta",
            "--threads", str(threads),
            "--force", "--allmodels",
        ], "checkm2 predict")

    if not os.path.exists(quality_tsv):
        log.error(f"CheckM2 quality_report.tsv not found: {quality_tsv}")
        return pd.DataFrame()

    qdf = pd.read_csv(quality_tsv, sep="\t", low_memory=False)

    def quality_tier(row) -> str:
        c, x = row["Completeness_General"], row["Contamination"]
        if c >= 90 and x <= 5:
            return "high"
        if c >= CHECKM2_MIN_COMPLETENESS and x <= CHECKM2_MAX_CONTAMINATION:
            return "medium"
        return "low"

    qdf["quality"] = qdf.apply(quality_tier, axis=1)
    qdf["cluster"] = qdf["Name"].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0].removesuffix("_bin")
    )
    log.info(f"CheckM2 done: {qdf['quality'].value_counts().to_dict()}")
    return qdf


def run_drep(
    clusters_dir: str,
    checkm2_df: pd.DataFrame,
    outdir: str,
    threads: int,
) -> str | None:
    """Run dRep dereplication on CheckM2-passing bins.

    Returns path to dereplicated genomes directory, or None if skipped.
    """
    drep_dir = os.path.join(outdir, "drep")
    derep_dir = os.path.join(drep_dir, "dereplicated_genomes")

    if os.path.exists(derep_dir) and os.listdir(derep_dir):
        log.info(f"dRep output already exists, skipping: {derep_dir}")
        return derep_dir

    # If caller passed an empty DataFrame (e.g. resuming after --skip-checkm2 was
    # used in a previous run), try to reload the quality report from disk.
    if checkm2_df.empty:
        quality_tsv = os.path.join(outdir, "checkm2", "quality_report.tsv")
        if os.path.exists(quality_tsv):
            log.info("checkm2_df is empty but quality_report.tsv found — reloading for dRep")
            checkm2_df = pd.read_csv(quality_tsv, sep="\t")
            checkm2_df["cluster"] = checkm2_df["Name"].apply(
                lambda x: os.path.splitext(str(x))[0]
            )

            def quality_tier(row) -> str:
                c, x = row["Completeness_General"], row["Contamination"]
                if c >= 90 and x <= 5:
                    return "high"
                if c >= CHECKM2_MIN_COMPLETENESS and x <= CHECKM2_MAX_CONTAMINATION:
                    return "medium"
                return "low"

            checkm2_df["quality"] = checkm2_df.apply(quality_tier, axis=1)
        else:
            log.warning("No CheckM2 results — skipping dRep")
            return None

    passing = checkm2_df[checkm2_df["quality"].isin(["high", "medium"])].copy()
    if passing.empty:
        log.warning("No bins pass CheckM2 thresholds — skipping dRep")
        return None

    os.makedirs(drep_dir, exist_ok=True)
    genome_info_path = os.path.join(drep_dir, "genome_info.csv")
    bin_paths: list = []
    rows: list = []

    for _, row in passing.iterrows():
        cid = row["cluster"]
        bin_path = os.path.join(clusters_dir, "bins", cid + ".fasta")
        if not os.path.exists(bin_path):
            continue
        bin_paths.append(bin_path)
        rows.append({
            "genome": bin_path,
            "completeness": row["Completeness_General"],
            "contamination": row["Contamination"],
            "strain_heterogeneity": 0,
        })

    if not rows:
        log.warning("No dRep-eligible bin files found")
        return None

    pd.DataFrame(rows).to_csv(genome_info_path, index=False)
    run([
        "dRep", "dereplicate", drep_dir,
        "-g", *bin_paths,
        "--genomeInfo", genome_info_path,
        "-pa", "0.9",
        "-sa", str(DREP_ANI / 100.0),
        "-nc", "0.30",
        "-comp", str(CHECKM2_MIN_COMPLETENESS),
        "-con", str(CHECKM2_MAX_CONTAMINATION),
        "-p", str(threads),
    ], "dRep dereplicate")
    return derep_dir if os.path.exists(derep_dir) else None


# Import run from tools
from tritanc.tools import run

__all__ = [
    "write_outputs",
    "run_checkm2",
    "run_drep",
]
