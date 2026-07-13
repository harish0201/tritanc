"""External tool runners: mmseqs, skani, samtools, pyrodigal, etc."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tritanc.config import TOOL_HINTS

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Tool checks
# ═════════════════════════════════════════════════════════════════════════════

from tritanc.config import TOOL_HINTS


def check_tools(needed: list[str]) -> None:
    missing = [t for t in needed if shutil.which(t) is None]
    if missing:
        lines = ["The following required tools were not found on PATH:"]
        for t in missing:
            lines.append(f"  {t:<35} {TOOL_HINTS.get(t, '')}")
        log.error("\n".join(lines))
        sys.exit(1)
    if needed:
        log.info(f"Tool check passed: {', '.join(needed)}")


def tools_needed_for(args) -> list[str]:
    needed = []
    if not args.taxonomy:
        needed.append("mmseqs")
    if not args.ani:
        needed.append("skani")
    if not args.depth:
        needed += ["jgi_summarize_bam_contig_depths", "samtools"]
    if not args.prot_sim and not args.skip_prot_sim:
        needed += ["pyrodigal", "mmseqs"]
    if not args.skip_checkm2:
        needed.append("checkm2")
    if not args.skip_drep:
        needed.append("dRep")
    # Deduplicate while preserving order
    seen: set = set()
    return [x for x in needed if not (x in seen or seen.add(x))]


# ═════════════════════════════════════════════════════════════════════════════
# External tool runners
# ═════════════════════════════════════════════════════════════════════════════

def run(cmd: list[str], desc: str) -> None:
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        log.error(f"{desc} failed (exit {result.returncode})")
        sys.exit(result.returncode)


def run_mmseqs_taxonomy(fasta: str, db: str, outdir: str, threads: int) -> str:
    tax_dir = os.path.join(outdir, "taxonomy")
    os.makedirs(tax_dir, exist_ok=True)
    fasta_db = os.path.join(tax_dir, "assembly_db")
    result_db = os.path.join(tax_dir, "taxonomy_result")
    tmp_dir = os.path.join(tax_dir, "tmp")
    tsv_out = os.path.join(tax_dir, "taxonomy.tsv")
    if os.path.exists(tsv_out):
        log.info(f"MMseqs2 TSV already exists, skipping: {tsv_out}")
        return tsv_out
    run(["mmseqs", "createdb", fasta, fasta_db], "mmseqs createdb")
    run([
        "mmseqs", "taxonomy", fasta_db, db, result_db, tmp_dir,
        "--tax-lineage", str(MMSEQS_TAX_LINEAGE),
        "--sensitivity", str(MMSEQS_SENS),
        "--threads", str(threads),
    ], "mmseqs taxonomy")
    run(["mmseqs", "createtsv", fasta_db, result_db, tsv_out], "mmseqs createtsv")
    return tsv_out


def run_skani(fasta: str, outdir: str, threads: int) -> str:
    ani_dir = os.path.join(outdir, "ani")
    os.makedirs(ani_dir, exist_ok=True)
    tsv_out = os.path.join(ani_dir, "skani_triangle.tsv")
    if os.path.exists(tsv_out):
        log.info(f"skani output already exists, skipping: {tsv_out}")
        return tsv_out
    run([
        "skani", "triangle", "-i", fasta, "-o", tsv_out,
        "--sparse", "-c", "20", "-m", "50", "--robust", "-s", "80",
        "-t", str(threads),
    ], "skani triangle")
    return tsv_out


def run_depth(bams: list[str], outdir: str) -> str:
    depth_dir = os.path.join(outdir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    tsv_out = os.path.join(depth_dir, "depth_matrix.txt")
    if os.path.exists(tsv_out):
        log.info(f"Depth matrix already exists, skipping: {tsv_out}")
        return tsv_out
    for bam in bams:
        if not os.path.exists(bam + ".bai"):
            run(["samtools", "index", bam], "samtools index")
    run(
        ["jgi_summarize_bam_contig_depths", "--outputDepth", tsv_out] + bams,
        "jgi_summarize_bam_contig_depths",
    )
    return tsv_out


def run_mmseqs_protein_similarity(
    fasta: str,
    outdir: str,
    threads: int,
    min_seq_id: float = 0.50,
    min_aln_cov: float = 0.50,
    sensitivity: float = 5.7,
) -> str:
    """Predict ORFs with Prodigal then run MMseqs2 all-vs-all protein search.

    Returns path to a TSV with columns: query_contig, ref_contig, prot_sim
    where prot_sim is the mean amino-acid sequence identity across all
    significant protein hits between the two contigs, scaled 0–100.

    min_seq_id   : minimum amino-acid identity to retain a hit (default 50%)
    min_aln_cov  : minimum alignment coverage on query protein (default 50%)
    sensitivity  : MMseqs2 -s value; 5.7 = sensitive, 7.5 = very sensitive
    """
    prot_dir = os.path.join(outdir, "protein_similarity")
    os.makedirs(prot_dir, exist_ok=True)
    tsv_out = os.path.join(prot_dir, "prot_sim.tsv")

    if os.path.exists(tsv_out):
        log.info(f"Protein similarity TSV already exists, skipping: {tsv_out}")
        return tsv_out

    # ── Step 1: predict ORFs with Prodigal (metagenome mode) ─────────────────
    proteins_faa = os.path.join(prot_dir, "proteins.faa")
    gff_out      = os.path.join(prot_dir, "proteins.gff")
    if not os.path.exists(proteins_faa):
        log.info("Predicting ORFs with Prodigal (-p meta)...")
        run([
            "pyrodigal", "-i", fasta, "-a", proteins_faa,
            "-f", "gff", "-o", gff_out, "-p", "meta", "-j", str(threads),
        ], "pyrodigal")
    else:
        log.info(f"Prodigal output already exists, skipping: {proteins_faa}")

    # Strip stop-codon asterisks that break MMseqs2 createdb
    clean_faa = os.path.join(prot_dir, "proteins_clean.faa")
    if not os.path.exists(clean_faa):
        log.info("Cleaning Prodigal FASTA (stripping stop codons)...")
        with open(proteins_faa) as fin, open(clean_faa, "w") as fout:
            for line in fin:
                fout.write(line.rstrip("*\n").rstrip() + "\n" if not line.startswith(">") else line)

    # ── Step 2: MMseqs2 all-vs-all protein search ─────────────────────────────
    prot_db   = os.path.join(prot_dir, "prot_db")
    result_db = os.path.join(prot_dir, "result_db")
    hits_tsv  = os.path.join(prot_dir, "hits.tsv")
    tmp_dir   = os.path.join(prot_dir, "tmp")

    if not os.path.exists(prot_db + ".index"):
        run(["mmseqs", "createdb", clean_faa, prot_db], "mmseqs createdb (proteins)")

    if not os.path.exists(result_db + ".index"):
        run([
            "mmseqs", "search",
            prot_db, prot_db, result_db, tmp_dir,
            "--min-seq-id", str(min_seq_id),
            "-c",           str(min_aln_cov),
            "--cov-mode",   "0",          # coverage on both query and target
            "-s",           str(sensitivity),
            "--threads",    str(threads),
            "-e",           "1e-5",
        ], "mmseqs search (proteins)")

    if not os.path.exists(hits_tsv):
        run([
            "mmseqs", "convertalis",
            prot_db, prot_db, result_db, hits_tsv,
            "--format-output", "query,target,fident,alnlen,qlen,tlen",
        ], "mmseqs convertalis")

    # ── Step 3: aggregate protein hits → contig-level similarity ─────────────
    log.info("Aggregating protein hits to contig-level similarity...")
    hits = pd.read_csv(
        hits_tsv, sep="\t", header=None,
        names=["query_prot", "ref_prot", "fident", "alnlen", "qlen", "tlen"],
    )

    def _contig_id(prot_id: str) -> str:
        # Prodigal appends _N for the Nth ORF on a contig — strip it.
        return prot_id.rsplit("_", 1)[0]

    hits["query_contig"] = hits["query_prot"].map(_contig_id)
    hits["ref_contig"]   = hits["ref_prot"].map(_contig_id)

    # Drop self-hits (same contig)
    hits = hits[hits["query_contig"] != hits["ref_contig"]].copy()

    if hits.empty:
        log.warning("No cross-contig protein hits found — writing empty prot_sim.tsv")
        pd.DataFrame(columns=["query", "ref", "prot_sim"]).to_csv(tsv_out, sep="\t", index=False)
        return tsv_out

    # Normalise to canonical pair order so (A,B) and (B,A) are the same row
    hits["q_norm"] = hits[["query_contig", "ref_contig"]].min(axis=1)
    hits["r_norm"] = hits[["query_contig", "ref_contig"]].max(axis=1)

    # Mean fident per contig pair — weight by alignment length
    hits["weighted"] = hits["fident"] * hits["alnlen"]
    agg = (
        hits.groupby(["q_norm", "r_norm"])
        .agg(total_weighted=("weighted", "sum"), total_alnlen=("alnlen", "sum"))
        .reset_index()
    )
    agg["prot_sim"] = (agg["total_weighted"] / agg["total_alnlen"]) * 100.0

    out = agg[["q_norm", "r_norm", "prot_sim"]].rename(
        columns={"q_norm": "query", "r_norm": "ref"}
    )
    out.to_csv(tsv_out, sep="\t", index=False)
    log.info(f"Protein similarity: {len(out):,} contig pairs written to {tsv_out}")
    return tsv_out


# Import constants from config
from tritanc.config import MMSEQS_THREADS, MMSEQS_TAX_LINEAGE, MMSEQS_SENS

__all__ = [
    "check_tools",
    "tools_needed_for",
    "run",
    "run_mmseqs_taxonomy",
    "run_skani",
    "run_depth",
    "run_mmseqs_protein_similarity",
]
