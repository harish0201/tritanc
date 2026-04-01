#!/usr/bin/env python3
"""
Taxonomy-aware metagenomic contig clustering pipeline — v10.

USAGE — skip individual steps by supplying pre-computed files:
  python tritanc_v10.py \
    --fasta assembly.fasta \
    --taxonomy saliva_tax.tsv \
    --taxonomy-format taxometer \
    --ani skani.tsv \
    --depth depth_matrix.txt \
    --outdir results/

Changes in v10 (multimodal signal prioritisation):

  Tiers 1 and 2 are now fully multimodal via _build_multimodal_graph():
    Previously: ANI-only graph; empty if skani is sparse → near-all singletons.
    Now: three edge sources in priority order —
      1. ANI edges (skani pairs passing threshold)
      2. Protein-similarity edges (where ANI has no hit)
      3. Taxonomy edges (same-taxon pairs where neither ANI nor protein found)
    Short or divergent contigs that skani cannot pair can now gain graph
    support from co-abundance + composition alone (taxonomy / protein edges).

  Hybrid gating replaces the old hard/soft coverage switch in Tiers 1/2:
    Edge is added if:
      (ANI passes AND at least one of cov/TNF passes), OR
      (cov AND TNF both pass — for divergent but co-abundant contigs)
    Controlled by --tnf-gate-main (default 0.85).

  Rebalanced edge weights throughout (less ANI-dominant):
    ANI edge      : 0.40 * ani  + 0.35 * cov_r + 0.25 * tnf
    Protein edge  : 0.30 * prot + 0.40 * cov_r + 0.30 * tnf
    Taxonomy edge : 0.00 * seq  + 0.50 * cov_r + 0.50 * tnf
    Tier 3 scoring updated to match (was 0.55/0.30/0.15).

  Tier 3 uses cluster centroids (not single representatives) for coverage
    correlation, consistent with Tier 5. One representative can be a poor
    proxy for a heterogeneous bin; the centroid is more stable.

  Taxonomy O(n²) pair generation capped at 50,000 pairs per taxon group
    to prevent stalls on large genus groups (e.g. 5000 Streptococcus contigs).

  TNF uses canonical RC-collapsed 4-mers; short contigs (<1000 bp) are TNF-neutral
  BH-FDR correction applied when n >= 20 samples
  Recovery uses lineage-aware taxonomy fallback (not leaf-only)
  Checkpointing covers all parsed inputs and cluster results
  Tier 5 rho matrix is chunked to bound peak RAM
  Taxonomy + ANI candidates evaluated jointly in Tier 3 (both-paths fix)
  Minimum score threshold (0.08) applied before taxonomy-only assignments
  Leiden community detection at all tiers (leidenalg + python-igraph)
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from Bio import SeqIO
from scipy.stats import rankdata
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests


# ── Constants ────────────────────────────────────────────────────────────────
MIN_LEN = 2000          # bp — min contig length for main/secondary clustering
TNF_MIN_LEN = 1000      # bp — contigs shorter than this skip TNF (too noisy)
ANI_MIN_AF = 0.0        # alignment fraction filter default (disabled)
ANI_THRESHOLD = 95.0    # CLI default; overridden adaptively at runtime
COV_THRESHOLD = 0.90    # CLI default; overridden adaptively at runtime
COV_PVAL = 0.05

NOISE_TAXA = {"", "root", "cellular organisms", "unclassified", "N/A", "NA"}
CANONICAL_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
GENUS_IDX = 5
MAIN_RANKS = {"genus", "species"}
SECONDARY_RANKS = {"order", "family", "class", "phylum"}

CHECKM2_MIN_COMPLETENESS = 50.0
CHECKM2_MAX_CONTAMINATION = 10.0
DREP_ANI = 95.0

MMSEQS_THREADS = 8
MMSEQS_TAX_LINEAGE = 1
MMSEQS_SENS = 4

MAX_TAX_CANDIDATES = 50

LEIDEN_RES_MAIN = 3.5       # finer splits for closely related oral taxa
LEIDEN_RES_SECONDARY = 2.0  # coarser; appropriate for above-genus groups
LEIDEN_RES_T4 = 1.5 # recovery at Tier 4 resolution

TNF_GATE_MAIN = 0.93        # minimum TNF cosine similarity for hybrid gate in Tiers 1/2

TAXOMETER_PREFIX = {
    "d": "domain", "p": "phylum", "c": "class",
    "o": "order",  "f": "family", "g": "genus", "s": "species",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive thresholds
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveThresholds:
    n_samples: int
    ani_main: float
    ani_secondary: float
    ani_recovery: float
    cov_main: float
    cov_secondary: float
    cov_recovery: float | None
    cov_pval: float
    use_fdr: bool
    use_permutation: bool
    n_permutations: int
    coverage_is_hard_gate: bool
    tnf_main: float           # minimum TNF cosine similarity for hybrid gate


def get_adaptive_thresholds(
    n_samples: int,
    ani_override: float | None = None,
    cov_override: float | None = None,
    coverage_as_tiebreaker: bool = False,
    tnf_main_override: float | None = None,
) -> AdaptiveThresholds:
    if n_samples >= 30:
        ani = 95.0
        cov_main, cov_secondary, cov_recovery = 0.90, 0.82, 0.75
        cov_pval = 0.05
        use_fdr = True
        use_permutation = False
        n_permutations = 0
        coverage_is_hard_gate = True
    elif n_samples >= 20:
        ani = 95.0
        cov_main, cov_secondary, cov_recovery = 0.85, 0.78, 0.70
        cov_pval = 0.05
        use_fdr = True
        use_permutation = True
        n_permutations = 999
        coverage_is_hard_gate = True
    elif n_samples >= 10:
        ani = 96.0
        cov_main, cov_secondary, cov_recovery = 0.75, 0.65, 0.60
        cov_pval = 0.10
        use_fdr = False
        use_permutation = True
        n_permutations = 9999
        coverage_is_hard_gate = True
    else:
        ani = 97.0
        cov_main, cov_secondary = 0.60, 0.50
        cov_recovery = None
        cov_pval = 0.10
        use_fdr = False
        use_permutation = False
        n_permutations = 0
        coverage_is_hard_gate = False

    log.info(
        f"Adaptive thresholds for n={n_samples} samples: ANI={ani}% | "
        f"cov_main={cov_main} cov_secondary={cov_secondary} cov_recovery={cov_recovery} | "
        f"use_fdr={use_fdr} use_permutation={use_permutation} hard_gate={coverage_is_hard_gate}"
    )

    ani_main = ani_secondary = ani_recovery = ani

    if ani_override is not None:
        ani_main = ani_secondary = ani_recovery = ani_override
        log.info(f"ANI overridden by --ani-threshold: {ani_override}%")

    if cov_override is not None:
        cov_main = cov_secondary = cov_override
        cov_recovery = cov_override
        log.info(f"Coverage threshold overridden by --cov-threshold: {cov_override}")

    if coverage_as_tiebreaker:
        coverage_is_hard_gate = False
        log.info("Coverage-as-tiebreaker: hard coverage gate disabled")

    tnf_main = tnf_main_override if tnf_main_override is not None else TNF_GATE_MAIN
    log.info(f"TNF gate (hybrid gating in Tiers 1/2): tnf_main={tnf_main}")

    return AdaptiveThresholds(
        n_samples=n_samples,
        ani_main=ani_main,
        ani_secondary=ani_secondary,
        ani_recovery=ani_recovery,
        cov_main=cov_main,
        cov_secondary=cov_secondary,
        cov_recovery=cov_recovery,
        cov_pval=cov_pval,
        use_fdr=use_fdr,
        use_permutation=use_permutation,
        n_permutations=n_permutations,
        coverage_is_hard_gate=coverage_is_hard_gate,
        tnf_main=tnf_main,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tool checks
# ═════════════════════════════════════════════════════════════════════════════

TOOL_HINTS = {
    "mmseqs": "https://github.com/soedinglab/MMseqs2",
    "skani": "https://github.com/bluenote-1577/skani",
    "pyrodigal": "conda install -c bioconda pyrodigal",
    "jgi_summarize_bam_contig_depths": "part of MetaBAT2 — conda install -c bioconda metabat2",
    "samtools": "https://www.htslib.org",
    "checkm2": "https://github.com/chklovski/CheckM2",
    "dRep": "https://github.com/MrOlm/drep",
    "taxometer": "part of VAMB - https://github.com/RasmussenLab/vamb"
}


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
    # Prodigal header format: >contig_id_N  (last underscore-separated field is
    # the ORF index).  We strip it to recover the source contig ID.
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


# ═════════════════════════════════════════════════════════════════════════════
# Parsers
# ═════════════════════════════════════════════════════════════════════════════

def parse_fasta(path: str) -> dict:
    log.info(f"Loading assembly: {path}")
    records = {r.id: r for r in SeqIO.parse(path, "fasta")}
    log.info(f"{len(records):,} contigs")
    return records


def _empty_tax_row(contig: str) -> dict:
    return {"contig": contig, "taxid": None, "rank": "unclassified",
            "name": "unclassified", "lineage": [], "scores": []}


def parse_taxonomy_mmseqs2(path: str) -> pd.DataFrame:
    log.info(f"Parsing MMseqs2 taxonomy: {path}")
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            contig = parts[0]
            rank = parts[2].strip()
            name = parts[3].strip()
            lineage_str = parts[8].strip() if len(parts) > 8 else ""
            lineage = []
            for lvl in lineage_str.split(";"):
                lvl = lvl.strip()
                if not lvl:
                    continue
                if len(lvl) >= 2 and lvl[1] == "_":
                    lvl = lvl[2:]
                lineage.append(lvl)
            rows.append({
                "contig": contig,
                "taxid": parts[1].strip() if len(parts) > 1 else None,
                "rank": rank or "unclassified",
                "name": name or "unclassified",
                "lineage": lineage,
                "scores": [],
            })
    df = pd.DataFrame(rows).set_index("contig")
    log.info(f"{len(df):,} MMseqs2 assignments loaded")
    return df


def _strip_rank_prefix(s: str) -> str:
    if len(s) >= 3 and s[1] == "_" and s[0].lower() in TAXOMETER_PREFIX:
        return s[2:]
    return s


def parse_taxonomy_taxometer(path: str, min_score: float = 0.0) -> pd.DataFrame:
    log.info(f"Parsing Taxometer taxonomy: {path} (min_score={min_score})")
    rows = []
    no_tax_count = 0
    with open(path) as fh:
        header = next(fh, "").rstrip()
        if header != "contigs\tpredictions\tscores":
            raise ValueError(f"Unexpected Taxometer header: {repr(header)}")
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            contig = parts[0].strip()
            if len(parts) == 1:
                no_tax_count += 1
                rows.append(_empty_tax_row(contig))
                continue
            if len(parts) != 3:
                raise ValueError(f"Line {lineno}: expected 1 or 3 fields, got {len(parts)}")
            lineage_raw = parts[1].strip().split(";")
            scores_raw = parts[2].strip().split(";")
            scores = []
            for s in scores_raw:
                try:
                    scores.append(float(s))
                except ValueError:
                    scores.append(0.0)
            while len(scores) < len(lineage_raw):
                scores.append(0.0)
            trusted_raw, trusted_scores = [], []
            for lvl, sc in zip(lineage_raw, scores):
                if sc < min_score:
                    break
                trusted_raw.append(lvl.strip())
                trusted_scores.append(sc)
            lineage_names = [_strip_rank_prefix(lvl) for lvl in trusted_raw]
            depth = len(lineage_names)
            if depth == 0:
                rank, name = "unclassified", "unclassified"
            else:
                rank = CANONICAL_RANKS[min(depth - 1, len(CANONICAL_RANKS) - 1)]
                name = lineage_names[-1]
            rows.append({
                "contig": contig, "taxid": None,
                "rank": rank, "name": name,
                "lineage": lineage_names, "scores": trusted_scores,
            })
    df = pd.DataFrame(rows).set_index("contig")
    log.info(f"{len(df):,} Taxometer assignments loaded ({no_tax_count:,} with no taxonomy)")
    return df


def build_taxonomy_df(tax_df: pd.DataFrame, all_contigs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in all_contigs if c not in tax_df.index]
    if missing:
        log.info(f"{len(missing):,} contigs absent from taxonomy -> marked unclassified")
        missing_df = pd.DataFrame([_empty_tax_row(c) for c in missing]).set_index("contig")
        tax_df = pd.concat([tax_df, missing_df])
    return tax_df, missing


def parse_protein_similarity(path: str, min_prot_sim: float = 50.0) -> pd.DataFrame:
    """Load precomputed or pipeline-generated protein similarity TSV.

    Expects columns: query, ref, prot_sim (0–100 scale).
    Filters to pairs >= min_prot_sim and ensures both orientations exist
    so downstream code can look up (A→B) or (B→A) interchangeably.
    """
    log.info(f"Loading protein similarity: {path} (min_prot_sim={min_prot_sim})")
    df = pd.read_csv(path, sep="\t")
    if not {"query", "ref", "prot_sim"}.issubset(df.columns):
        raise ValueError(f"prot_sim TSV missing expected columns. Found: {list(df.columns)}")
    df["prot_sim"] = pd.to_numeric(df["prot_sim"], errors="coerce")
    df = df.dropna(subset=["prot_sim"])
    df = df[df["prot_sim"] >= min_prot_sim].copy()
    # Add reverse orientation so (A,B) can be found as (B,A)
    rev = df.rename(columns={"query": "ref", "ref": "query"})
    df = pd.concat([df, rev], ignore_index=True).drop_duplicates(subset=["query", "ref"])
    log.info(f"{len(df) // 2:,} protein-similar pairs loaded (both orientations stored)")
    return df


def parse_depth(path: str) -> pd.DataFrame:
    log.info(f"Loading depth matrix: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    depth_cols = [
        c for c in df.columns
        if not c.endswith("-var") and c not in ("contigLen", "totalAvgDepth")
    ]
    df = np.log1p(df[depth_cols].copy())
    log.info(f"{len(df):,} contigs x {len(depth_cols)} samples")
    return df.astype(np.float32)


def parse_ani(path: str, min_af: float = 0.0) -> pd.DataFrame:
    log.info(f"Loading ANI results: {path}")
    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    expected = {"Ref_name", "Query_name", "ANI"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"skani output missing expected columns. "
            f"Found: {list(df.columns)}. Expected at least: {sorted(expected)}"
        )
    rename_map: dict[str, str] = {
        "Ref_name": "query", "Query_name": "ref", "ANI": "ani",
    }
    if "Align_fraction_ref" in df.columns:
        rename_map["Align_fraction_ref"] = "qcov"
    if "Align_fraction_query" in df.columns:
        rename_map["Align_fraction_query"] = "rcov"
    df = df.rename(columns=rename_map)
    if "qcov" not in df.columns:
        df["qcov"] = 1.0
    if "rcov" not in df.columns:
        df["rcov"] = 1.0
    df = df[["query", "ref", "ani", "qcov", "rcov"]].copy()
    for col in ("ani", "qcov", "rcov"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ani"]).reset_index(drop=True)
    if min_af > 0.0:
        before = len(df)
        df = df[df[["qcov", "rcov"]].max(axis=1) >= min_af].reset_index(drop=True)
        log.info(f"Filtered {before - len(df):,} ANI pairs below min_af={min_af}")
    log.info(f"{len(df):,} ANI pairs loaded")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — representative, TNF, Spearman
# ═════════════════════════════════════════════════════════════════════════════

def representative(contigs: set, records: dict, depth_df: pd.DataFrame) -> str:
    def score(c: str) -> tuple:
        length = len(records[c].seq) if c in records else 0
        depth = float(depth_df.loc[c].mean()) if c in depth_df.index else 0.0
        return (length, depth)
    return max(contigs, key=score)


def _revcomp(kmer: str) -> str:
    return kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _canonical(kmer: str) -> str:
    rc = _revcomp(kmer)
    return kmer if kmer <= rc else rc


def compute_tnf(records: dict, min_len: int = TNF_MIN_LEN) -> dict:
    log.info(f"Computing TNF (canonical RC-collapsed 4-mers; min_len={min_len} bp)...")
    k = 4
    raw_kmers = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    canonical = sorted({_canonical(km) for km in raw_kmers})
    canon_idx = {km: i for i, km in enumerate(canonical)}
    tnf: dict[str, np.ndarray] = {}
    skipped = 0
    for cid, rec in records.items():
        if len(rec.seq) < min_len:
            skipped += 1
            continue
        seq = str(rec.seq).upper()
        counts = np.zeros(len(canonical), dtype=np.float32)
        n_valid = 0
        for i in range(len(seq) - k + 1):
            km = seq[i:i + k]
            if set(km) <= {"A", "C", "G", "T"}:
                counts[canon_idx[_canonical(km)]] += 1.0
                n_valid += 1
        if n_valid > 0:
            counts /= n_valid
        tnf[cid] = counts
    log.info(f"TNF computed for {len(tnf):,} contigs ({skipped:,} skipped — below {min_len} bp)")
    return tnf


def tnf_similarity(a: str, b: str, tnf: dict) -> float | None:
    """Return cosine similarity, or None if either contig is absent (too short).
    Callers should treat None as 0.0 — neutral, not penalised.
    """
    if a not in tnf or b not in tnf:
        return None
    va, vb = tnf[a], tnf[b]
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _compute_rho(q_ranked: np.ndarray, r_ranked: np.ndarray) -> np.ndarray:
    """Vectorised Spearman r for n_pairs rows."""
    q_c = q_ranked - q_ranked.mean(axis=1, keepdims=True)
    r_c = r_ranked - r_ranked.mean(axis=1, keepdims=True)
    num = (q_c * r_c).sum(axis=1)
    denom = np.linalg.norm(q_c, axis=1) * np.linalg.norm(r_c, axis=1)
    return np.where(denom > 0, num / denom, 0.0)


def _analytical_pvalues(rho: np.ndarray, n_samples: int) -> np.ndarray:
    t_stat = rho * np.sqrt((n_samples - 2) / np.maximum(1.0 - rho ** 2, 1e-12))
    return 2 * t_dist.sf(np.abs(t_stat), df=n_samples - 2)


def _permutation_pvalues(
    rho_obs: np.ndarray,
    q_ranked: np.ndarray,
    r_ranked: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_pairs = q_ranked.shape[0]
    n_samples = q_ranked.shape[1]
    exceed = np.zeros(n_pairs, dtype=np.int32)
    abs_obs = np.abs(rho_obs)
    log.info(f"Running {n_permutations:,} permutations over {n_pairs:,} pairs ({n_samples} samples)...")
    for _ in range(n_permutations):
        perm = rng.permutation(n_samples)
        exceed += (np.abs(_compute_rho(q_ranked[:, perm], r_ranked)) >= abs_obs).astype(np.int32)
    return (exceed + 1) / (n_permutations + 1)


def vectorised_spearman_pairs(
    candidates: pd.DataFrame,
    depth_df: pd.DataFrame,
    thresholds: AdaptiveThresholds,
) -> pd.DataFrame:
    """Compute Spearman r + p-value for every (query, ref) row.
    Applies BH-FDR if thresholds.use_fdr is True.
    """
    candidates = candidates.copy()
    if candidates.empty:
        candidates["cov_r"] = pd.Series(dtype=float)
        candidates["pval"] = pd.Series(dtype=float)
        return candidates

    all_ctgs = list(set(candidates["query"]) | set(candidates["ref"]))
    ctgs_with_depth = [c for c in all_ctgs if c in depth_df.index]
    if not ctgs_with_depth:
        candidates["cov_r"] = np.nan
        candidates["pval"] = 1.0
        return candidates

    depth_mat = depth_df.loc[ctgs_with_depth].values.astype(np.float64)
    ranked_mat = rankdata(depth_mat, axis=1)
    ctg_idx = {c: i for i, c in enumerate(ctgs_with_depth)}
    n_samples = ranked_mat.shape[1]

    q_idx = candidates["query"].map(ctg_idx)
    r_idx = candidates["ref"].map(ctg_idx)
    valid = q_idx.notna() & r_idx.notna()
    candidates = candidates[valid].copy()

    if candidates.empty:
        candidates["cov_r"] = pd.Series(dtype=float)
        candidates["pval"] = pd.Series(dtype=float)
        return candidates

    q_ranked = ranked_mat[q_idx[valid].astype(int).values]
    r_ranked = ranked_mat[r_idx[valid].astype(int).values]
    rho = _compute_rho(q_ranked, r_ranked)

    if thresholds.use_permutation:
        pvals = _permutation_pvalues(
            rho, q_ranked, r_ranked,
            thresholds.n_permutations,
            np.random.default_rng(seed=42),
        )
    else:
        pvals = _analytical_pvalues(rho, n_samples)

    if thresholds.use_fdr and len(pvals) > 1:
        _, pvals, _, _ = multipletests(pvals, alpha=thresholds.cov_pval, method="fdr_bh")

    candidates["cov_r"] = rho
    candidates["pval"] = pvals
    return candidates


def leiden_communities(
    G: nx.Graph,
    resolution: float,
    seed: int = 42,
) -> list[set]:
    nodes = list(G.nodes())
    if not nodes:
        return []
    if not G.edges():
        return [{n} for n in nodes]

    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    return [{nodes[i] for i in community} for community in partition]

def build_cluster_centroids(
    clusters: dict,
    depth_df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Return a pseudo-rep list and a (n_clusters, n_samples) centroid matrix.
    
    For each cluster, the centroid is the mean depth vector across all members
    that appear in depth_df. Falls back to the single member if only one exists.
    Clusters with no members in depth_df are skipped.
    """
    cids = []
    centroids = []
    for cid, members in clusters.items():
        in_depth = [m for m in members if m in depth_df.index]
        if not in_depth:
            continue
        centroid = depth_df.loc[in_depth].values.astype(np.float64).mean(axis=0)
        cids.append(cid)
        centroids.append(centroid)
    return cids, np.array(centroids)   # (n_clusters, n_samples)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — Main clustering (genus / species)
# ═════════════════════════════════════════════════════════════════════════════

def _build_multimodal_graph(
    members: list[str],
    member_set: set[str],
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    tnf: dict,
    tax_df: pd.DataFrame,
    prot_sim_df: pd.DataFrame | None,
    thresholds: AdaptiveThresholds,
    ani_threshold: float,
    cov_threshold: float,
    w_ani: float = 0.40,
    w_cov: float = 0.35,
    w_tnf: float = 0.25,
) -> nx.Graph:
    """Build a weighted graph over `members` using ANI, taxonomy, and protein
    similarity as edge sources, with hybrid gating and rebalanced weights.

    Edge sources and their base weights:
      ANI edge    : w_ani * ani  + w_cov * cov_r + w_tnf * tnf_sim
      Protein edge: 0.30 * prot + 0.40 * cov_r  + 0.30 * tnf_sim
      Taxonomy edge: 0.00 seq   + 0.50 * cov_r  + 0.50 * tnf_sim
        (taxonomy has no sequence identity signal — weight is purely
         co-abundance + composition to avoid inflation)

    Hybrid gate (replaces single hard-coverage gate):
      Edge is added if:
        (ANI passes threshold AND at least one of cov/TNF passes), OR
        (cov AND TNF both pass — allows divergent but co-abundant contigs
         to connect even when skani is sparse)

    This means short or divergent contigs that skani cannot pair can still
    gain graph support from taxonomy or protein similarity edges, as long
    as co-abundance and composition signals agree.
    """
    G = nx.Graph()
    G.add_nodes_from(members)

    # ── 1. ANI edges ─────────────────────────────────────────────────────────
    ani_cands = ani_df[
        (ani_df["ani"] >= ani_threshold) &
        ani_df["query"].isin(member_set) &
        ani_df["ref"].isin(member_set)
    ].copy()

    ani_pairs: set[tuple] = set()
    if not ani_cands.empty:
        ani_cands = vectorised_spearman_pairs(ani_cands, depth_df, thresholds)
        for _, row in ani_cands.iterrows():
            a, b = row["query"], row["ref"]
            cov_r   = float(row["cov_r"])
            ani_val = float(row["ani"])
            tnf_sim = tnf_similarity(a, b, tnf)
            tnf_val = tnf_sim if tnf_sim is not None else 0.0

            ani_ok = True                                    # already filtered above
            cov_ok = cov_r  >= cov_threshold
            tnf_ok = tnf_sim is not None and tnf_sim >= thresholds.tnf_main

            if not ((ani_ok and (cov_ok or tnf_ok)) or (cov_ok and tnf_ok)):
                continue

            score = (
                w_ani * (ani_val / 100.0) +
                w_cov * max(cov_r, 0.0) +
                w_tnf * tnf_val
            )
            key = (min(a, b), max(a, b))
            if not G.has_edge(a, b) or G[a][b]["weight"] < score:
                G.add_edge(a, b, weight=score)
            ani_pairs.add(key)

    # ── 2. Protein-similarity edges (where ANI has no hit) ───────────────────
    if prot_sim_df is not None and not prot_sim_df.empty:
        prot_cands = prot_sim_df[
            prot_sim_df["query"].isin(member_set) &
            prot_sim_df["ref"].isin(member_set)
        ].copy()
        if not prot_cands.empty:
            prot_cands = vectorised_spearman_pairs(
                prot_cands.rename(columns={"prot_sim": "ani"}),
                depth_df, thresholds,
            )
            for _, row in prot_cands.iterrows():
                a, b = row["query"], row["ref"]
                key = (min(a, b), max(a, b))
                if key in ani_pairs:
                    continue          # ANI edge already present — don't downgrade
                cov_r    = float(row["cov_r"])
                prot_val = float(row["ani"]) / 100.0
                tnf_sim  = tnf_similarity(a, b, tnf)
                tnf_val  = tnf_sim if tnf_sim is not None else 0.0

                cov_ok = cov_r  >= cov_threshold
                tnf_ok = tnf_sim is not None and tnf_sim >= thresholds.tnf_main

                if not (cov_ok or tnf_ok):
                    continue         # protein alone is not enough — need at least one other signal

                score = 0.30 * prot_val + 0.40 * max(cov_r, 0.0) + 0.30 * tnf_val
                if not G.has_edge(a, b) or G[a][b]["weight"] < score:
                    G.add_edge(a, b, weight=score)

    # ── 3. Taxonomy edges (same-taxon pairs with no ANI or protein hit) ──────
    taxon_groups: dict[str, list] = defaultdict(list)
    for c in members:
        if c in tax_df.index:
            name = tax_df.loc[c, "name"]
            if name not in NOISE_TAXA:
                taxon_groups[name].append(c)

    for taxon, grp in taxon_groups.items():
        if len(grp) < 2:
            continue
        # Cap pairs per taxon group to avoid O(n²) blowup on large genera.
        # With e.g. 5000 Streptococcus contigs the full matrix is 12.5M pairs.
        # Randomly sample a manageable subset — coverage correlation will catch
        # missed within-group merges in Tier 3 recovery.
        MAX_TAX_PAIRS_PER_GROUP = 50_000
        pair_list = [
            {"query": a, "ref": b}
            for i, a in enumerate(grp)
            for b in grp[i + 1:]
            if (min(a, b), max(a, b)) not in ani_pairs
        ]
        if len(pair_list) > MAX_TAX_PAIRS_PER_GROUP:
            import random
            random.seed(42)
            pair_list = random.sample(pair_list, MAX_TAX_PAIRS_PER_GROUP)
            log.debug(
                f"Taxon '{taxon}': sampled {MAX_TAX_PAIRS_PER_GROUP:,} of "
                f"{len(grp) * (len(grp)-1) // 2:,} possible taxonomy pairs"
            )
        tax_cands_df = pd.DataFrame(pair_list)
        if tax_cands_df.empty:
            continue
        tax_cands_df = vectorised_spearman_pairs(tax_cands_df, depth_df, thresholds)
        for _, row in tax_cands_df.iterrows():
            a, b = row["query"], row["ref"]
            cov_r   = float(row["cov_r"])
            tnf_sim = tnf_similarity(a, b, tnf)
            tnf_val = tnf_sim if tnf_sim is not None else 0.0

            cov_ok = cov_r  >= cov_threshold
            tnf_ok = tnf_sim is not None and tnf_sim >= thresholds.tnf_main

            # Taxonomy edges require BOTH cov and TNF — no sequence support
            if not (cov_ok and tnf_ok):
                continue

            score = 0.50 * max(cov_r, 0.0) + 0.50 * tnf_val
            if not G.has_edge(a, b) or G[a][b]["weight"] < score:
                G.add_edge(a, b, weight=score)

    return G


def build_main_clusters(
    contigs_main: list[str],
    tax_df: pd.DataFrame,
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    tnf: dict,
    thresholds: AdaptiveThresholds,
    leiden_res: float = LEIDEN_RES_MAIN,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict]:
    log.info(
        f"Tier 1 — main clustering: {len(contigs_main):,} genus/species contigs "
        f"(leiden_res={leiden_res}, multimodal edges: ANI + protein + taxonomy)..."
    )
    main_set = set(contigs_main)

    G = _build_multimodal_graph(
        members=contigs_main,
        member_set=main_set,
        ani_df=ani_df,
        depth_df=depth_df,
        tnf=tnf,
        tax_df=tax_df,
        prot_sim_df=prot_sim_df,
        thresholds=thresholds,
        ani_threshold=thresholds.ani_main,
        cov_threshold=thresholds.cov_main,
        w_ani=0.40, w_cov=0.35, w_tnf=0.25,
    )

    n_edges = G.number_of_edges()
    log.info(f"Tier 1 graph: {n_edges:,} edges among {len(contigs_main):,} contigs")

    communities = leiden_communities(G, resolution=leiden_res, seed=42)
    clusters: dict[str, set] = {}
    membership: dict[str, str] = {}
    for i, comm in enumerate(communities):
        cid = f"cluster_{i:04d}"
        clusters[cid] = set(comm)
        for c in comm:
            membership[c] = cid

    n_multi = sum(1 for s in clusters.values() if len(s) > 1)
    log.info(
        f"-> {len(clusters):,} clusters from {len(contigs_main):,} main contigs "
        f"({n_multi:,} non-singleton)"
    )
    return clusters, membership


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2 — Secondary clustering (order / family / class / phylum)
# ═════════════════════════════════════════════════════════════════════════════

def build_secondary_clusters(
    contigs_secondary: list[str],
    clusters: dict,
    membership: dict,
    tax_df: pd.DataFrame,
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    tnf: dict,
    cluster_id_offset: int,
    thresholds: AdaptiveThresholds,
    leiden_res: float = LEIDEN_RES_SECONDARY,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict, list[str], int]:
    log.info(
        f"Tier 2 — secondary clustering: {len(contigs_secondary):,} above-genus contigs "
        f"(leiden_res={leiden_res}, multimodal edges: ANI + protein + taxonomy)..."
    )

    taxon_groups: dict[str, list] = defaultdict(list)
    ungrouped: list[str] = []
    for c in contigs_secondary:
        name = tax_df.loc[c, "name"] if c in tax_df.index else ""
        if name not in NOISE_TAXA:
            taxon_groups[name].append(c)
        else:
            ungrouped.append(c)

    clustered: set[str] = set()
    n_new = 0

    for _, members in taxon_groups.items():
        if len(members) == 1:
            continue
        member_set = set(members)

        G = _build_multimodal_graph(
            members=members,
            member_set=member_set,
            ani_df=ani_df,
            depth_df=depth_df,
            tnf=tnf,
            tax_df=tax_df,
            prot_sim_df=prot_sim_df,
            thresholds=thresholds,
            ani_threshold=thresholds.ani_secondary,
            cov_threshold=thresholds.cov_secondary,
            w_ani=0.40, w_cov=0.35, w_tnf=0.25,
        )

        for comm in leiden_communities(G, resolution=leiden_res, seed=42):
            if len(comm) == 1:
                continue
            cid = f"cluster_{cluster_id_offset:04d}"
            cluster_id_offset += 1
            clusters[cid] = set(comm)
            for c in comm:
                membership[c] = cid
                clustered.add(c)
            n_new += 1

    remaining = [c for c in contigs_secondary if c not in clustered] + ungrouped
    log.info(
        f"-> {n_new:,} secondary clusters from {len(clustered):,} contigs | "
        f"{len(remaining):,} pass to recovery"
    )
    return clusters, membership, remaining, cluster_id_offset


# ═════════════════════════════════════════════════════════════════════════════
# Tier 3 — Recovery
# ═════════════════════════════════════════════════════════════════════════════

def build_lineage_tax_index(cluster_reps: dict, tax_df: pd.DataFrame) -> dict[str, list[str]]:
    taxon_to_clusters: dict[str, set] = defaultdict(set)
    for cid, rep in cluster_reps.items():
        if rep not in tax_df.index:
            continue
        row = tax_df.loc[rep]
        for name in row["lineage"]:
            if name not in NOISE_TAXA:
                taxon_to_clusters[name].add(cid)
        if row["name"] not in NOISE_TAXA:
            taxon_to_clusters[row["name"]].add(cid)
    return {k: sorted(v) for k, v in taxon_to_clusters.items()}


def _taxonomy_candidates(
    contig: str,
    tax_df: pd.DataFrame,
    taxon_to_clusters: dict,
    max_cands: int = MAX_TAX_CANDIDATES,
) -> set[str]:
    if contig not in tax_df.index:
        return set()
    row = tax_df.loc[contig]
    names: list[str] = []
    if row["name"] not in NOISE_TAXA:
        names.append(row["name"])
    for x in reversed(row["lineage"]):
        if x not in NOISE_TAXA and x not in names:
            names.append(x)
    for name in names:
        matches = taxon_to_clusters.get(name, [])
        if matches:
            return set(matches[:max_cands])
    return set()


def recover_contigs(
    contigs_recover: list[str],
    clusters: dict,
    membership: dict,
    cluster_reps: dict,
    tax_df: pd.DataFrame,
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    absent_from_taxonomy: set[str],
    thresholds: AdaptiveThresholds,
    tnf: dict,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict, list[str]]:
    log.info(f"Tier 3 — recovery: {len(contigs_recover):,} contigs...")

    rep_to_cluster = {rep: cid for cid, rep in cluster_reps.items()}
    rep_set = set(cluster_reps.values())
    recover_set = set(contigs_recover)

    ani_rec = ani_df[
        (ani_df["ani"] >= thresholds.ani_recovery) &
        (
            (ani_df["query"].isin(recover_set) & ani_df["ref"].isin(rep_set)) |
            (ani_df["ref"].isin(recover_set) & ani_df["query"].isin(rep_set))
        )
    ].copy()

    q_is_recover = ani_rec["query"].isin(recover_set)
    df_a = ani_rec[q_is_recover][["query", "ref", "ani"]].rename(
        columns={"query": "contig", "ref": "rep"}
    )
    df_b = ani_rec[~q_is_recover][["ref", "query", "ani"]].rename(
        columns={"ref": "contig", "query": "rep"}
    )
    hits = pd.concat([df_a, df_b], ignore_index=True)
    hits = hits[hits["rep"].isin(rep_to_cluster)].copy()
    hits["cid"] = hits["rep"].map(rep_to_cluster)

    ani_hits: dict[str, dict[str, float]] = defaultdict(dict)
    for _, row in hits.sort_values("ani", ascending=False).iterrows():
        contig, cid, ani = row["contig"], row["cid"], row["ani"]
        if cid not in ani_hits[contig]:
            ani_hits[contig][cid] = float(ani)
    ani_hits = dict(ani_hits)
    log.info(f"ANI hits: {len(ani_hits):,} contigs have at least one representative hit")

    # ── Protein similarity hits ───────────────────────────────────────────────
    # Build contig → {cid: best_prot_sim} using same logic as ANI hits.
    # Only used when ANI and taxonomy both fail to find candidates.
    prot_hits: dict[str, dict[str, float]] = {}
    if prot_sim_df is not None and not prot_sim_df.empty:
        prot_rec = prot_sim_df[
            prot_sim_df["query"].isin(recover_set) &
            prot_sim_df["ref"].isin(rep_set)
        ].copy()
        prot_rec = prot_rec[prot_rec["ref"].isin(rep_to_cluster)]
        prot_rec["cid"] = prot_rec["ref"].map(rep_to_cluster)
        tmp: dict[str, dict[str, float]] = defaultdict(dict)
        for _, row in prot_rec.sort_values("prot_sim", ascending=False).iterrows():
            contig, cid, sim = row["query"], row["cid"], float(row["prot_sim"])
            if cid not in tmp[contig]:
                tmp[contig][cid] = sim
        prot_hits = dict(tmp)
        log.info(f"Protein hits: {len(prot_hits):,} contigs have at least one protein-similar rep")

    taxon_to_clusters = build_lineage_tax_index(cluster_reps, tax_df)
    log.info(f"Taxonomy index: {len(taxon_to_clusters):,} unique lineage names")

    # ── Build centroid matrix for Tier 3 coverage correlation ────────────────
    # Using per-cluster mean depth (centroid) rather than a single representative
    # makes recovery more robust when the representative is atypical for its bin.
    cid_list_t3, centroid_mat_t3 = build_cluster_centroids(clusters, depth_df)
    centroid_ranked_t3 = rankdata(centroid_mat_t3, axis=1)
    centroid_dev_t3    = centroid_ranked_t3 - centroid_ranked_t3.mean(axis=1, keepdims=True)
    centroid_idx_t3    = {cid: i for i, cid in enumerate(cid_list_t3)}
    del centroid_mat_t3, centroid_ranked_t3

    def correlate_candidates(contig_vec: np.ndarray, candidate_cids: set) -> dict[str, float]:
        """Spearman r between contig and each candidate cluster centroid."""
        idxs, valid_cids = [], []
        for cid in candidate_cids:
            i = centroid_idx_t3.get(cid)
            if i is not None:
                idxs.append(i)
                valid_cids.append(cid)
        if not idxs:
            return {}
        contig_ranked = rankdata(contig_vec)
        c_dev = contig_ranked - contig_ranked.mean()
        mat   = centroid_dev_t3[idxs]
        num   = (mat * c_dev).sum(axis=1)
        denom = np.sqrt((mat ** 2).sum(axis=1)) * np.sqrt((c_dev ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r_vals = np.where(denom > 0, num / denom, 0.0)
        return dict(zip(valid_cids, r_vals.tolist()))

    recovered_ani_only: list[str] = []
    recovered_tax_aided: list[str] = []
    recovered_ani_tax: list[str] = []
    recovered_prot: list[str] = []
    unassigned: list[str] = []
    n_no_candidates = 0
    n_no_depth = 0
    cov_thresh = thresholds.cov_recovery
    log_every = 50_000

    for i, contig in enumerate(contigs_recover):
        if i > 0 and i % log_every == 0:
            total_so_far = (len(recovered_ani_only) + len(recovered_tax_aided)
                            + len(recovered_ani_tax) + len(recovered_prot))
            pct = 100 * i / len(contigs_recover)
            log.info(
                f"Recovery progress: {i:,}/{len(contigs_recover):,} ({pct:.1f}%) — "
                f"recovered so far: {total_so_far:,}"
            )

        ani_candidates: set[str] = set(ani_hits.get(contig, {}).keys())

        tax_candidates = _taxonomy_candidates(contig, tax_df, taxon_to_clusters)

        # Protein candidates: only consulted when ANI + taxonomy both fail
        prot_candidates: set[str] = set()
        if not ani_candidates and not tax_candidates:
            prot_candidates = set(prot_hits.get(contig, {}).keys())

        candidate_clusters = ani_candidates | tax_candidates | prot_candidates
        if not candidate_clusters:
            n_no_candidates += 1
            unassigned.append(contig)
            continue

        if contig not in depth_df.index:
            n_no_depth += 1
            unassigned.append(contig)
            continue

        contig_vec = depth_df.loc[contig].values.astype(np.float64)
        r_map = correlate_candidates(contig_vec, candidate_clusters)

        scored: list[tuple[str, float]] = []
        for cid in candidate_clusters:
            r = r_map.get(cid, 0.0)
            if cov_thresh is not None and r < cov_thresh:
                continue
            if cid in ani_candidates:
                # ANI: rebalanced to 0.40/0.35/0.25 consistent with Tiers 1/2
                seq_component = (ani_hits[contig][cid] / 100.0) * 0.40
                cov_w, tnf_w  = 0.35, 0.25
            elif cid in prot_candidates:
                # Protein: less precise than ANI, cov+TNF dominate
                seq_component = (prot_hits[contig][cid] / 100.0) * 0.30
                cov_w, tnf_w  = 0.40, 0.30
            else:
                # Taxonomy only: no sequence signal, pure co-abundance + composition
                seq_component = 0.0
                cov_w, tnf_w  = 0.50, 0.50
            tnf_sim = tnf_similarity(contig, cluster_reps[cid], tnf)
            score = (
                seq_component
                + cov_w * max(float(r), 0.0)
                + tnf_w * (tnf_sim or 0.0)
            )
            scored.append((cid, score))

        if not scored:
            unassigned.append(contig)
            continue

        best_cid , best_score = max(scored, key=lambda x: x[1])
        if best_score < 0.08:
            unassigned.append(contig)
            continue

        clusters[best_cid].add(contig)
        membership[contig] = best_cid

        if best_cid in ani_candidates and best_cid in tax_candidates:
            recovered_ani_tax.append(contig)
        elif best_cid in ani_candidates:
            recovered_ani_only.append(contig)
        elif best_cid in prot_candidates:
            recovered_prot.append(contig)
        else:
            recovered_tax_aided.append(contig)

    total_recovered = (len(recovered_ani_only) + len(recovered_tax_aided)
                       + len(recovered_ani_tax) + len(recovered_prot))

    # Free the centroid matrix now that the recovery loop is done
    del centroid_dev_t3

    log.info(f"Recovered {total_recovered:,} contigs")
    log.info(f"  ANI-only path:        {len(recovered_ani_only):,}")
    log.info(f"  Taxonomy-only path:   {len(recovered_tax_aided):,}")
    log.info(f"  Both paths available: {len(recovered_ani_tax):,}")
    log.info(f"  Protein-only path:    {len(recovered_prot):,}")
    log.info(f"Unassigned: {len(unassigned):,}")
    log.info(f"  No candidates at all: {n_no_candidates:,}")
    log.info(f"  Missing from depth:   {n_no_depth:,}")
    log.info(
        f"  No taxonomy + no ANI: "
        f"{len([c for c in unassigned if c in absent_from_taxonomy]):,}"
    )
    log.info(
        f"  Had ANI hit, cov/score failed: "
        f"{len([c for c in unassigned if c in ani_hits]):,}"
    )
    return clusters, membership, unassigned


# ═════════════════════════════════════════════════════════════════════════════
# Tier 5 — coverage-only recovery (no ANI, no taxonomy required)
# ═════════════════════════════════════════════════════════════════════════════

def coverage_only_recovery(
    unassigned: list[str],
    clusters: dict,
    membership: dict,
    cluster_reps: dict,
    records: dict,
    depth_df: pd.DataFrame,
    tnf: dict,
    cov_r_min: float = 0.80,
    tnf_min: float | None = 0.90,
    rep_chunk_size: int = 2000,
) -> tuple[dict, dict, list[str]]:
    """Last-resort recovery using coverage correlation alone.

    For each unassigned contig, correlate its depth vector against every
    cluster centroid. If the best match exceeds `cov_r_min` AND
    (optionally) the TNF cosine similarity exceeds `tnf_min`, assign it.
    TNF guard prevents assigning plasmids/phage to the wrong host bin when
    coverage co-varies by chance.

    This runs after Tier 4 so cluster_reps already includes de-novo bins.

    Memory-efficient implementation: the full (n_contigs × n_reps) rho matrix
    is never materialised. Instead, representatives are processed in chunks of
    `rep_chunk_size`, keeping peak RAM at:
        n_contigs × rep_chunk_size × 8 bytes
    e.g. 406k contigs × 2000 reps × 8 B ≈ 6.5 GB per chunk pass,
    vs 406k × 50k × 8 B ≈ 162 GB for the full matrix.
    """
    log.info(
        f"Tier 5 — coverage-only recovery: {len(unassigned):,} contigs "
        f"(cov_r_min={cov_r_min}, tnf_min={tnf_min}, "
        f"rep_chunk_size={rep_chunk_size:,})..."
    )
    if not unassigned or not cluster_reps:
        return clusters, membership, unassigned

    cid_list, centroid_matrix = build_cluster_centroids(clusters, depth_df)
    if not cid_list:
        log.warning("Tier 5: no cluster centroids found in depth matrix — skipping")
        return clusters, membership, unassigned

    rep_ranked = rankdata(centroid_matrix, axis=1)
    r_dev      = rep_ranked - rep_ranked.mean(axis=1, keepdims=True)
    r_norms    = np.linalg.norm(r_dev, axis=1)
    del centroid_matrix, rep_ranked

    # ── Pre-rank unassigned contig depth vectors ──────────────────────────────
    eligible        = [c for c in unassigned if c in depth_df.index]
    skipped_no_depth = len(unassigned) - len(eligible)
    if not eligible:
        log.info("Tier 5: no unassigned contigs found in depth matrix")
        return clusters, membership, unassigned

    contig_depth  = depth_df.loc[eligible].values.astype(np.float64)
    contig_ranked = rankdata(contig_depth, axis=1)    # (n_contigs, n_samples)
    c_dev         = contig_ranked - contig_ranked.mean(axis=1, keepdims=True)
    c_norms       = np.linalg.norm(c_dev, axis=1)    # (n_contigs,)
    del contig_depth, contig_ranked

    n_contigs = len(eligible)
    #n_reps    = len(rep_list)
    n_reps = len(cid_list)
    # Accumulators — track best (r, rep_index) seen so far for each contig
    best_r       = np.full(n_contigs, -2.0, dtype=np.float64)
    best_rep_idx = np.zeros(n_contigs, dtype=np.int32)

    # ── Chunked matrix multiply over representative blocks ────────────────────
    n_chunks = (n_reps + rep_chunk_size - 1) // rep_chunk_size
    for chunk_i, rep_start in enumerate(range(0, n_reps, rep_chunk_size)):
        rep_end = min(rep_start + rep_chunk_size, n_reps)

        r_dev_chunk  = r_dev[rep_start:rep_end]       # (chunk, n_samples)
        r_norm_chunk = r_norms[rep_start:rep_end]     # (chunk,)

        # Numerator: (n_contigs, chunk)
        num_chunk  = c_dev @ r_dev_chunk.T

        # Denominator: outer product of norms  (n_contigs, chunk)
        denom_chunk = np.outer(c_norms, r_norm_chunk)

        with np.errstate(invalid="ignore", divide="ignore"):
            rho_chunk = np.where(denom_chunk > 0, num_chunk / denom_chunk, 0.0)

        del num_chunk, denom_chunk

        # Update best only where this chunk improves on prior best
        chunk_best_local_idx = rho_chunk.argmax(axis=1)         # (n_contigs,)
        chunk_best_r         = rho_chunk[
            np.arange(n_contigs), chunk_best_local_idx
        ]
        del rho_chunk

        improved = chunk_best_r > best_r
        best_r[improved]       = chunk_best_r[improved]
        best_rep_idx[improved] = rep_start + chunk_best_local_idx[improved]

        if (chunk_i + 1) % 10 == 0 or rep_end == n_reps:
            log.info(
                f"Tier 5: processed {rep_end:,}/{n_reps:,} reps "
                f"({100 * rep_end / n_reps:.0f}%)"
            )

    del c_dev, r_dev, c_norms, r_norms

    # ── Assign contigs to their best-matching representative ──────────────────
    recovered_cov:   list[str] = []
    still_unassigned: list[str] = []

    for i, contig in enumerate(eligible):
        r = best_r[i]
        if r < cov_r_min:
            still_unassigned.append(contig)
            continue

        #best_rep = rep_list[best_rep_idx[i]]
        best_cid = cid_list[best_rep_idx[i]]
        best_rep = cluster_reps[best_cid]   # still needed for TNF guard

        # Optional TNF guard — skip if contig's composition is too dissimilar.
        # Short contigs with no TNF entry (sim is None) are allowed through —
        # they have no composition signal so we cannot penalise them for it.
        if tnf_min is not None and tnf_min > 0.0:
            sim = tnf_similarity(contig, best_rep, tnf)
            if sim is not None and sim < tnf_min:
                still_unassigned.append(contig)
                continue

        #best_cid = cid_of_rep[best_rep]
        clusters[best_cid].add(contig)
        membership[contig] = best_cid
        recovered_cov.append(contig)

    # Contigs with no depth entry are permanently unassigned
    still_unassigned += [c for c in unassigned if c not in depth_df.index]

    log.info(f"Tier 5 recovered {len(recovered_cov):,} contigs via coverage-only path")
    log.info(f"  Skipped (no depth):  {skipped_no_depth:,}")
    log.info(f"  Still unassigned:    {len(still_unassigned):,}")
    return clusters, membership, still_unassigned

# ═════════════════════════════════════════════════════════════════════════════
# Tier 4 — de-novo clustering of unassigned contigs
# ═════════════════════════════════════════════════════════════════════════════

def cluster_unassigned(
    unassigned: list[str],
    clusters: dict,
    membership: dict,
    cluster_reps: dict,
    records: dict,
    tax_df: pd.DataFrame,
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    tnf: dict,
    thresholds: AdaptiveThresholds,
    cluster_id_offset: int,
    leiden_res: float = 0.8,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict, dict, list[str], int]:
    """Cluster unassigned contigs against each other using ANI + coverage + TNF.

    Protein similarity pairs (if supplied) are used as additional edges
    when no ANI hit exists — useful for contigs that are too divergent for
    skani but share homologous proteins.
    Contigs that form multi-member communities become new bins. Singletons
    remain unassigned. Returns updated clusters, membership, cluster_reps and
    a new unassigned list containing only the true singletons.
    """
    log.info(f"Tier 4 — de-novo clustering of {len(unassigned):,} unassigned contigs...")
    if not unassigned:
        return clusters, membership, cluster_reps, unassigned, cluster_id_offset

    unassigned_set = set(unassigned)

    candidates = ani_df[
        (ani_df["ani"] >= thresholds.ani_recovery) &
        ani_df["query"].isin(unassigned_set) &
        ani_df["ref"].isin(unassigned_set)
    ].copy()
    log.info(f"Tier 4: {len(candidates):,} ANI candidate pairs among unassigned contigs")

    # Supplement with protein similarity pairs where ANI has no hit
    if prot_sim_df is not None and not prot_sim_df.empty:
        prot_cands = prot_sim_df[
            prot_sim_df["query"].isin(unassigned_set) &
            prot_sim_df["ref"].isin(unassigned_set)
        ][["query", "ref", "prot_sim"]].copy()
        # Only keep pairs not already covered by ANI
        ani_pairs = set(zip(candidates["query"], candidates["ref"]))
        prot_cands = prot_cands[
            ~prot_cands.apply(lambda r: (r["query"], r["ref"]) in ani_pairs, axis=1)
        ].copy()
        prot_cands = prot_cands.rename(columns={"prot_sim": "ani"})  # reuse ani column name
        prot_cands["_is_prot"] = True
        candidates["_is_prot"] = False
        candidates = pd.concat([candidates, prot_cands], ignore_index=True)
        log.info(f"Tier 4: added {len(prot_cands):,} protein-similarity pairs")

    G = nx.Graph()
    G.add_nodes_from(unassigned)

    if not candidates.empty:
        candidates = vectorised_spearman_pairs(candidates, depth_df, thresholds)
        for _, row in candidates.iterrows():
            a, b = row["query"], row["ref"]
            if thresholds.coverage_is_hard_gate:
                if not (row["cov_r"] >= (thresholds.cov_recovery or 0.0)
                        and row["pval"] < thresholds.cov_pval):
                    continue
            tnf_sim = tnf_similarity(a, b, tnf)
            is_prot = row.get("_is_prot", False)
            ani_weight = (row["ani"] / 100.0) * (0.50 if is_prot else 0.60)
            score = (
                ani_weight +
                max(float(row["cov_r"]), 0.0) * 0.25 +
                (tnf_sim if tnf_sim is not None else 0.0) * 0.15
            )
            G.add_edge(a, b, weight=score)

    communities = leiden_communities(G, resolution=leiden_res, seed=42)

    n_new = 0
    still_unassigned: list[str] = []

    for comm in communities:
        if len(comm) == 1:
            still_unassigned.extend(comm)
            continue
        cid = f"cluster_{cluster_id_offset:04d}"
        cluster_id_offset += 1
        clusters[cid] = set(comm)
        rep = representative(set(comm), records, depth_df)
        cluster_reps[cid] = rep
        for c in comm:
            membership[c] = cid
        n_new += 1

    log.info(
        f"Tier 4: {n_new:,} new clusters from unassigned pool | "
        f"{len(still_unassigned):,} remain unassigned"
    )
    return clusters, membership, cluster_reps, still_unassigned, cluster_id_offset


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
        os.path.join(os.path.join(outdir,unassigned_dir), "unassigned_contigs.fasta"),
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

def run_checkm2(bin_list_path: str, outdir: str, threads: int) -> pd.DataFrame:
    checkm2_dir = os.path.join(outdir, "checkm2")
    os.makedirs(checkm2_dir, exist_ok=True)
    quality_tsv = os.path.join(checkm2_dir, "quality_report.tsv")

    if not os.path.exists(quality_tsv):
        with open(bin_list_path) as fh:
            bin_paths = [line.strip() for line in fh if line.strip()]
        if not bin_paths:
            log.warning("No bins in checkm2_bin_list.txt — skipping CheckM2")
            return pd.DataFrame()
        # Bins live in outdir/clusters/bins — pass that directory to checkm2.
        # Representatives are in outdir/clusters/representatives/ and must be
        # kept separate so CheckM2 doesn't try to assess single-contig files.
        #bin_input_dir = os.path.join(outdir, "clusters", "thresh_bins")
        
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
    drep_dir = os.path.join(outdir, "drep")
    derep_dir = os.path.join(drep_dir, "dereplicated_genomes")

    if os.path.exists(derep_dir) and os.listdir(derep_dir):
        # Non-empty dereplicated_genomes dir = dRep already completed successfully.
        # An empty dir means dRep ran but all bins were identical — we don't re-run.
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

# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ckpt_path(ckpt_dir: str, name: str, ext: str) -> str:
    return os.path.join(ckpt_dir, f"{name}.{ext}")


def _ckpt_exists(ckpt_dir: str, name: str, ext: str) -> bool:
    return os.path.exists(_ckpt_path(ckpt_dir, name, ext))


def save_parquet(df: pd.DataFrame, ckpt_dir: str, name: str) -> None:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(json.dumps)
    df.to_parquet(_ckpt_path(ckpt_dir, name, "parquet"))


def load_parquet(ckpt_dir: str, name: str, listcols=None) -> pd.DataFrame:
    df = pd.read_parquet(_ckpt_path(ckpt_dir, name, "parquet"))
    for col in listcols or []:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def save_json(obj: object, ckpt_dir: str, name: str) -> None:
    def default(o):
        if isinstance(o, set):
            return list(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")
    with open(_ckpt_path(ckpt_dir, name, "json"), "w") as fh:
        json.dump(obj, fh, default=default)


def load_json(ckpt_dir: str, name: str) -> object:
    with open(_ckpt_path(ckpt_dir, name, "json")) as fh:
        return json.load(fh)


def save_tnf(tnf: dict, ckpt_dir: str) -> None:
    contigs = list(tnf.keys())
    matrix = np.stack([tnf[c] for c in contigs]).astype(np.float32)
    df = pd.DataFrame(matrix, index=pd.Index(contigs, name="contig"))
    df.to_parquet(_ckpt_path(ckpt_dir, "tnf", "parquet"))


def load_tnf(ckpt_dir: str) -> dict:
    df = pd.read_parquet(_ckpt_path(ckpt_dir, "tnf", "parquet"))
    return {contig: row.values.astype(np.float32) for contig, row in df.iterrows()}


# ═════════════════════════════════════════════════════════════════════════════
# CLI — main()
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

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


if __name__ == "__main__":
    main()
