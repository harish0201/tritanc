#!/usr/bin/env python3
"""
Taxonomy-aware metagenomic contig clustering pipeline.

USAGE — fully automatic (only FASTA + BAMs + MMseqs2 DB required):
  python pipeline.py \\
    --fasta assembly.fasta \\
    --bams sample1.bam sample2.bam ... \\
    --mmseqs-db /path/to/GTDB \\
    --outdir results/

USAGE — skip individual steps by supplying pre-computed files:
  python pipeline.py \\
    --fasta assembly.fasta \\
    --bams sample*.bam \\            # omit if --depth supplied
    --mmseqs-db /path/to/GTDB \\     # omit if --taxonomy supplied
    --taxonomy saliva_tax.tsv \\     # skips MMseqs2
    --taxonomy-format mmseqs2 \\     # or 'taxometer'
    --ani skani.tsv \\               # skips skani
    --depth depth_matrix.txt \\      # skips jgi depth step
    --outdir results/

Step skip logic:
  --taxonomy supplied?  -> skip MMseqs2 taxonomy run
  --ani supplied?       -> skip skani triangle run
  --depth supplied?     -> skip jgi_summarize_bam_contig_depths run
  BAMs required unless  -> --depth is supplied
  MMseqs2 DB required   -> unless --taxonomy is supplied

Outputs:
  outdir/
    taxonomy/          MMseqs2 output (if run)
    ani/               skani output (if run)
    depth/             jgi depth matrix (if run)
    clusters/          cluster_NNNN_bin.fasta + cluster_NNNN_representative.fasta
    unassigned/        unassigned_contigs.fasta (not sent to CheckM2)
    cluster_summary.tsv
    checkm2_bin_list.txt
    checkm2/               CheckM2 completeness/contamination results (if run)
    drep/                  Dereplicated bins (if run)
    final_bins/            Symlinks to dereplicated high+medium quality bins
"""

import os
import sys
import glob
import json
import shutil
import argparse
import logging
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from Bio import SeqIO
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ── Defaults ───────────────────────────────────────────────────────────────────
MIN_LEN       = 5000   # bp
ANI_THRESHOLD = 95.0   # %
COV_THRESHOLD = 0.90   # Spearman r
COV_PVAL      = 0.05   # BH-FDR adjusted p-value

NOISE_TAXA = {"", "root", "cellular organisms", "unclassified", "N/A"}

# Canonical rank order (VAMB/Taxometer): position encodes rank level
# domain[0] phylum[1] class[2] order[3] family[4] genus[5] species[6]
CANONICAL_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
GENUS_IDX   = 5
SPECIES_IDX = 6

TAXOMETER_PREFIX = {
    "d": "domain", "p": "phylum", "c": "class",
    "o": "order",  "f": "family", "g": "genus", "s": "species",
}

# Ranks for main clustering (genus/species — tight ANI + coverage)
MAIN_RANKS = {"genus", "species"}
# Ranks for secondary clustering pass (above genus but below domain)
# These form clusters among themselves before falling to recovery
SECONDARY_RANKS = {"order", "family", "class", "phylum"}

# Quality thresholds for CheckM2 filtering and dRep
CHECKM2_MIN_COMPLETENESS  = 50.0   # %
CHECKM2_MAX_CONTAMINATION = 10.0   # %
DREP_ANI                  = 95.0   # % species-level dereplication threshold

# MMseqs2 taxonomy parameters
MMSEQS_THREADS   = 8
MMSEQS_TAX_LINEAGE = 1   # --tax-lineage value
MMSEQS_SENS      = 4     # --sensitivity
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tool checks
# ══════════════════════════════════════════════════════════════════════════════

TOOL_HINTS = {
    "mmseqs":                         "https://github.com/soedinglab/MMseqs2",
    "skani":                          "https://github.com/bluenote-1577/skani  (pip install skani  or  conda install -c bioconda skani)",
    "jgi_summarize_bam_contig_depths": "part of MetaBAT2 — conda install -c bioconda metabat2",
    "samtools":                       "https://www.htslib.org  (conda install -c bioconda samtools)",
    "checkm2":                        "https://github.com/chklovski/CheckM2  (conda install -c bioconda checkm2)",
    "dRep":                           "https://github.com/MrOlm/drep  (conda install -c bioconda drep)",
}

def check_tools(needed: list[str]):
    """
    Verify each tool in `needed` is on PATH.
    Raises SystemExit with a clear message for any missing tool.
    """
    missing = []
    for tool in needed:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        lines = ["The following required tools were not found on PATH:"]
        for t in missing:
            hint = TOOL_HINTS.get(t, "")
            lines.append(f"  {t:<40} {hint}")
        log.error("\n".join(lines))
        sys.exit(1)
    log.info(f"Tool check passed: {', '.join(needed)}")


def tools_needed_for(args) -> list[str]:
    """Return only the tools actually required given which files are pre-supplied."""
    needed = []
    if not args.taxonomy:
        needed.append("mmseqs")
    if not args.ani:
        needed.append("skani")
    if not args.depth:
        needed += ["jgi_summarize_bam_contig_depths", "samtools"]
    if not args.skip_checkm2:
        needed.append("checkm2")
    if not args.skip_drep:
        needed.append("dRep")
    return needed


# ══════════════════════════════════════════════════════════════════════════════
# 2. External tool runners
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd: list[str], desc: str):
    """Run a subprocess, stream stderr to log, raise on failure."""
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        log.error(f"{desc} failed (exit {result.returncode})")
        sys.exit(result.returncode)


def run_mmseqs_taxonomy(fasta: str, db: str, outdir: str, threads: int) -> str:
    """
    Run mmseqs taxonomy on the assembly FASTA against the provided database.
    Returns path to the final TSV result file.

    Equivalent to:
      mmseqs taxonomy <fasta_db> <ref_db> <result> <tmp> \\
        --tax-lineage 1 --sensitivity 4 --threads N
      mmseqs createtsv <fasta_db> <result> <result_tsv>
    """
    tax_dir = os.path.join(outdir, "taxonomy")
    os.makedirs(tax_dir, exist_ok=True)

    fasta_db  = os.path.join(tax_dir, "assembly_db")
    result_db = os.path.join(tax_dir, "taxonomy_result")
    tmp_dir   = os.path.join(tax_dir, "tmp")
    tsv_out   = os.path.join(tax_dir, "taxonomy.tsv")

    if os.path.exists(tsv_out):
        log.info(f"  MMseqs2 taxonomy TSV already exists, skipping: {tsv_out}")
        return tsv_out

    log.info("── MMseqs2 taxonomy ──────────────────────────────────────────")

    # Create sequence DB from assembly
    run(["mmseqs", "createdb", fasta, fasta_db], "mmseqs createdb")

    # Run taxonomy search + LCA
    run([
        "mmseqs", "taxonomy",
        fasta_db, db, result_db, tmp_dir,
        "--tax-lineage", str(MMSEQS_TAX_LINEAGE),
        "--sensitivity", str(MMSEQS_SENS),
        "--threads", str(threads),
    ], "mmseqs taxonomy")

    # Convert to TSV
    run([
        "mmseqs", "createtsv",
        fasta_db, result_db, tsv_out,
    ], "mmseqs createtsv")

    log.info(f"  MMseqs2 taxonomy done -> {tsv_out}")
    return tsv_out


def run_skani(fasta: str, outdir: str, threads: int) -> str:
    """
    Run skani triangle (all-vs-all ANI) on the assembly FASTA.
    Returns path to the output TSV.

    Equivalent to:
      skani triangle <fasta> -o <out.tsv> --sparse -s 80 --threads N
    """
    ani_dir = os.path.join(outdir, "ani")
    os.makedirs(ani_dir, exist_ok=True)
    tsv_out = os.path.join(ani_dir, "skani_triangle.tsv")

    if os.path.exists(tsv_out):
        log.info(f"  skani output already exists, skipping: {tsv_out}")
        return tsv_out

    log.info("── skani triangle ────────────────────────────────────────────")
    run([
        "skani", "triangle",
        "-i", fasta,
        "-o", tsv_out,
        "--sparse",
        "-c", "30",
        "-m", "200",
        "--faster-small",
        "-s", "80", # min sketch identity; filters very distant pairs early
        "-t", str(threads)
    ], "skani triangle")

    log.info(f"  skani done -> {tsv_out}")
    return tsv_out



def run_depth(bams: list[str], outdir: str) -> str:
    """
    Run jgi_summarize_bam_contig_depths on all BAM files.
    Returns path to the depth matrix TSV.

    Equivalent to:
      jgi_summarize_bam_contig_depths --outputDepth depth_matrix.txt *.bam
    """
    depth_dir = os.path.join(outdir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    tsv_out = os.path.join(depth_dir, "depth_matrix.txt")

    if os.path.exists(tsv_out):
        log.info(f"  Depth matrix already exists, skipping: {tsv_out}")
        return tsv_out

    log.info("── jgi_summarize_bam_contig_depths ───────────────────────────")

    # Verify BAI index exists for each BAM; create if missing
    for bam in bams:
        bai = bam + ".bai"
        if not os.path.exists(bai):
            log.info(f"  Indexing {os.path.basename(bam)}...")
            run(["samtools", "index", bam], "samtools index")

    run(
        ["jgi_summarize_bam_contig_depths", "--outputDepth", tsv_out] + bams,
        "jgi_summarize_bam_contig_depths",
    )

    log.info(f"  Depth matrix done -> {tsv_out}")
    return tsv_out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Parsers
# ══════════════════════════════════════════════════════════════════════════════

def parse_fasta(path):
    log.info(f"Loading assembly: {path}")
    records = {r.id: r for r in SeqIO.parse(path, "fasta")}
    log.info(f"  {len(records):,} contigs")
    return records


def _empty_tax_row(contig):
    return {
        "contig":  contig,
        "taxid":   None,
        "rank":    "unclassified",
        "name":    "unclassified",
        "lineage": [],
        "scores":  [],
    }


def parse_taxonomy_mmseqs2(path):
    """
    MMseqs2 LCA TSV (--tax-lineage 1).
    Columns: contig | top_hit | rank | name | nfrag | nret | nagreed | score | lineage
    """
    log.info(f"Parsing MMseqs2 taxonomy: {path}")
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            contig      = parts[0]
            rank        = parts[2].strip()
            name        = parts[3].strip()
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
                "contig":  contig,
                "taxid":   parts[1].strip() if len(parts) > 1 else None,
                "rank":    rank if rank else "unclassified",
                "name":    name if name else "unclassified",
                "lineage": lineage,
                "scores":  [],
            })
    df = pd.DataFrame(rows).set_index("contig")
    log.info(f"  {len(df):,} assignments loaded")
    return df


def _strip_rank_prefix(s):
    if len(s) >= 3 and s[1] == "_" and s[0].lower() in TAXOMETER_PREFIX:
        return s[2:]
    return s


def parse_taxonomy_taxometer(path, min_score=0.0):
    """
    Taxometer output (vamb taxometer).
    Header: contigs\\tpredictions\\tscores
    Data:   contig\\tlineage\\tscores  (3 fields)
            contig                     (1 field — no taxonomy assigned)
    Rank determined by position in canonical 7-rank hierarchy.
    """
    log.info(f"Parsing Taxometer taxonomy: {path} (min_score={min_score})")
    rows = []
    no_tax_count = 0

    with open(path) as fh:
        header = next(fh, "").rstrip()
        if header != "contigs\tpredictions\tscores":
            raise ValueError(
                f"Unexpected Taxometer header: {repr(header)}\n"
                f"Expected: 'contigs\\tpredictions\\tscores'"
            )
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts  = line.split("\t")
            contig = parts[0].strip()

            if len(parts) == 1:
                no_tax_count += 1
                rows.append(_empty_tax_row(contig))
                continue

            if len(parts) != 3:
                raise ValueError(
                    f"Line {lineno}: expected 1 or 3 fields, got {len(parts)}: {repr(line)}"
                )

            lineage_raw = parts[1].strip().split(";")
            scores_raw  = parts[2].strip().split(";")
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
                rank_idx = min(depth - 1, len(CANONICAL_RANKS) - 1)
                rank = CANONICAL_RANKS[rank_idx]
                name = lineage_names[-1]

            rows.append({
                "contig":  contig,
                "taxid":   None,
                "rank":    rank,
                "name":    name,
                "lineage": lineage_names,
                "scores":  trusted_scores,
            })

    df = pd.DataFrame(rows).set_index("contig")
    log.info(
        f"  {len(df):,} assignments loaded "
        f"({no_tax_count:,} with no taxonomy in file)"
    )
    return df


def build_taxonomy_df(tax_df, all_contigs):
    """
    Ensure every contig in the assembly has a taxonomy row.
    Contigs absent from the taxonomy file are marked unclassified.
    Returns the augmented DataFrame plus a list of the absent contig IDs
    (used later for recovery breakdown reporting).
    """
    missing = [c for c in all_contigs if c not in tax_df.index]
    if missing:
        log.info(
            f"  {len(missing):,} contigs absent from taxonomy file "
            f"-> marked unclassified (ANI-only recovery)"
        )
        missing_df = pd.DataFrame(
            [_empty_tax_row(c) for c in missing]
        ).set_index("contig")
        tax_df = pd.concat([tax_df, missing_df])
    return tax_df, missing


def parse_depth(path):
    log.info(f"Loading depth matrix: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    depth_cols = [
        c for c in df.columns
        if not c.endswith("-var") and c not in ("contigLen", "totalAvgDepth")
    ]
    df = df[depth_cols].copy()
    df = np.log1p(df)
    log.info(f"  {len(df):,} contigs x {len(depth_cols)} samples")
    return df


def parse_ani(path):
    """
    Parse skani triangle output.

    skani triangle emits a header line:
      Ref_file  Query_file  ANI  Align_fraction_ref  Align_fraction_query  Ref_name  Query_name

    Contig IDs are in Ref_name (col 5) and Query_name (col 6).
    ANI is col 2, alignment fractions are cols 3 and 4.
    """
    log.info(f"Loading ANI results: {path}")
    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)

    # Validate expected columns are present
    expected = {"Ref_name", "Query_name", "ANI"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"skani output missing expected columns. "
            f"Found: {list(df.columns)}. Expected at least: {sorted(expected)}"
        )

    df = df.rename(columns={
        "Ref_name":              "query",
        "Query_name":            "ref",
        "ANI":                   "ani",
        "Align_fraction_ref":    "qcov",
        "Align_fraction_query":  "rcov",
    })[["query", "ref", "ani", "qcov", "rcov"]].copy()

    # Ensure numeric
    for col in ("ani", "qcov", "rcov"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ani"]).reset_index(drop=True)

    log.info(f"  {len(df):,} ANI pairs loaded")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 4. Helpers
# ══════════════════════════════════════════════════════════════════════════════

def lineage_set(row):
    s = set(row["lineage"]) - NOISE_TAXA
    if row["name"] not in NOISE_TAXA:
        s.add(row["name"])
    return s


def coverage_corr(a, b, depth_df):
    if a not in depth_df.index or b not in depth_df.index:
        return None, None
    r, p = spearmanr(depth_df.loc[a], depth_df.loc[b])
    return float(r), float(p)


def representative(contigs, records, depth_df):
    def score(c):
        length = len(records[c].seq) if c in records else 0
        depth  = float(depth_df.loc[c].mean()) if c in depth_df.index else 0.0
        return (length, depth)
    return max(contigs, key=score)


def get_genus_name(contig, tax_df):
    row = tax_df.loc[contig]
    if row["rank"] == "genus":
        return row["name"]
    if row["rank"] == "species":
        lin = row["lineage"]
        if len(lin) > GENUS_IDX:
            return lin[GENUS_IDX]
        if len(lin) >= 2:
            return lin[-2]
    return row["name"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main clustering
# ══════════════════════════════════════════════════════════════════════════════

def build_main_clusters(contigs_main, tax_df, ani_df, depth_df):
    log.info("Building main clusters...")
    main_set = set(contigs_main)

    candidates = ani_df[
        (ani_df["ani"] >= ANI_THRESHOLD) &
        (ani_df["query"].isin(main_set)) &
        (ani_df["ref"].isin(main_set))
    ].copy()

    genus_map = {c: get_genus_name(c, tax_df) for c in contigs_main if c in tax_df.index}
    candidates = candidates[
        candidates.apply(
            lambda r: genus_map.get(r["query"]) == genus_map.get(r["ref"]), axis=1
        )
    ].copy()
    log.info(f"  {len(candidates):,} within-genus ANI pairs above threshold")

    G = nx.Graph()
    G.add_nodes_from(contigs_main)

    if not candidates.empty:
        cov_r_list, pval_list = [], []
        for _, row in candidates.iterrows():
            r, p = coverage_corr(row["query"], row["ref"], depth_df)
            cov_r_list.append(r if r is not None else np.nan)
            pval_list.append(p if p is not None else 1.0)

        candidates = candidates.copy()
        candidates["cov_r"]    = cov_r_list
        candidates["pval"]     = pval_list
        _, adj_pvals, _, _     = multipletests(pval_list, method="fdr_bh")
        candidates["pval_adj"] = adj_pvals

        passing = candidates[
            (candidates["cov_r"] >= COV_THRESHOLD) &
            (candidates["pval_adj"] < COV_PVAL)
        ]
        log.info(f"  {len(passing):,} pairs pass coverage correlation filter")
        for _, row in passing.iterrows():
            G.add_edge(row["query"], row["ref"])

    clusters, membership = {}, {}
    for i, component in enumerate(nx.connected_components(G)):
        cid = f"cluster_{i:04d}"
        clusters[cid] = set(component)
        for c in component:
            membership[c] = cid

    log.info(f"  -> {len(clusters):,} clusters from {len(contigs_main):,} contigs")
    return clusters, membership



# ══════════════════════════════════════════════════════════════════════════════
# 5b. Secondary clustering (above-genus assigned contigs)
# ══════════════════════════════════════════════════════════════════════════════

def get_coarse_taxon(contig, tax_df):
    """
    Return the deepest available taxon name for a contig assigned above genus.
    Used to group contigs for secondary clustering — we only cluster within
    the same named taxon (e.g. all contigs assigned to 'Oscillospirales').
    """
    row = tax_df.loc[contig]
    return row["name"] if row["name"] not in NOISE_TAXA else None


def build_secondary_clusters(contigs_secondary, clusters, membership,
                              tax_df, ani_df, depth_df, cluster_id_offset):
    """
    Second-pass clustering for long contigs assigned above genus level
    (order, family, class, phylum).

    Logic:
      - Group by assigned taxon name
      - Within each group: ANI >= ANI_THRESHOLD AND coverage r >= COV_THRESHOLD
      - No FDR correction (groups are smaller; raw p < 0.05 used instead)
      - Multi-member connected components become new clusters
      - Singletons pass through to the recovery module

    Returns updated clusters, membership, list of unplaced contigs,
    and the next cluster_id_offset to keep IDs globally unique.
    """
    log.info(f"Secondary clustering: {len(contigs_secondary):,} above-genus contigs...")

    taxon_groups = defaultdict(list)
    ungrouped = []
    for c in contigs_secondary:
        taxon = get_coarse_taxon(c, tax_df)
        if taxon:
            taxon_groups[taxon].append(c)
        else:
            ungrouped.append(c)

    n_new_clusters = 0
    clustered = set()

    for taxon, members in taxon_groups.items():
        if len(members) == 1:
            continue  # singleton — goes to recovery

        member_set = set(members)
        candidates = ani_df[
            (ani_df["ani"] >= ANI_THRESHOLD) &
            (ani_df["query"].isin(member_set)) &
            (ani_df["ref"].isin(member_set))
        ].copy()

        G = nx.Graph()
        G.add_nodes_from(members)

        for _, row in candidates.iterrows():
            r, p = coverage_corr(row["query"], row["ref"], depth_df)
            if r is not None and r >= COV_THRESHOLD and p < 0.05:
                G.add_edge(row["query"], row["ref"])

        for component in nx.connected_components(G):
            if len(component) == 1:
                continue  # singleton stays unplaced
            cid = f"cluster_{cluster_id_offset:04d}"
            cluster_id_offset += 1
            clusters[cid] = set(component)
            for c in component:
                membership[c] = cid
                clustered.add(c)
            n_new_clusters += 1

    # Contigs not placed into any secondary cluster pass to recovery
    remaining = [c for c in contigs_secondary if c not in clustered] + ungrouped

    log.info(
        f"  -> {n_new_clusters:,} secondary clusters from "
        f"{len(clustered):,} contigs | {len(remaining):,} pass to recovery"
    )
    return clusters, membership, remaining, cluster_id_offset

# ══════════════════════════════════════════════════════════════════════════════
# 6. Recovery module
# ══════════════════════════════════════════════════════════════════════════════

def recover_contigs(contigs_recover, clusters, membership, cluster_reps,
                    tax_df, ani_df, depth_df, absent_from_taxonomy: set):
    """
    Assign short / above-genus / unclassified contigs to existing clusters.

    Speed optimisations vs naive implementation:
      1. Inverted taxonomy index: taxon_name -> [cluster_ids]
         O(1) lookup per taxon name instead of iterating all clusters per contig.
      2. Coverage correlation vectorised over all candidates at once using numpy
         instead of calling scipy spearmanr per pair.
      3. ANI candidates short-circuit: if a contig has an ANI hit, skip taxonomy
         lookup (ANI is a stronger signal; avoids iterating the taxon index).
      4. Pre-extract depth matrix as numpy array for fast row access.
    """
    log.info(f"Recovery module: {len(contigs_recover):,} contigs...")

    rep_to_cluster = {v: k for k, v in cluster_reps.items()}
    rep_set        = set(cluster_reps.values())
    recover_set    = set(contigs_recover)

    # ── ANI index: recovery contig -> [cluster_ids with ANI hit to rep] ───────
    ani_hits = defaultdict(list)
    ani_rec = ani_df[
        (ani_df["ani"] >= ANI_THRESHOLD) &
        (
            (ani_df["query"].isin(recover_set) & ani_df["ref"].isin(rep_set)) |
            (ani_df["ref"].isin(recover_set)   & ani_df["query"].isin(rep_set))
        )
    ]
    for _, row in ani_rec.iterrows():
        contig = row["query"] if row["query"] in recover_set else row["ref"]
        rep    = row["ref"]   if row["query"] in recover_set else row["query"]
        if rep in rep_to_cluster:
            ani_hits[contig].append(rep_to_cluster[rep])

    # ── Inverted taxonomy index: taxon_name -> [cluster_ids] ─────────────────
    # Build once; each contig lookup is then O(len(contig_lineage)) dict lookups
    # instead of O(n_clusters) set intersections.
    taxon_to_clusters = defaultdict(list)
    for cid, rep in cluster_reps.items():
        if rep in tax_df.index:
            for taxon in lineage_set(tax_df.loc[rep]):
                taxon_to_clusters[taxon].append(cid)

    # ── Pre-extract depth matrix as numpy for fast vectorised correlation ─────
    # Build ordered list of rep contigs that exist in depth_df
    rep_list    = [rep for rep in cluster_reps.values() if rep in depth_df.index]
    cid_list    = [rep_to_cluster[rep] for rep in rep_list]
    rep_depth   = depth_df.loc[rep_list].values.astype(np.float64)  # (n_reps, n_samples)
    rep_idx     = {rep: i for i, rep in enumerate(rep_list)}

    def spearman_vectorised(contig_vec, candidate_cids):
        """
        Compute Spearman r between contig_vec and all candidate rep vectors
        at once. Returns dict of cid -> r.
        """
        # Gather indices of candidate reps that are in depth_df
        idxs, valid_cids = [], []
        for cid in candidate_cids:
            rep = cluster_reps[cid]
            if rep in rep_idx:
                idxs.append(rep_idx[rep])
                valid_cids.append(cid)
        if not idxs:
            return {}

        mat = rep_depth[idxs]               # (n_candidates, n_samples)
        n   = mat.shape[1]

        # Rank both the contig vector and all rep rows simultaneously
        def rankdata_2d(x):
            # x shape: (k, n) — rank each row
            ranked = np.zeros_like(x, dtype=np.float64)
            for i in range(x.shape[0]):
                ranked[i] = np.argsort(np.argsort(x[i])).astype(np.float64)
            return ranked

        contig_ranked = np.argsort(np.argsort(contig_vec)).astype(np.float64)
        mat_ranked    = rankdata_2d(mat)

        # Pearson on ranks = Spearman
        c_mean  = contig_ranked.mean()
        m_means = mat_ranked.mean(axis=1, keepdims=True)
        c_dev   = contig_ranked - c_mean
        m_dev   = mat_ranked - m_means

        num     = (m_dev * c_dev).sum(axis=1)
        denom   = np.sqrt((m_dev**2).sum(axis=1)) * np.sqrt((c_dev**2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r_vals = np.where(denom > 0, num / denom, 0.0)

        return {cid: float(r) for cid, r in zip(valid_cids, r_vals)}

    # ── Main recovery loop ────────────────────────────────────────────────────
    recovered_ani_only  = []
    recovered_tax_aided = []
    recovered_ani_tax   = []
    unassigned          = []

    for contig in contigs_recover:
        # ANI candidates
        ani_candidates = set(ani_hits.get(contig, []))

        # Taxonomy candidates — only if no ANI hit (ANI is stronger signal)
        tax_candidates = set()
        if not ani_candidates and contig in tax_df.index:
            contig_lset = lineage_set(tax_df.loc[contig])
            for taxon in contig_lset:
                for cid in taxon_to_clusters.get(taxon, []):
                    tax_candidates.add(cid)

        candidate_clusters = ani_candidates | tax_candidates
        if not candidate_clusters:
            unassigned.append(contig)
            continue

        # Vectorised coverage correlation over all candidates
        if contig not in depth_df.index:
            unassigned.append(contig)
            continue

        contig_vec = depth_df.loc[contig].values.astype(np.float64)
        r_map      = spearman_vectorised(contig_vec, candidate_clusters)

        best_cluster = max(
            (cid for cid, r in r_map.items() if r >= COV_THRESHOLD),
            key=lambda cid: r_map[cid],
            default=None,
        )

        if best_cluster is not None:
            clusters[best_cluster].add(contig)
            membership[contig] = best_cluster
            if best_cluster in ani_candidates and best_cluster in tax_candidates:
                recovered_ani_tax.append(contig)
            elif best_cluster in ani_candidates:
                recovered_ani_only.append(contig)
            else:
                recovered_tax_aided.append(contig)
        else:
            unassigned.append(contig)

    total_recovered = len(recovered_ani_only) + len(recovered_tax_aided) + len(recovered_ani_tax)
    log.info(f"  -> Recovered {total_recovered:,} contigs:")
    log.info(f"       ANI-only path:            {len(recovered_ani_only):,}")
    log.info(f"       Taxonomy-only path:        {len(recovered_tax_aided):,}")
    log.info(f"       Both paths available:      {len(recovered_ani_tax):,}")
    log.info(f"  -> Unassigned:                 {len(unassigned):,}")

    unassigned_absent_tax = [c for c in unassigned if c in absent_from_taxonomy]
    log.info(f"       Of unassigned:")
    log.info(f"         No taxonomy + no ANI hit: {len(unassigned_absent_tax):,}")
    log.info(f"         Had taxonomy, no ANI hit: {len([c for c in unassigned if c not in absent_from_taxonomy and not ani_hits.get(c)]):,}")
    log.info(f"         Had ANI hit(s), cov fail: {len([c for c in unassigned if ani_hits.get(c)]):,}")

    return clusters, membership, unassigned


# ══════════════════════════════════════════════════════════════════════════════
# 7. Output
# ══════════════════════════════════════════════════════════════════════════════

def write_outputs(clusters, cluster_reps, unassigned, records, tax_df, depth_df, outdir):
    cluster_dir    = os.path.join(outdir, "clusters")
    unassigned_dir = os.path.join(outdir, "unassigned")
    os.makedirs(cluster_dir,    exist_ok=True)
    os.makedirs(unassigned_dir, exist_ok=True)

    summary_rows = []

    for cid, members in clusters.items():
        rep = cluster_reps[cid]
        SeqIO.write(
            records[rep] if rep in records else [],
            os.path.join(cluster_dir, f"{cid}_representative.fasta"),
            "fasta",
        )
        SeqIO.write(
            [records[c] for c in members if c in records],
            os.path.join(cluster_dir, f"{cid}_bin.fasta"),
            "fasta",
        )
        rep_row = tax_df.loc[rep] if rep in tax_df.index else None
        for c in members:
            c_row = tax_df.loc[c] if c in tax_df.index else None
            summary_rows.append({
                "contig":      c,
                "cluster":     cid,
                "is_rep":      c == rep,
                "contig_len":  len(records[c].seq) if c in records else None,
                "mean_depth":  float(depth_df.loc[c].mean()) if c in depth_df.index else None,
                "rank":        c_row["rank"]               if c_row is not None else None,
                "name":        c_row["name"]               if c_row is not None else None,
                "lineage":     ";".join(c_row["lineage"])  if c_row is not None else None,
                "rep_name":    rep_row["name"]             if rep_row is not None else None,
                "rep_rank":    rep_row["rank"]             if rep_row is not None else None,
                "rep_lineage": ";".join(rep_row["lineage"]) if rep_row is not None else None,
            })

    SeqIO.write(
        [records[c] for c in unassigned if c in records],
        os.path.join(unassigned_dir, "unassigned_contigs.fasta"),
        "fasta",
    )
    log.info(f"  Wrote {len(unassigned):,} unassigned contigs")

    for c in unassigned:
        c_row = tax_df.loc[c] if c in tax_df.index else None
        summary_rows.append({
            "contig":      c,
            "cluster":     "unassigned",
            "is_rep":      False,
            "contig_len":  len(records[c].seq) if c in records else None,
            "mean_depth":  float(depth_df.loc[c].mean()) if c in depth_df.index else None,
            "rank":        c_row["rank"]              if c_row is not None else None,
            "name":        c_row["name"]              if c_row is not None else None,
            "lineage":     ";".join(c_row["lineage"]) if c_row is not None else None,
            "rep_name":    None,
            "rep_rank":    None,
            "rep_lineage": None,
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(outdir, "cluster_summary.tsv"), sep="\t", index=False
    )
    log.info("  Wrote cluster_summary.tsv")

    with open(os.path.join(outdir, "checkm2_bin_list.txt"), "w") as fh:
        for cid in clusters:
            fh.write(os.path.join(cluster_dir, f"{cid}_bin.fasta") + "\n")
    log.info("  Wrote checkm2_bin_list.txt")



# ══════════════════════════════════════════════════════════════════════════════
# 8. CheckM2 + dRep post-processing
# ══════════════════════════════════════════════════════════════════════════════

def run_checkm2(bin_list_path: str, outdir: str, threads: int) -> pd.DataFrame:
    """
    Run CheckM2 on all cluster bins.
    Returns a DataFrame of quality results indexed by bin name.

    Adds columns to the cluster summary:
      checkm2_completeness, checkm2_contamination, checkm2_quality
      quality = 'high'   (>=90% complete, <=5% contamination)
               'medium' (>=50% complete, <=10% contamination)
               'low'    (everything else — kept but flagged)
    """
    checkm2_dir = os.path.join(outdir, "checkm2")
    os.makedirs(checkm2_dir, exist_ok=True)

    quality_tsv = os.path.join(checkm2_dir, "quality_report.tsv")

    if os.path.exists(quality_tsv):
        log.info(f"  CheckM2 results already exist, skipping: {quality_tsv}")
    else:
        log.info("── CheckM2 completeness/contamination ────────────────────")

        # Read bin paths from the list file
        with open(bin_list_path) as fh:
            bin_paths = [l.strip() for l in fh if l.strip()]

        if not bin_paths:
            log.warning("  No bins in checkm2_bin_list.txt — skipping CheckM2")
            return pd.DataFrame()

        # Write a temporary directory of bins for CheckM2
        bin_input_dir = os.path.join(checkm2_dir, "bins_input")
        os.makedirs(bin_input_dir, exist_ok=True)
        for bp in bin_paths:
            dst = os.path.join(bin_input_dir, os.path.basename(bp))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(bp), dst)

        run([
            "checkm2", "predict",
            "--input",            bin_input_dir,
            "--output-directory", checkm2_dir,
            "--extension",        "fasta",
            "--threads",          str(threads),
            "--force",
        ], "checkm2 predict")

    # Parse results
    if not os.path.exists(quality_tsv):
        log.error(f"CheckM2 output not found: {quality_tsv}")
        return pd.DataFrame()

    qdf = pd.read_csv(quality_tsv, sep="\t")
    # Normalise bin name: strip extension, match cluster IDs
    qdf["cluster"] = qdf["Name"].apply(
        lambda x: os.path.splitext(x)[0].replace("_bin", "")
    )

    def quality_tier(row):
        c, x = row["Completeness"], row["Contamination"]
        if c >= 90 and x <= 5:
            return "high"
        if c >= CHECKM2_MIN_COMPLETENESS and x <= CHECKM2_MAX_CONTAMINATION:
            return "medium"
        return "low"

    qdf["quality"] = qdf.apply(quality_tier, axis=1)

    n_high   = (qdf["quality"] == "high").sum()
    n_medium = (qdf["quality"] == "medium").sum()
    n_low    = (qdf["quality"] == "low").sum()
    log.info(f"  CheckM2 done: {n_high} high-quality | {n_medium} medium | {n_low} low (flagged)")

    return qdf


def run_drep(clusters_dir: str, checkm2_df: pd.DataFrame,
             outdir: str, threads: int) -> str:
    """
    Dereplicate medium+high quality bins with dRep.

    dRep requires a CheckM-format genome info CSV:
      genome,completeness,contamination,strain_heterogeneity
    We derive this from the CheckM2 output (strain heterogeneity set to 0,
    which is conservative but acceptable — dRep uses it for scoring only).

    Returns path to the dereplicated bins directory.
    """
    drep_dir = os.path.join(outdir, "drep")
    derep_bins_dir = os.path.join(drep_dir, "dereplicated_genomes")

    if os.path.exists(derep_bins_dir) and os.listdir(derep_bins_dir):
        log.info(f"  dRep output already exists, skipping: {derep_bins_dir}")
        return derep_bins_dir

    log.info("── dRep dereplication ────────────────────────────────────────")

    if checkm2_df.empty:
        log.warning("  No CheckM2 results available — skipping dRep")
        return ""

    # Filter to medium+high quality bins
    passing = checkm2_df[checkm2_df["quality"].isin(["high", "medium"])].copy()
    if passing.empty:
        log.warning("  No bins pass quality thresholds — skipping dRep")
        return ""

    log.info(f"  {len(passing):,} bins pass quality thresholds for dereplication")

    # Build genome info CSV for dRep --genomeInfo
    genome_info_path = os.path.join(drep_dir, "genome_info.csv")
    os.makedirs(drep_dir, exist_ok=True)

    bin_paths = []
    genome_info_rows = []
    for _, row in passing.iterrows():
        # Reconstruct bin filename from Name column
        bin_name = row["Name"] if row["Name"].endswith(".fasta") else row["Name"] + ".fasta"
        bin_path = os.path.join(clusters_dir, bin_name)
        if not os.path.exists(bin_path):
            # Try _bin.fasta suffix pattern
            bin_path = os.path.join(clusters_dir, bin_name.replace(".fasta", "_bin.fasta"))
        if not os.path.exists(bin_path):
            log.warning(f"  Could not find bin file for {row['Name']}, skipping")
            continue
        bin_paths.append(bin_path)
        genome_info_rows.append({
            "genome":               bin_path,
            "completeness":         row["Completeness"],
            "contamination":        row["Contamination"],
            "strain_heterogeneity": 0,
        })

    if not genome_info_rows:
        log.warning("  No bin files found for passing bins — skipping dRep")
        return ""

    pd.DataFrame(genome_info_rows).to_csv(genome_info_path, index=False)

    run([
        "dRep", "dereplicate",
        drep_dir,
        "-g",            ] + bin_paths + [
        "--genomeInfo",  genome_info_path,
        "-pa",           "0.9",          # MASH pre-filter ANI (90%)
        "-sa",           str(DREP_ANI / 100),   # secondary ANI threshold (0.95)
        "-nc",           "0.30",         # min fraction of genome aligned
        "-comp",         str(CHECKM2_MIN_COMPLETENESS),
        "-con",          str(CHECKM2_MAX_CONTAMINATION),
        "-p",            str(threads),
    ], "dRep dereplicate")

    n_derep = len(os.listdir(derep_bins_dir)) if os.path.exists(derep_bins_dir) else 0
    log.info(f"  dRep done: {n_derep} dereplicated bins -> {derep_bins_dir}")
    return derep_bins_dir


def merge_checkm2_into_summary(summary_path: str, checkm2_df: pd.DataFrame):
    """
    Add CheckM2 completeness, contamination, and quality columns to
    cluster_summary.tsv in-place. Bins not in CheckM2 results (unassigned,
    low-quality flagged) get NaN for numeric cols and 'low' for quality.
    """
    if checkm2_df.empty:
        return

    summary = pd.read_csv(summary_path, sep="\t")

    # Build lookup: cluster_id -> quality row
    # CheckM2 Name column is like "cluster_0001_bin"
    qmap = {}
    for _, row in checkm2_df.iterrows():
        # Extract cluster id from bin name e.g. "cluster_0001_bin" -> "cluster_0001"
        cid = row["Name"].replace("_bin", "")
        qmap[cid] = row

    def get_q(cid, field, default):
        r = qmap.get(cid)
        return r[field] if r is not None else default

    summary["checkm2_completeness"]  = summary["cluster"].apply(
        lambda c: get_q(c, "Completeness", float("nan")))
    summary["checkm2_contamination"] = summary["cluster"].apply(
        lambda c: get_q(c, "Contamination", float("nan")))
    summary["checkm2_quality"]       = summary["cluster"].apply(
        lambda c: get_q(c, "quality", "not_assessed"))

    summary.to_csv(summary_path, sep="\t", index=False)
    log.info("  Updated cluster_summary.tsv with CheckM2 quality columns")


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def ckpt_path(ckpt_dir: str, name: str, ext: str) -> str:
    return os.path.join(ckpt_dir, f"{name}.{ext}")


def ckpt_exists(ckpt_dir: str, name: str, ext: str) -> bool:
    return os.path.exists(ckpt_path(ckpt_dir, name, ext))


def save_parquet(df: pd.DataFrame, ckpt_dir: str, name: str):
    """Save a DataFrame as parquet. Lists are JSON-encoded per cell first."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(json.dumps)
    path = ckpt_path(ckpt_dir, name, "parquet")
    df.to_parquet(path)
    log.info(f"  Checkpoint saved: {path}")


def load_parquet(ckpt_dir: str, name: str, list_cols: list[str] = None) -> pd.DataFrame:
    """Load a parquet checkpoint, decoding any JSON-encoded list columns."""
    path = ckpt_path(ckpt_dir, name, "parquet")
    df = pd.read_parquet(path)
    for col in (list_cols or []):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    log.info(f"  Checkpoint loaded: {path}")
    return df


def save_json(obj, ckpt_dir: str, name: str):
    """Save a dict/list as JSON. Sets are converted to lists."""
    def default(o):
        if isinstance(o, set):
            return list(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")
    path = ckpt_path(ckpt_dir, name, "json")
    with open(path, "w") as fh:
        json.dump(obj, fh, default=default)
    log.info(f"  Checkpoint saved: {path}")


def load_json(ckpt_dir: str, name: str):
    path = ckpt_path(ckpt_dir, name, "json")
    with open(path) as fh:
        obj = json.load(fh)
    log.info(f"  Checkpoint loaded: {path}")
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Taxonomy-aware metagenomic contig clustering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Override globals
    global MIN_LEN, ANI_THRESHOLD, COV_THRESHOLD
    
    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument("--fasta",   required=True,
                        help="Assembly FASTA (all contigs)")
    parser.add_argument("--outdir",  required=True,
                        help="Output directory")

    # ── Pre-computed inputs (each makes the corresponding step optional) ──────
    parser.add_argument("--taxonomy", default=None,
                        help="Pre-computed taxonomy file. If omitted, MMseqs2 is run.")
    parser.add_argument("--taxonomy-format", default="mmseqs2",
                        choices=["mmseqs2", "taxometer"],
                        help="Format of --taxonomy file")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="[Taxometer] Min per-level confidence score (0-1)")
    parser.add_argument("--ani", default=None,
                        help="Pre-computed skani triangle TSV. If omitted, skani is run.")
    parser.add_argument("--depth", default=None,
                        help="Pre-computed depth matrix TSV. If omitted, jgi is run.")

    # ── Required only when running external tools ─────────────────────────────
    parser.add_argument("--bams", nargs="+", default=None,
                        help="BAM files (sorted, one per sample). Required unless --depth supplied.")
    parser.add_argument("--mmseqs-db", default=None,
                        help="MMseqs2 taxonomy database path. Required unless --taxonomy supplied.")

    # ── Tunable parameters ────────────────────────────────────────────────────
    parser.add_argument("--threads",       type=int,   default=MMSEQS_THREADS,
                        help="Threads for MMseqs2 and skani")
    parser.add_argument("--min-len",       type=int,   default=MIN_LEN,
                        help="Min contig length (bp) for main clustering")
    parser.add_argument("--ani-threshold", type=float, default=ANI_THRESHOLD,
                        help="ANI threshold (%%) for clustering and recovery")
    parser.add_argument("--cov-threshold",   type=float, default=COV_THRESHOLD,
                        help="Spearman r threshold for coverage correlation")

    # ── Post-processing ───────────────────────────────────────────────────────
    parser.add_argument("--skip-checkm2",    action="store_true",
                        help="Skip CheckM2 completeness/contamination assessment "
                             "(also skips dRep)")
    parser.add_argument("--skip-drep",       action="store_true",
                        help="Skip dRep dereplication (CheckM2 still runs)")
    parser.add_argument("--checkm2-db",      default=None,
                        help="Path to CheckM2 diamond database directory. "
                             "Optional if CHECKM2DB is already set in environment.")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint-dir",  default=None,
                        help="Directory to store/load intermediate checkpoints. "
                             "Defaults to <outdir>/checkpoints/. Pass an explicit "
                             "path to share checkpoints across runs with different "
                             "--outdir values.")
    parser.add_argument("--no-cache",        action="store_true",
                        help="Ignore existing checkpoints and recompute everything.")

    args = parser.parse_args()

    # ── Validate argument combinations ────────────────────────────────────────
    if not args.taxonomy and not args.mmseqs_db:
        parser.error("--mmseqs-db is required when --taxonomy is not supplied")
    if not args.depth and not args.bams:
        parser.error("--bams is required when --depth is not supplied")

    os.makedirs(args.outdir, exist_ok=True)

    # Override globals
    MIN_LEN       = args.min_len
    ANI_THRESHOLD = args.ani_threshold
    COV_THRESHOLD = args.cov_threshold

    # ── Tool availability check ───────────────────────────────────────────────
    needed = tools_needed_for(args)
    if needed:
        check_tools(needed)

    # ── Step 1: taxonomy ──────────────────────────────────────────────────────
    taxonomy_path = args.taxonomy
    if not taxonomy_path:
        taxonomy_path = run_mmseqs_taxonomy(
            args.fasta, args.mmseqs_db, args.outdir, args.threads
        )
        # MMseqs2 output is always mmseqs2 format
        args.taxonomy_format = "mmseqs2"

    # ── Step 2: ANI ───────────────────────────────────────────────────────────
    ani_path = args.ani
    if not ani_path:
        ani_path = run_skani(args.fasta, args.outdir, args.threads)

    # ── Step 3: depth ─────────────────────────────────────────────────────────
    depth_path = args.depth
    if not depth_path:
        depth_path = run_depth(args.bams, args.outdir)

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir or os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    use_cache = not args.no_cache

    def cached(name, ext):
        """Return True if a valid checkpoint exists and caching is enabled."""
        return use_cache and ckpt_exists(ckpt_dir, name, ext)

    # ── Load: depth ───────────────────────────────────────────────────────────
    log.info("── Loading inputs ────────────────────────────────────────────")
    if cached("depth_df", "parquet"):
        depth_df = load_parquet(ckpt_dir, "depth_df")
    else:
        depth_df = parse_depth(depth_path)
        save_parquet(depth_df, ckpt_dir, "depth_df")

    # ── Load: ANI ─────────────────────────────────────────────────────────────
    if cached("ani_df", "parquet"):
        ani_df = load_parquet(ckpt_dir, "ani_df")
    else:
        ani_df = parse_ani(ani_path)
        save_parquet(ani_df, ckpt_dir, "ani_df")

    # ── Load: FASTA (always needed for SeqRecord objects) ─────────────────────
    records = parse_fasta(args.fasta)

    # ── Load: taxonomy ────────────────────────────────────────────────────────
    if cached("tax_df", "parquet"):
        tax_df = load_parquet(ckpt_dir, "tax_df", list_cols=["lineage", "scores"])
        absent_from_taxonomy = load_json(ckpt_dir, "absent_from_taxonomy")
    else:
        if args.taxonomy_format == "taxometer":
            tax_df = parse_taxonomy_taxometer(taxonomy_path, min_score=args.min_score)
        else:
            tax_df = parse_taxonomy_mmseqs2(taxonomy_path)
        tax_df, absent_from_taxonomy = build_taxonomy_df(tax_df, list(records.keys()))
        save_parquet(tax_df, ckpt_dir, "tax_df")
        save_json(absent_from_taxonomy, ckpt_dir, "absent_from_taxonomy")

    absent_set = set(absent_from_taxonomy)

    # ── Partition into three tiers ────────────────────────────────────────────
    # Tier 1 (main):      long (>= MIN_LEN) + genus or species assigned
    # Tier 2 (secondary): long (>= MIN_LEN) + above-genus assigned (order/family/class/phylum)
    # Tier 3 (recovery):  short OR unclassified — rescued by ANI + coverage only
    contigs_main, contigs_secondary, contigs_recover = [], [], []
    for c in records:
        row     = tax_df.loc[c]
        is_long = len(records[c].seq) >= MIN_LEN
        rank    = row["rank"]
        if is_long and rank in MAIN_RANKS:
            contigs_main.append(c)
        elif is_long and rank in SECONDARY_RANKS:
            contigs_secondary.append(c)
        else:
            contigs_recover.append(c)

    log.info(
        f"Partitioned: {len(contigs_main):,} main (genus/species) "
        f"| {len(contigs_secondary):,} secondary (order/family/class/phylum) "
        f"| {len(contigs_recover):,} recovery (short/unclassified)"
    )

    # ── Clustering (all three tiers, single checkpoint) ───────────────────────
    log.info("── Clustering ────────────────────────────────────────────────")
    if cached("clusters", "json") and cached("membership", "json") and cached("cluster_reps", "json"):
        clusters_raw  = load_json(ckpt_dir, "clusters")
        clusters      = {cid: set(members) for cid, members in clusters_raw.items()}
        membership    = load_json(ckpt_dir, "membership")
        cluster_reps  = load_json(ckpt_dir, "cluster_reps")
        unassigned    = load_json(ckpt_dir, "unassigned")
        log.info(f"  Loaded {len(clusters):,} clusters from checkpoint")
    else:
        # Tier 1: Main clustering
        clusters, membership = build_main_clusters(contigs_main, tax_df, ani_df, depth_df)
        cluster_id_offset = len(clusters)
        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        # Tier 2: Secondary clustering (above-genus)
        clusters, membership, contigs_recover_extra, cluster_id_offset = build_secondary_clusters(
            contigs_secondary, clusters, membership,
            tax_df, ani_df, depth_df, cluster_id_offset
        )
        contigs_recover = contigs_recover + contigs_recover_extra
        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        # Tier 3: Recovery
        clusters, membership, unassigned = recover_contigs(
            contigs_recover, clusters, membership, cluster_reps,
            tax_df, ani_df, depth_df, absent_set,
        )
        cluster_reps = {
            cid: representative(members, records, depth_df)
            for cid, members in clusters.items()
        }

        # Save clustering checkpoint
        save_json({cid: list(m) for cid, m in clusters.items()}, ckpt_dir, "clusters")
        save_json(membership,   ckpt_dir, "membership")
        save_json(cluster_reps, ckpt_dir, "cluster_reps")
        save_json(unassigned,   ckpt_dir, "unassigned")

    # ── Write ─────────────────────────────────────────────────────────────────
    log.info("── Writing outputs ───────────────────────────────────────────")
    write_outputs(clusters, cluster_reps, unassigned, records, tax_df, depth_df, args.outdir)

    # ── CheckM2 ───────────────────────────────────────────────────────────────
    checkm2_df = pd.DataFrame()
    if not args.skip_checkm2:
        if args.checkm2_db:
            os.environ["CHECKM2DB"] = args.checkm2_db
        elif "CHECKM2DB" not in os.environ:
            log.warning(
                "CHECKM2DB environment variable not set and --checkm2-db not supplied. "
                "CheckM2 may fail if the database path is not configured. "
                "Set it with: export CHECKM2DB=/path/to/checkm2/db  "
                "or pass --checkm2-db."
            )
        bin_list = os.path.join(args.outdir, "checkm2_bin_list.txt")
        checkm2_df = run_checkm2(bin_list, args.outdir, args.threads)

        # Merge quality columns into cluster_summary.tsv
        summary_path = os.path.join(args.outdir, "cluster_summary.tsv")
        merge_checkm2_into_summary(summary_path, checkm2_df)
    else:
        log.info("Skipping CheckM2 (--skip-checkm2)")

    # ── dRep ──────────────────────────────────────────────────────────────────
    drep_out = ""
    if not args.skip_drep:
        if args.skip_checkm2:
            log.warning(
                "dRep requires CheckM2 quality scores for scoring. "
                "Skipping dRep because --skip-checkm2 was set."
            )
        elif checkm2_df.empty:
            log.warning("No CheckM2 results available — skipping dRep")
        else:
            clusters_dir = os.path.join(args.outdir, "clusters")
            drep_out = run_drep(clusters_dir, checkm2_df, args.outdir, args.threads)
    else:
        log.info("Skipping dRep (--skip-drep)")

    log.info("═" * 60)
    log.info(f"Total clusters:         {len(clusters):,}")
    log.info(f"Total assigned contigs: {sum(len(v) for v in clusters.values()):,}")
    log.info(f"Unassigned contigs:     {len(unassigned):,}")
    if not checkm2_df.empty:
        n_hq = (checkm2_df["quality"] == "high").sum()
        n_mq = (checkm2_df["quality"] == "medium").sum()
        n_lq = (checkm2_df["quality"] == "low").sum()
        log.info(f"Bin quality:            {n_hq} high | {n_mq} medium | {n_lq} low (flagged in summary)")
    if drep_out:
        n_derep = len([f for f in os.listdir(drep_out) if f.endswith(".fasta")])
        log.info(f"Dereplicated bins:      {n_derep} -> {drep_out}")
    log.info(f"Output:                 {args.outdir}/")
    log.info("Done.")


if __name__ == "__main__":
    main()
