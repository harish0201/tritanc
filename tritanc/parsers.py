"""Input parsers for FASTA, taxonomy, ANI, depth, and protein similarity data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from Bio import SeqIO

if TYPE_CHECKING:
    from tritanc.config import AdaptiveThresholds

log = logging.getLogger(__name__)


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


# Import constants from config
from tritanc.config import CANONICAL_RANKS, TAXOMETER_PREFIX

__all__ = [
    "parse_fasta",
    "_empty_tax_row",
    "parse_taxonomy_mmseqs2",
    "_strip_rank_prefix",
    "parse_taxonomy_taxometer",
    "build_taxonomy_df",
    "parse_protein_similarity",
    "parse_depth",
    "parse_ani",
]
