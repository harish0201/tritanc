"""Tier 1–3 + Tier 4–5 clustering logic."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if TYPE_CHECKING:
    from tritanc.config import AdaptiveThresholds

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1 — Main clustering (genus / species)
# ═════════════════════════════════════════════════════════════════════════════

def build_main_clusters(
    contigs_main: list[str],
    tax_df: pd.DataFrame,
    ani_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    tnf: dict,
    thresholds: AdaptiveThresholds,
    leiden_res: float = 3.5,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict]:
    """Build Tier 1 clusters from genus/species-level contigs using multimodal graph.

    Uses ANI + protein similarity + taxonomy edges with hybrid gating.
    Leiden community detection at resolution=3.5 for fine splits of closely
    related oral taxa.

    Returns (clusters dict, membership dict).
    """
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
    leiden_res: float = 2.0,
    prot_sim_df: pd.DataFrame | None = None,
) -> tuple[dict, dict, list[str], int]:
    """Build Tier 2 clusters from above-genus contigs.

    Groups contigs by shared taxon name, then runs multimodal graph + Leiden
    on each group. Single-member groups pass through to recovery unchanged.

    Returns (clusters, membership, remaining_unassigned, new_cluster_id_offset).
    """
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


# Import dependencies from signals and config
from tritanc.signals import (
    build_cluster_centroids, tnf_similarity, leiden_communities,
    vectorised_spearman_pairs, representative,
)
from tritanc.graph import _build_multimodal_graph
import networkx as nx
from tritanc.config import NOISE_TAXA

__all__ = [
    "build_main_clusters",
    "build_secondary_clusters",
    "cluster_unassigned",
    "coverage_only_recovery",
]
