"""Recovery-specific logic: Tier 3 recovery and lineage-aware taxonomy fallback."""

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


def build_lineage_tax_index(cluster_reps: dict, tax_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build a mapping from lineage taxon names to cluster IDs that share them.

    Used by Tier 3 recovery to find candidate clusters for unassigned contigs
    based on shared taxonomy at any lineage level (not just leaf).
    """
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
    max_cands: int = 50,
) -> set[str]:
    """Find candidate cluster IDs for a contig based on lineage taxonomy.

    Walks the contig's lineage from leaf to root, returning clusters that
    share any of its taxon names. Caps at max_cands candidates per contig.
    """
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
    """Tier 3 recovery: assign unassigned contigs to existing clusters.

    Uses ANI, protein similarity, and taxonomy as candidate sources,
    then scores each candidate by a weighted combination of sequence identity,
    coverage correlation (Spearman r), and TNF cosine similarity.

    Returns updated clusters, membership, and remaining unassigned contigs.
    """
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


# Import dependencies from signals and config
from tritanc.signals import build_cluster_centroids, tnf_similarity
from tritanc.config import NOISE_TAXA

__all__ = [
    "recover_contigs",
    "build_lineage_tax_index",
    "_taxonomy_candidates",
]
