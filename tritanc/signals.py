"""Core algorithms: TNF, Spearman correlation, Leiden clustering, centroids."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from scipy.stats import rankdata, t as t_dist
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from tritanc.config import AdaptiveThresholds

log = logging.getLogger(__name__)


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


def compute_tnf(records: dict, min_len: int = 1000) -> dict:
    """Compute TNF (k-mer frequency) vectors for all contigs >= min_len.

    Uses canonical reverse-complement collapsed 4-mers.
    Contigs shorter than min_len are skipped (too noisy for TNF).
    """
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


__all__ = [
    "representative",
    "_revcomp", "_canonical",
    "compute_tnf", "tnf_similarity",
    "_compute_rho", "_analytical_pvalues", "_permutation_pvalues",
    "vectorised_spearman_pairs",
    "leiden_communities",
    "build_cluster_centroids",
]
