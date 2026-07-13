"""Graph building: multimodal edge construction and scoring."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from tritanc.config import AdaptiveThresholds

log = logging.getLogger(__name__)


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


# Import dependencies from signals and config
from tritanc.signals import vectorised_spearman_pairs, tnf_similarity
from tritanc.config import NOISE_TAXA

__all__ = [
    "_build_multimodal_graph",
]
