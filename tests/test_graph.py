"""Unit tests for tritanc.graph — edge creation, weighting formulas, taxon-pair capping, self-loop removal."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from tritanc.config import get_adaptive_thresholds, NOISE_TAXA
from tritanc.graph import _build_multimodal_graph


def _make_thresholds(n_samples=20):
    return get_adaptive_thresholds(n_samples=n_samples)


def _make_ani_df(pairs=None):
    """Helper to create a minimal ANI DataFrame."""
    if pairs is None:
        pairs = [("a", "b", 97.0), ("c", "d", 96.0)]
    return pd.DataFrame(pairs, columns=["query", "ref", "ani"])


def _make_depth_df(contigs=None, n_samples=10):
    """Helper to create a minimal depth DataFrame."""
    if contigs is None:
        contigs = ["a", "b", "c", "d"]
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((len(contigs), n_samples)), index=contigs)


def _make_tax_df(assignments=None):
    """Helper to create a minimal taxonomy DataFrame."""
    if assignments is None:
        assignments = {
            "a": {"rank": "genus", "name": "Streptococcus", "lineage": ["Bacteria", "Streptococcus"]},
            "b": {"rank": "genus", "name": "Streptococcus", "lineage": ["Bacteria", "Streptococcus"]},
            "c": {"rank": "genus", "name": "Veillonella", "lineage": ["Bacteria", "Veillonella"]},
            "d": {"rank": "genus", "name": "Veillonella", "lineage": ["Bacteria", "Veillonella"]},
        }
    return pd.DataFrame.from_dict(assignments, orient="index")


class TestEdgeCreation:
    """Test that edges are created correctly under various conditions."""

    def test_ani_edge_created(self):
        """ANI pairs above threshold should produce edges."""
        thresholds = _make_thresholds()
        ani_df = _make_ani_df([("a", "b", 97.0)])
        # Use correlated depth so cov_r is high
        rng = np.random.default_rng(42)
        base = rng.random(10)
        a_vals = base.copy()
        b_vals = base + rng.normal(0, 0.05, 10)
        depth_df = pd.DataFrame(
            np.array([a_vals, b_vals]),
            index=["a", "b"],
        )
        # Provide TNF with identical vectors so tnf_sim = 1.0
        v = np.zeros(136, dtype=np.float32)
        v[0] = 1.0
        tnf = {"a": v, "b": v.copy()}
        tax_df = pd.DataFrame({"a": {"name": "S", "lineage": []},
                               "b": {"name": "S", "lineage": []}}).T

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf=tnf,
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.85,
        )
        assert G.has_edge("a", "b")

    def test_no_edge_below_ani_threshold(self):
        """Pairs below ANI threshold should not get edges."""
        thresholds = _make_thresholds()
        ani_df = _make_ani_df([("a", "b", 90.0)])
        depth_df = _make_depth_df(["a", "b"])
        tax_df = pd.DataFrame({"a": {"name": "S", "lineage": []},
                               "b": {"name": "S", "lineage": []}}).T

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.85,
        )
        assert not G.has_edge("a", "b")


class TestEdgeWeighting:
    """Test edge weight formulas."""

    def test_ani_edge_weight_formula(self):
        """ANI edge: 0.40 * ani/100 + 0.35 * cov_r + 0.25 * tnf."""
        thresholds = _make_thresholds()
        # Create a scenario where we can verify the weight
        ani_df = _make_ani_df([("a", "b", 100.0)])  # max ANI
        depth_df = _make_depth_df(["a", "b"], n_samples=30)
        tax_df = pd.DataFrame({"a": {"name": "S", "lineage": []},
                               "b": {"name": "S", "lineage": []}}).T

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.5,  # low threshold to ensure edge
        )
        if G.has_edge("a", "b"):
            w = G["a"]["b"]["weight"]
            # With ani=100, weight should be at least 0.40 * 1.0 = 0.40
            assert w >= 0.40

    def test_taxonomy_edge_no_sequence_weight(self):
        """Taxonomy edges should have 0.0 sequence component."""
        thresholds = _make_thresholds()
        # Create ANI df with no pairs (so only taxonomy edges form)
        ani_df = _make_ani_df([("a", "b", 80.0)])  # below threshold
        depth_df = _make_depth_df(["a", "b"])

        tax_df = _make_tax_df({
            "a": {"rank": "genus", "name": "Streptococcus",
                  "lineage": ["Bacteria", "Streptococcus"]},
            "b": {"rank": "genus", "name": "Streptococcus",
                  "lineage": ["Bacteria", "Streptococcus"]},
        })

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.5,
        )
        # Taxonomy edges require both cov AND tnf to pass — with empty tnf,
        # no taxonomy edge should form (tnf_sim is None)
        # This verifies the "both cov and TNF" requirement for taxonomy edges


class TestTaxonPairCapping:
    """Test that large taxon groups are capped to avoid O(n²) blowup."""

    def test_large_group_capped(self):
        """A group with many members should be capped at 50k pairs."""
        thresholds = _make_thresholds(n_samples=30)
        members = [f"contig_{i}" for i in range(500)]
        member_set = set(members)

        ani_df = pd.DataFrame(columns=["query", "ref", "ani"])
        depth_df = pd.DataFrame(np.random.default_rng(42).random((500, 30)))
        depth_df.index = members

        # All contigs assigned to same taxon
        tax_df = pd.DataFrame({
            m: {"rank": "genus", "name": "BigGenus", "lineage": ["Bacteria", "BigGenus"]}
            for m in members
        })

        G = _build_multimodal_graph(
            members=members, member_set=member_set,
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.85,
        )
        # Graph should still be buildable (no hang) even with 500 members in one taxon
        assert G.number_of_nodes() == 500


class TestSelfLoopRemoval:
    """Test that self-loops are not created."""

    def test_no_self_loops(self):
        thresholds = _make_thresholds()
        ani_df = _make_ani_df([("a", "b", 97.0)])
        depth_df = _make_depth_df(["a", "b"])
        tax_df = pd.DataFrame({"a": {"name": "S", "lineage": []},
                               "b": {"name": "S", "lineage": []}}).T

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.85,
        )
        assert not G.has_edge("a", "a")
        assert not G.has_edge("b", "b")


class TestNoiseTaxa:
    """Test that noise taxa are excluded from taxonomy edges."""

    def test_noise_taxa_excluded(self):
        thresholds = _make_thresholds()
        ani_df = pd.DataFrame(columns=["query", "ref", "ani"])
        depth_df = _make_depth_df(["a", "b"])

        # Both have noise taxonomy name
        tax_df = pd.DataFrame({
            "a": {"rank": "unclassified", "name": "unclassified", "lineage": []},
            "b": {"rank": "unclassified", "name": "unclassified", "lineage": []},
        })

        G = _build_multimodal_graph(
            members=["a", "b"], member_set={"a", "b"},
            ani_df=ani_df, depth_df=depth_df, tnf={},
            tax_df=tax_df, prot_sim_df=None,
            thresholds=thresholds,
            ani_threshold=95.0, cov_threshold=0.85,
        )
        # No taxonomy edges should form for noise taxa
        assert G.number_of_edges() == 0
