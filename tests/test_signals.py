"""Unit tests for tritanc.signals — TNF, Spearman rho, permutation fallback, Leiden clustering."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from tritanc.signals import (
    _revcomp,
    _canonical,
    compute_tnf,
    tnf_similarity,
    _compute_rho,
    _analytical_pvalues,
    _permutation_pvalues,
    vectorised_spearman_pairs,
    leiden_communities,
    build_cluster_centroids,
    representative,
)


class TestRevcomp:
    def test_basic(self):
        assert _revcomp("ACGT") == "ACGT"
        assert _revcomp("AAAA") == "TTTT"

    def test_palindrome(self):
        # ACGT reverse complement is ACGT itself
        assert _revcomp("ACGT") == "ACGT"


class TestCanonical:
    def test_canonical_returns_smaller(self):
        k = "ACGT"
        c = _canonical(k)
        rc = _revcomp(k)
        assert c == min(k, rc)


class TestTNF:
    def test_compute_tnf_short_skipped(self):
        """Contigs shorter than min_len should be skipped."""
        records = {"short": type("SeqRecord", (), {"seq": "ACGT"})()}
        tnf = compute_tnf(records, min_len=10)
        assert "short" not in tnf

    def test_compute_tnf_produces_vectors(self):
        """Long contigs should produce TNF vectors."""
        seq = "ACGT" * 500  # 2000 bp
        records = {"long": type("SeqRecord", (), {"seq": seq})()}
        tnf = compute_tnf(records, min_len=10)
        assert "long" in tnf
        assert len(tnf["long"]) == 136  # number of canonical 4-mers

    def test_tnf_similarity_identical(self):
        """Identical vectors should have cosine similarity 1.0."""
        v = np.zeros(136, dtype=np.float32)
        v[0] = 1.0
        tnf = {"a": v, "b": v.copy()}
        assert tnf_similarity("a", "b", tnf) == pytest.approx(1.0)

    def test_tnf_similarity_none_for_missing(self):
        tnf = {"a": np.zeros(10)}
        assert tnf_similarity("a", "missing", tnf) is None


class TestSpearman:
    def test_compute_rho_perfect_positive(self):
        q = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        r = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        rho = _compute_rho(q, r)
        assert rho[0] == pytest.approx(1.0)

    def test_compute_rho_perfect_negative(self):
        q = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
        r = np.array([[5, 4, 3, 2, 1]], dtype=np.float64)
        rho = _compute_rho(q, r)
        assert rho[0] == pytest.approx(-1.0)

    def test_analytical_pvalues_symmetric(self):
        """P-values should be symmetric around rho=0."""
        rho = np.array([0.5, -0.5, 0.0])
        pvals = _analytical_pvalues(rho, n_samples=20)
        assert pvals[0] == pytest.approx(pvals[1])

    def test_analytical_pvalues_perfect_correlation(self):
        """rho=1 or -1 should give p=0."""
        rho = np.array([1.0, -1.0])
        pvals = _analytical_pvalues(rho, n_samples=20)
        assert pvals[0] == pytest.approx(0.0)
        assert pvals[1] == pytest.approx(0.0)


class TestPermutationPvalues:
    def test_permutation_pvalues_range(self):
        """Permutation p-values should be in (0, 1]."""
        n_pairs = 10
        n_samples = 20
        rho_obs = np.random.default_rng(42).normal(0, 0.5, n_pairs)
        q_ranked = np.random.default_rng(42).standard_normal((n_pairs, n_samples))
        r_ranked = np.random.default_rng(43).standard_normal((n_pairs, n_samples))
        q_ranked = rankdata(q_ranked, axis=1)
        r_ranked = rankdata(r_ranked, axis=1)

        rng = np.random.default_rng(seed=42)
        pvals = _permutation_pvalues(rho_obs, q_ranked, r_ranked, 99, rng)
        assert all(0 < p <= 1 for p in pvals)


class TestVectorisedSpearman:
    def test_empty_candidates(self):
        from tritanc.config import get_adaptive_thresholds
        thresholds = get_adaptive_thresholds(n_samples=10)
        df = pd.DataFrame(columns=["query", "ref"])
        depth_df = pd.DataFrame()
        result = vectorised_spearman_pairs(df, depth_df, thresholds)
        assert "cov_r" in result.columns
        assert "pval" in result.columns

    def test_with_data(self):
        from tritanc.config import get_adaptive_thresholds
        thresholds = get_adaptive_thresholds(n_samples=15)
        candidates = pd.DataFrame({
            "query": ["a", "b"],
            "ref": ["c", "d"],
        })
        depth_df = pd.DataFrame(
            np.random.default_rng(42).random((4, 15)),
            index=["a", "b", "c", "d"],
        )
        result = vectorised_spearman_pairs(candidates, depth_df, thresholds)
        assert len(result) == 2
        assert "cov_r" in result.columns
        assert "pval" in result.columns


class TestLeiden:
    def test_leiden_simple_graph(self):
        """Simple graph with two clear communities should split."""
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])       # cluster A
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])       # cluster B
        G.add_edge(2, 3)                                   # weak link
        communities = leiden_communities(G, resolution=1.0)
        assert len(communities) >= 2

    def test_leiden_empty_graph(self):
        import networkx as nx
        G = nx.Graph()
        assert leiden_communities(G, resolution=1.0) == []

    def test_leiden_single_node(self):
        import networkx as nx
        G = nx.Graph()
        G.add_node("a")
        comms = leiden_communities(G, resolution=1.0)
        assert len(comms) == 1 and comms[0] == {"a"}


class TestCentroids:
    def test_build_cluster_centroids(self):
        clusters = {"c1": {"a", "b"}, "c2": {"c"}}
        depth_df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            index=["a", "b", "c"],
        )
        cids, centroids = build_cluster_centroids(clusters, depth_df)
        assert len(cids) == 2
        # c1 centroid = mean of a and b rows
        np.testing.assert_array_almost_equal(centroids[0], [2.5, 3.5, 4.5])

    def test_build_cluster_centroids_skips_missing(self):
        clusters = {"c1": {"a", "x"}}  # x not in depth_df
        depth_df = pd.DataFrame(np.array([[1, 2]]), index=["a"])
        cids, centroids = build_cluster_centroids(clusters, depth_df)
        assert len(cids) == 1


class TestRepresentative:
    def test_representative_prefers_longer(self):
        class MockRec:
            def __init__(self, seq):
                self.seq = seq

        records = {
            "short": MockRec("ACGT"),
            "long": MockRec("ACGT" * 100),
        }
        depth_df = pd.DataFrame({"x": [1.0, 1.0]})
        depth_df.index = ["short", "long"]
        rep = representative({"short", "long"}, records, depth_df)
        assert rep == "long"
