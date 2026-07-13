"""Unit tests for tritanc.config — adaptive thresholds and tool-check helpers."""

import pytest
import shutil

from tritanc.config import (
    AdaptiveThresholds,
    get_adaptive_thresholds,
    TOOL_HINTS,
    NOISE_TAXA,
    CANONICAL_RANKS,
    MAIN_RANKS,
    SECONDARY_RANKS,
)
from tritanc.tools import check_tools, tools_needed_for


class TestAdaptiveThresholds:
    """Test adaptive threshold computation for different sample counts."""

    def test_many_samples(self):
        """30+ samples: strict thresholds, FDR enabled."""
        t = get_adaptive_thresholds(n_samples=30)
        assert t.ani_main == 95.0
        assert t.cov_main == 0.90
        assert t.cov_secondary == 0.82
        assert t.cov_recovery == 0.75
        assert t.use_fdr is True
        assert t.use_permutation is False
        assert t.coverage_is_hard_gate is True

    def test_medium_samples(self):
        """20-29 samples: moderate thresholds, permutation + FDR."""
        t = get_adaptive_thresholds(n_samples=25)
        assert t.ani_main == 95.0
        assert t.cov_main == 0.85
        assert t.use_fdr is True
        assert t.use_permutation is True
        assert t.n_permutations == 999

    def test_few_samples(self):
        """10-19 samples: relaxed thresholds, permutation only."""
        t = get_adaptive_thresholds(n_samples=15)
        assert t.ani_main == 96.0
        assert t.cov_main == 0.75
        assert t.use_fdr is False
        assert t.use_permutation is True
        assert t.n_permutations == 9999

    def test_very_few_samples(self):
        """<10 samples: relaxed thresholds, no FDR/permutation."""
        t = get_adaptive_thresholds(n_samples=5)
        assert t.ani_main == 97.0
        assert t.cov_recovery is None
        assert t.use_fdr is False
        assert t.coverage_is_hard_gate is False

    def test_ani_override(self):
        """ANI override should set all ANI thresholds."""
        t = get_adaptive_thresholds(n_samples=20, ani_override=93.0)
        assert t.ani_main == 93.0
        assert t.ani_secondary == 93.0
        assert t.ani_recovery == 93.0

    def test_cov_override(self):
        """Coverage override should set all coverage thresholds."""
        t = get_adaptive_thresholds(n_samples=20, cov_override=0.80)
        assert t.cov_main == 0.80
        assert t.cov_secondary == 0.80
        assert t.cov_recovery == 0.80

    def test_coverage_as_tiebreaker(self):
        """coverage_as_tiebreaker disables hard gate."""
        t = get_adaptive_thresholds(n_samples=20, coverage_as_tiebreaker=True)
        assert t.coverage_is_hard_gate is False

    def test_tnf_main_override(self):
        """TNF main override should be used."""
        t = get_adaptive_thresholds(n_samples=20, tnf_main_override=0.85)
        assert t.tnf_main == 0.85

    def test_tnf_main_default(self):
        """TNF main should default to TNF_GATE_MAIN."""
        from tritanc.config import TNF_GATE_MAIN
        t = get_adaptive_thresholds(n_samples=20)
        assert t.tnf_main == TNF_GATE_MAIN


class TestConstants:
    """Test that constants are properly defined."""

    def test_noise_taxa(self):
        assert "" in NOISE_TAXA
        assert "unclassified" in NOISE_TAXA
        assert "root" in NOISE_TAXA

    def test_canonical_ranks(self):
        assert "genus" in CANONICAL_RANKS
        assert "species" in CANONICAL_RANKS
        assert len(CANONICAL_RANKS) == 7

    def test_main_ranks(self):
        assert MAIN_RANKS == {"genus", "species"}

    def test_secondary_ranks(self):
        assert SECONDARY_RANKS == {"order", "family", "class", "phylum"}

    def test_tool_hints(self):
        assert "mmseqs" in TOOL_HINTS
        assert "skani" in TOOL_HINTS


class TestToolChecks:
    """Test tool checking helpers."""

    def test_check_tools_all_present(self, monkeypatch):
        """check_tools should not exit when all tools are present."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
        # Should not raise SystemExit
        check_tools(["mmseqs", "skani"])

    def test_check_tools_missing(self, monkeypatch):
        """check_tools should exit when tools are missing."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda x: None)
        with pytest.raises(SystemExit):
            check_tools(["mmseqs"])

    def test_tools_needed_for(self, monkeypatch):
        """tools_needed_for should return correct tool list."""

        class MockArgs:
            taxonomy = None
            ani = None
            depth = None
            prot_sim = None
            skip_prot_sim = False
            skip_checkm2 = False
            skip_drep = False

        monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
        needed = tools_needed_for(MockArgs())
        assert "mmseqs" in needed
        assert "skani" in needed
