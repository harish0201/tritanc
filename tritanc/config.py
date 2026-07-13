"""Constants, adaptive thresholds, tool hints, and logging configuration."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass


# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

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
# Tool hints
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


__all__ = [
    "MIN_LEN", "TNF_MIN_LEN", "ANI_MIN_AF", "ANI_THRESHOLD", "COV_THRESHOLD",
    "COV_PVAL", "NOISE_TAXA", "CANONICAL_RANKS", "GENUS_IDX",
    "MAIN_RANKS", "SECONDARY_RANKS",
    "CHECKM2_MIN_COMPLETENESS", "CHECKM2_MAX_CONTAMINATION", "DREP_ANI",
    "MMSEQS_THREADS", "MMSEQS_TAX_LINEAGE", "MMSEQS_SENS",
    "MAX_TAX_CANDIDATES",
    "LEIDEN_RES_MAIN", "LEIDEN_RES_SECONDARY", "LEIDEN_RES_T4",
    "TNF_GATE_MAIN",
    "TAXOMETER_PREFIX",
    "TOOL_HINTS",
    "AdaptiveThresholds", "get_adaptive_thresholds",
]
