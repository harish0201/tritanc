"""TriTanc — Taxonomy-aware metagenomic contig clustering pipeline.

A modular Python package for clustering metagenomic assembly contigs into
bins using multiple signals: ANI (skani), protein similarity (MMseqs2),
taxonomy (MMseqs2/Taxometer), co-abundance (coverage correlation), and
composition (TNF k-mer frequencies).

Modules:
    config     — Constants, adaptive thresholds, tool hints
    checkpoint — Parquet/JSON/TNF save & load helpers
    parsers    — Input parsers (FASTA, taxonomy, ANI, depth, protein sim)
    signals    — Core algorithms (TNF, Spearman, Leiden, centroids)
    tools      — External tool runners (mmseqs, skani, samtools, etc.)
    graph      — Multimodal graph building and edge scoring
    tiers      — Tier 1–2 clustering, Tier 4 de-novo, Tier 5 coverage recovery
    recovery   — Tier 3 lineage-aware contig recovery
    output     — FASTA bin writing, CheckM2, dRep integration
    cli        — Argument parsing and main() orchestration
"""

__version__ = "10.0.0"
