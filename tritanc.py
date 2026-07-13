#!/usr/bin/env python3
"""TriTanc: Taxonomy-aware metagenomic contig clustering pipeline.

USAGE — skip individual steps by supplying pre-computed files:
  python tritanc.py \
    --fasta assembly.fasta \
    --taxonomy saliva_tax.tsv \
    --taxonomy-format taxometer \
    --ani skani.tsv \
    --depth depth_matrix.txt \
    --outdir results/

"""

from tritanc.cli import main

if __name__ == "__main__":
    main()
