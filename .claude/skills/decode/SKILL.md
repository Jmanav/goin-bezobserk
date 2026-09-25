---
name: decode
description: Convert pairwise match scores/decisions into final entity clusters or resolution output (e.g. connected components, clustering). Use after matching, before score.
---

# Decode

Turn pairwise match decisions/scores into the final entity resolution
output structure (e.g. entity clusters, canonical IDs).

## Steps
1. Load scored/decided pairs from `data/interim/` (produced by `matching`).
2. Resolve pairwise decisions into clusters — e.g. connected components over
   the match graph, or a clustering algorithm if scores are probabilistic.
3. Handle conflicts/transitivity issues explicitly (A~B, B~C, but A!~C) —
   document the resolution strategy used.
4. Assign canonical entity IDs per `docs/io_rules.md`.
5. Write resolved output to `data/processed/` per `safe-io` rules.

## Output
Final entity-resolved dataset, structured per the schema in
`docs/io_rules.md`, ready for `score` or `submit`.
