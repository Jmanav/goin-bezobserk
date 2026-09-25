---
name: matching
description: Compute pairwise/set similarity scores for candidate entity pairs produced by blocking, and apply match decision thresholds. Use after blocking, before decode.
---

# Matching

Score candidate pairs from `blocking` and decide which are true matches.

## Steps
1. Load candidate pairs from `data/interim/` (produced by `blocking`).
2. Compute similarity features per pair — name similarity (token/edit/
   phonetic), address similarity, other available fields.
3. Combine features into a match score (rule-based, learned model, or both
   per `docs/research.md`).
4. Apply a decision threshold (or output raw scores for `decode` to cluster).
5. Write scored pairs to `data/interim/` per `safe-io` rules.

## Notes
- Keep thresholds/config tunable and documented — do not hardcode magic
  numbers without noting where they came from.
- If using a learned model, keep training/eval split leakage-free relative
  to any held-out validation set used by `score`.
