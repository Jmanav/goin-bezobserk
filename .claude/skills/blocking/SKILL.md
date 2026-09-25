---
name: blocking
description: Generate candidate pairs/blocks for entity matching using blocking keys (e.g. normalized name prefix, postal code, country) to cut pairwise comparisons down from all-pairs. Use after data-audit, before matching.
---

# Blocking

Generate candidate entity pairs (blocks) to avoid full O(n^2) pairwise
comparison during matching.

## Steps
1. Confirm data has been audited (`data-audit`) and quality issues are known.
2. Choose blocking key(s) appropriate to business entity data, e.g.:
   - Normalized name prefix / token
   - Postal code / country / region
   - Phone number digits, domain name
3. Generate candidate pairs per block.
4. Report blocking effectiveness:
   - Reduction ratio (candidate pairs vs. all-pairs)
   - Estimated recall (if labeled data available) — are true matches being
     dropped by the blocking key?
5. Write candidate pairs to `data/interim/` per `safe-io` rules.

## Gate
Do not proceed to `matching` until blocking recall is acceptable — a bad
blocking key silently drops true matches before matching ever sees them.
