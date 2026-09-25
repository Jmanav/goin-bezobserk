---
name: score
description: Compute local validation metrics for the resolved entity output against any labeled/held-out data, to sanity-check quality before submission. Use after decode, before submit.
---

# Score

Validate the entity resolution output locally before submitting.

## Steps
1. Load resolved output from `data/processed/` (produced by `decode`).
2. If labeled/held-out data is available, compute the competition's actual
   metric (see `docs/io_rules.md` / `docs/research.md` for which metric).
3. Otherwise, compute sanity-check proxies: cluster size distribution,
   singleton rate, comparison against any known ground-truth pairs.
4. Compare against the current best score / baseline in `docs/roadmap.md`.
5. Report the score clearly — do not modify the output file here.

## Gate
Only proceed to `submit` once the score is stable and reviewed — this skill
should never silently pass a regression through.
