# Business Entity Resolution — Project Constitution

## What this project is
Amazon ML Challenge '26 submission: business entity resolution pipeline. The
goal is to match/dedupe business entity records (names, addresses, etc.)
across noisy, messy real-world data and produce a scored submission.

## Source of truth
- `docs/research.md` — the build plan (approach, model choices, pipeline design)
- `docs/io_rules.md` — I/O contract: exact input/output formats, file paths,
  column schemas. Treat this as binding — do not guess formats.
- `docs/roadmap.md` — sprint plan and go/no-go gates

## Pipeline stages (see `.claude/skills/`)
1. **data-audit** — profile raw data, find quality issues, encoding problems
2. **safe-io** — enforce the I/O contract (read/write formats, no corruption)
3. **blocking** — candidate generation / blocking keys to cut pairwise comparisons
4. **matching** — pairwise/set similarity + matching logic
5. **decode** — turn match scores/clusters into final entity resolution output
6. **score** — local validation metrics against any labeled/held-out data
7. **submit** — final packaging and submission (manual trigger only)

## Ground rules
- Never modify raw input data in place. Read-only unless explicitly copied to a working dir.
- All intermediate artifacts go under a `data/interim/` or `data/processed/`
  style path (create as needed) — never scattered in repo root.
- Follow `docs/io_rules.md` exactly for final submission format — this is
  graded automatically and format mismatches are the most common failure.
- No committing large data files (raw datasets, model checkpoints) — check
  `.gitignore` before adding data.
- The `submit` skill only runs on explicit `/submit` invocation — never
  auto-trigger a submission.

## Working style
- Prefer small, verifiable steps: audit → validate I/O → block → match →
  decode → score locally → only then submit.
- Surface data quality issues early (encoding, duplicate IDs, null keys)
  rather than letting them silently corrupt downstream matching.
