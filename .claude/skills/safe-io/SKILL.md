---
name: safe-io
description: Enforce the project's I/O contract (docs/io_rules.md) when reading raw data or writing any intermediate/final artifact. Use before writing any output file, and whenever unsure of expected format.
---

# Safe I/O

Enforce the I/O contract defined in `docs/io_rules.md` for all reads/writes
in this pipeline.

## Rules
- Raw input data is **read-only**. Never overwrite or mutate files under the
  raw data location named in `docs/io_rules.md`.
- Every write goes to a clearly named path under `data/interim/` (intermediate)
  or `data/processed/` (final-stage) — create these directories if missing.
- Before writing any file, validate against `docs/io_rules.md`:
  - Correct file format (CSV/Parquet/JSON as specified)
  - Correct column names, order, and dtypes
  - Correct encoding (UTF-8 unless specified otherwise)
- Never commit raw or large processed data files to git — check `.gitignore`
  covers `data/` before adding new data paths.

## When invoked
Use this skill as a checklist immediately before any file write, and when
loading raw data to confirm you're reading it per the documented contract.
