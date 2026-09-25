---
name: data-audit
description: Profile raw business entity data — schema, nulls, duplicates, encoding issues, cardinality — before any pipeline work begins. Use at the start of the project or whenever new/updated raw data arrives.
---

# Data Audit

Profile the raw input data referenced in `docs/io_rules.md` and report data
quality issues before downstream blocking/matching work begins.

## Steps
1. Read `docs/io_rules.md` for expected schema and file locations.
2. Load raw data **read-only** — never write back to the raw source.
3. Report per column: dtype, null rate, distinct count, sample values.
4. Flag specific business-entity-resolution concerns:
   - Encoding artifacts (mojibake, mixed encodings) in name/address fields
   - Duplicate IDs or primary keys
   - Inconsistent casing/whitespace in name/address fields
   - Missing or malformed keys needed for blocking (e.g. country, postal code)
5. Summarize findings as a short report — do not silently fix issues here;
   that belongs in `safe-io` or `blocking`.

## Output
A data quality summary (text or notebook cell), not a modified dataset.
