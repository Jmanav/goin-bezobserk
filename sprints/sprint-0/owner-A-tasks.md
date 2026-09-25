# Sprint 0 — Owner A: Normalisation / Data Audit / Rule Lists

**Roadmap Sprint 0 deliverable (verbatim):**
> `read_source()` safe TSV reader + the 3 sanity checks; normaliser skeleton with US/IN/FR suffix & abbreviation rule lists; placeholder detector

Owner A area: Normalisation / data audit / rule lists (US, IN, FR) — roadmap.md § Owners.

---

## A0. Blocking tasks (`[VERIFY]` — not coding tasks)

These are **decisions/confirmations**, not code. Do not write dependent code paths until each is resolved. Every item is a `[VERIFY]` carried from research.md / io_rules.md.

- [x] **A0.1 — Offline parser ruling (Q2, research.md §11 + §3.1 + io_rules.md §9). RESOLVED: libpostal is NOT usable.** Gazetteer-backed parsers (libpostal — model data derived from OSM/OpenAddresses/GeoNames) count as external data and are excluded. deepparse (LGPL-3.0) excluded regardless. Address parsing stays regex-only per research.md §3.1 ("regex plus a small CRF/token tagger trained only on the provided training addresses"). No further action; this is no longer a blocking question.
  - Blocks: any address-parsing implementation choice beyond regex (A3). Until resolved, build regex-only per research.md §3.1 ("Recommendation: regex plus a small CRF/token tagger trained only on the provided training addresses if needed").
- [ ] **A0.2 — Fragments map to ≤ 1 S1? (Q3, research.md §11 + §1.1 + audit #3).** Cannot be confirmed on real data pre-event; confirm on **proxy data generator invariants** now and re-run on real train at H0–6. Default assumption: YES. Expect 0 fragments whose ID appears in more than one S1 ground-truth row.
- [ ] **A0.3 — Within-vendor 1:1 per S1? (Q4, research.md §11 + §1.1 + audit #3).** For each S1 entity, count matched S2 IDs and matched S3 IDs **separately**; if max is 1, the within-vendor one-to-one constraint holds. Default: **Unknown.**
  - Blocks: Owner D's optional bipartite-matching branch in N2 (research.md §3.4 `if VERIFY(one_per_vendor_per_S1)`). Tell D the answer is "unknown" so D codes the branch as **off by default**.
- [ ] **A0.4 — Singleton & orphan rates (Q5, research.md §11 + audit #4, #6).** Unknown until real data. Set the proxy generator to a **parameterised** singleton rate so the rate is a knob, not a baked-in constant (research.md §1.2 notes a 40% singleton share would swing up to 0.40 of the score).
- [ ] **A0.5 — Exact country string values (io_rules.md §3 `[VERIFY]`).** `"France"` vs `"FR"` vs `"france"`; `"US"` vs `"USA"` vs `"United States"`. Unknown until real data. Consequence for code now: **normalise casing but keep the label**, and route anything unseen to the `"other/unseen"` bucket — never a `{US, India}` branch.
- [ ] **A0.6 — Non-Latin script census (research.md §3.1 `[VERIFY]`).** Count non-Latin characters per country. Only add rule-based transliteration **if** Devanagari appears **and** the library ships no external dictionaries. Until then: char n-gram similarity + Double Metaphone only (both algorithmic, not data).

---

### A0 status after the Sprint 0 implementation

Implementation is done (A1–A5, `src/ber/`, 54 tests). What the **proxy** audit can and cannot answer:

| Probe | Proxy result | Still needed |
|---|---|---|
| A0.1 offline parser | n/a | **RESOLVED — excluded.** Address parsing is regex-only. A CRF trained only on provided addresses is the sole escalation path (Sprint 2+, only if the audit shows regex leaving signal) |
| A0.2 fragment → ≤ 1 S1 | HOLDS (0 multi-owned) | re-run on real train at H0–6 |
| A0.3 within-vendor 1:1 | **does NOT hold** (max 2/vendor) | tell Owner D: keep N2 bipartite branch **off** (D5.3) |
| A0.4 singleton/orphan rates | parameterised knob, not baked in | real rates unknown until H0–6 |
| A0.5 exact country strings | audit reports exact distinct values | real label spellings unknown |
| A0.6 non-Latin census | Latin + accents only in proxy | real Devanagari census at H0–6 |

Run it with `python -m ber.run_audit --proxy` (or `--data-dir dataset/train` on real data).

---

## A1. `read_source()` safe TSV reader

Exact signature and params are fixed by io_rules.md §1 — copy them, do not re-derive.

- [x] **A1.1 — Implement `read_source(path)`** exactly per io_rules.md §1: `sep="\t"`, `dtype=str`, `keep_default_na=False`, `na_filter=False`, `quoting=csv.QUOTE_NONE`, `encoding="utf-8"`, `on_bad_lines="warn"`. No other defaults. (Also research.md §1.4 and §5 "TSV/pandas NA traps".)
- [x] **A1.2 — Implement the 3 sanity checks** as a callable that runs on **every** read (io_rules.md §1 "do these every run"):
  1. `len(df) == wc -l(file) - 1` (row count vs raw line count minus header).
  2. Every raw line has exactly **4 fields** — count tabs **per raw line before trusting the parse** (`entity_id, business_name, business_address, country`, io_rules.md §2).
  3. `df.isna().sum().sum() == 0` (no unexpected NaN anywhere).
- [x] **A1.3 — Make the checks fail loudly.** A failed sanity check must raise, not warn — these are the "silent killers" in io_rules.md §8. Emit which check failed and the offending line numbers.
- [x] **A1.4 — ID/namespace assertions** (io_rules.md §2 + research.md §10 audit #2): assert every ID carries the prefix expected for its file (`S1-`/`S2-`/`S3-`), assert the three ID sets are **disjoint**, and assert ground-truth IDs ⊆ train IDs. **Never strip the prefix** — bare numbers collide across sources (`S2-00001` ≠ `S3-00001`).
- [x] **A1.5 — Country audit pass** (io_rules.md §3 + research.md §10 audit #7): report counts per source, exact distinct `country` string values, cross-country matches in train (expect 0), and unseen labels. Output is a report, not a filter — **no code path may filter, one-hot, or hard-code `{US, India}`**.

## A2. Normaliser skeleton

Inputs/outputs fixed by research.md §3.1 — the output field list is a contract for Owner C's features and Owner B's blocking keys.

- [x] **A2.1 — Define the normaliser output contract** (research.md §3.1 "Outputs"): `name_norm`, `name_core` (legal suffixes + generic descriptors removed), `name_tokens`, `addr_norm`, `addr_numbers` (house number, unit, floor), `postcode`, `landmark_flag`, `script_flag`, plus a **folded ASCII variant**. Publish this field list to B, C, D before implementing — they code against it.
- [x] **A2.2 — Unicode layer** (research.md §3.1): NFKC → casefold → an accent-folded **copy** (keep **both**; "Société Générale" vs "Societe Generale"). Normalise punctuation variants (`’ ' \` “ ”`) and ampersands (`&` ↔ `and`/`et`).
- [x] **A2.3 — Token IDF computation** (research.md §3.1 + §3.3): compute each token's document frequency in **S1** names. Needed by C's IDF features and B's rare-name-token key.
- [x] **A2.4 — Descriptor down-weighting (learned, not hard-coded)** (research.md §3.1): tokens above a DF percentile ("restaurant", "pharmacy", "store", "traders") get **down-weighted by IDF rather than deleted**. Learn from training positives which tokens often *differ* between matched pairs (vendors appending "Store"/"Branch").
- [x] **A2.5 — Address number/unit extraction (regex only)** (research.md §3.1): house number, units (`Ste 200`, `Flat 3B`, `#12`, `Apt`), US ZIP (5 or 9 digits), India PIN (6 digits), France postcode (5 digits, `CEDEX`). Regex-based. A0.1 resolved: no gazetteer-backed parser, so this regex path is final, not provisional.
- [x] **A2.6 — Landmark phrase tagging** (research.md §3.1): `near`, `opp`, `opposite`, `behind`, `beside`, `nr`, `b/h`, `next to`, `en face de`, `près de` → `landmark_flag`.
- [x] **A2.7 — Phonetic key** (research.md §3.1 + §3.2 Keys row): Double Metaphone of `name_core`. Algorithmic, not a data lookup. Feeds B's `(Double Metaphone of name_core, postcode prefix)` key.

## A3. US/IN/FR suffix & abbreviation rule lists

Hand-written rule lists are **domain knowledge, not an external lookup** — but they **must be documented in the methodology doc** (research.md §3.1, §9).

- [x] **A3.1 — Legal-suffix lists, all three countries** (research.md §3.1, verbatim scope):
  - **US:** Inc, Incorporated, LLC, L.L.C., Corp, Co, Ltd, LLP, PLLC
  - **India:** Pvt Ltd, Private Limited, Pvt. Ltd., (P) Ltd, LLP, `M/s`/`M/S` prefix, `& Sons`, `Enterprises`, `Traders`
  - **France:** SARL, SAS, SASU, SA, EURL, SCI, SNC, `Société`, `Ets`/`Établissements`
- [x] **A3.2 — Store suffix as a *feature*, not only a strip** (research.md §3.1 + §5 "Legal-suffix confusion"): emit **suffix agreement / conflict** so "X Pvt Ltd" vs "X LLP" is a conflict signal, not silently erased. This is a feature handed to Owner C.
- [x] **A3.3 — Street abbreviation expansion** (research.md §3.1): `St↔Street`, `Rd↔Road`, `Ave↔Avenue`, `Bd/Blvd↔Boulevard`, `Nagar`, `Marg`.
- [x] **A3.4 — FR rule list written now, not deferred.** France is **test-only** (io_rules.md §3, research.md §1.4) — any rule tuned only to US+India "will quietly degrade there." FR suffixes and `en face de`/`près de`/`CEDEX` land in Sprint 0, not Sprint 2.
- [x] **A3.5 — Transliteration variant handling** (research.md §3.1): Shri/Shree/Sri, Enterprises/Ent., Aggarwal/Agarwal — via char n-gram similarity + Double Metaphone (A2.7). Gated by A0.6 for anything script-specific.
- [ ] **A3.6 — Start the methodology-doc rule-list appendix.** Every rule list above gets documented as hand-written domain knowledge (research.md §9 compliance checklist). Stub it now; full doc is Sprint 3.

## A4. Placeholder detector

- [x] **A4.1 — Detect the placeholder set** (io_rules.md §4): `""`, `"-"`, `"0"`, `"NA"`, `"None"`, `"null"`, `"n/a"` → set a `missing` flag.
- [x] **A4.2 — Data-driven placeholder discovery** (research.md §5 "Placeholders/missing" test column): count strings with DF > 1% of length ≤ 3. Catches placeholders not in the hardcoded list.
- [x] **A4.3 — Enforce the never-match rule** (io_rules.md §4): **never** let two records match *because* they share a placeholder token. Empty address ≠ match signal. Empty name → lean on address only, and be **conservative**. This is a hard constraint on the flag's downstream use — publish it to C.

## A5. Proxy data contribution

- [x] **A5.1 — Author the seed business list yourself** (roadmap.md § Sprint 0 "Proxy data", research.md §7): synthetic S1/S2/S3 by corrupting business lists **you create yourself** — do **not** ship external data into training for the real event.
- [x] **A5.2 — Include all three countries in the proxy, with France test-only.** Mirrors the real train/test country shift (io_rules.md §3) so the leave-one-country-out split in Sprint 2 has something to bite on.
- [x] **A5.3 — Seed the proxy with the §5 edge cases** (research.md §5): chains/franchises, shared addresses, generic names, one-distinguishing-token ("Store #12" vs "#14"), numeric/unit mismatch, landmark-only, legal-suffix conflicts, acronyms, transliteration/Unicode, placeholders, singletons, large clusters (t ≥ 5). Parameterised singleton rate per A0.4.

---

## Must satisfy (io_rules.md rules applicable to Owner A)

- [ ] **Safe TSV read flags, exactly** (§1): `sep="\t"`, `dtype=str`, `keep_default_na=False`, `na_filter=False`, `quoting=csv.QUOTE_NONE`, `encoding="utf-8"`, `on_bad_lines="warn"`.
- [ ] **The 3 sanity checks run every run** (§1): row count vs `wc -l` − 1; exactly 4 fields per raw line (count tabs before trusting the parse); zero unexpected NaN.
- [ ] **Never strip ID prefixes** (§2, §8): the prefix defines the namespace; bare numbers may collide across sources. Full IDs are the key everywhere.
- [ ] **Source is prefix + which file the record came from** (§2) — there is no `source` column.
- [ ] **Country is an open set of string labels** (§3, §8): do not hard-code, filter, or one-hot to `{US, India}`. Unseen/missing country must still flow through the full pipeline and appear in the submission. Use an explicit `"other/unseen"` bucket, never a US/India-only branch. Normalise casing but **keep the label**.
- [ ] **Placeholder detector sets a `missing` flag and never creates matches** (§4): no two records match *because* they share a placeholder. Empty address ≠ match signal.
- [x] **No external data** (§9): no registries, geocoders, ER APIs, scraped lists. Hand-written rule lists are documented as domain knowledge. Offline gazetteer-backed parsers (libpostal, deepparse) are **external data and excluded** — A0.1 resolved, not a default.
- [ ] **Reproducible from raw TSV** (§9): fixed seeds, pinned deps; the normaliser must be deterministic.

---

## Gate 0 — pass/fail (roadmap.md § Sprint 0, verbatim criteria)

> **Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

| # | Criterion | PASS / FAIL |
|---|---|---|
| 1 | Pipeline runs **end-to-end on proxy data** | ☐ PASS ☐ FAIL |
| 2 | It emits a **validator-PASS** submission | ☐ PASS ☐ FAIL |
| 3 | Scorer **reproduces a hand-computed F0.5 on a toy set exactly** | ☐ PASS ☐ FAIL |

Gate 0 is a single shared checkpoint for all four owners — A passes only when the whole pipeline passes. Owner A's contribution to the gate: `read_source()` + sanity checks + normaliser + placeholder detector are callable by B/C/D, and the proxy data exists for the end-to-end run.
