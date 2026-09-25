# Progress log — Sprint 0

Last updated 2026-09-25. **129 tests passing** (63 Owner A, 26 Owner B, 40 Owner D).

Records what is built, what the real data actually says, and what is still open.
For the task breakdowns see `sprints/sprint-0/`; for the plan see `docs/`.

---

## 1. Findings from the real data

Run over the full training set with `scripts/probe_cardinality_script.py`.
**Several of these contradict assumptions the plan was written against**, so they
override `docs/research.md` where they conflict.

### 1.1 Cardinality — the plan's central assumption was wrong

| quantity | research.md assumed | **measured** |
|---|---|---|
| singleton rate (t=0) | "say, 40%" (§1.2) | **5.58%** |
| t=1 | — | 5.40% |
| t=2–4 | — | **63.0%** |
| t≥5 | — | **26.0%** |
| mean t | — | **3.46** (max 11) |
| singleton rate US / India | — | 5.583% / 5.588% |

This is a **many-matches** problem, not a singleton problem. Consequences:

- Singleton handling is worth ~0.056 of the score, not ~0.40.
- The empty set is the wrong answer for **94%** of rows. It stays the default
  when candidates are weak, but the decoder is mostly choosing a set *size*.
- λ_null falls sharply — 94.4% of S1 rows have at least one match.
- Precision pressure moves to the t=2–4 band: at t=2, missing one costs 0.167
  while adding one wrong costs 0.286.
- research.md §2.2 rejected Shopee "min-k" forcing because it "would zero every
  true singleton" — an argument worth 0.40 then and 0.056 now. The conclusion is
  probably still right, but **must be re-derived on CV, not inherited**.

### 1.2 Non-Latin script is large and multi-script

| source | India rows scanned | non-Latin | rate |
|---|---|---|---|
| S2 | 400,819 | 165,341 | **41.3%** |
| S3 | 400,995 | 130,043 | **32.4%** |

research.md §3.1 anticipated Devanagari only, gated behind a `[VERIFY]`. The data
carries **at least six scripts**: Devanagari, Malayalam, Gujarati, Tamil, Telugu,
Bengali. Two distinct patterns:

1. **Whole name transliterated** — `ഡ്രീം മീഡിയ പ്രൈവറ്റ് ലിമിറ്റഡ്` is "Dream
   Media Private Limited". Char n-grams give **zero** overlap with a Latin S1
   record, and Double Metaphone is Latin-only. Only the dense multilingual
   encoder can bridge these, which makes §3.2's dense channel **load-bearing,
   not optional**.
2. **Latin name, native-script state in the address tail** — `...Vadodara,
   ગુજરાત`. Much easier; the state token behaves as a categorical.

### 1.3 Scale — ~24M rows, roughly 3× the §6 estimate

| | train | test |
|---|---|---|
| S1 | 2,206,821 | 1,732,544 |
| S2 | 5,034,616 | 4,887,273 |
| S3 | 5,285,603 | 5,082,316 |

§6 assumed 8–9M total. **The §6 compute table needs redoing**: the number that
governs Sprint 1 is candidate-pair volume, K × ~10M test fragments. At K=25 that
is 250M pairs against §6's assumed 150M, on $200 of credits.

### 1.4 Country labels — A0.5 RESOLVED

Both splits carry exactly three casefolded labels. The test split adds France:

| split | labels |
|---|---|
| train | `us`, `india` |
| test | `us`, `india`, **`france`** |

France is literally `"France"` — no `FR`, no `USA`, no `United States`, and no
missing or empty values in the scanned window. `normalise_country` (casefold,
keep the label) already handles this; no code change was needed.

**France is 287,023 of 2M scanned test fragment rows (~14%)** — a substantial
slice, not a token presence. Every country-keyed path needs a working France
branch.

Non-Latin rates hold across the split, so there is no train→test script shift:

| | train | test |
|---|---|---|
| S2 India non-Latin | 41.3% | 41.7% |
| S3 India non-Latin | 32.4% | 32.9% |

### 1.5 Ground-truth file shape — resolves Q7

`train_ground_truth.tsv` has header `source1_entity_id<TAB>matched_entity_ids`
with `n_data_rows == n_distinct_s1 == 2,206,821`: the io_rules.md §5.1 list
shape, one row per S1. So **io_rules.md §5.2 is right and research.md §11 Q7's
competing "two columns" default is wrong**. `candidate_pairs.tsv` uses the list
shape.

### 1.6 Blocking measured on real data — Gate 1 target met at K=5

`scripts/probe_blocking.py`, S1-first samples of real train, all four channels
with the dense channel active on a Colab T4.

| run | S1 indexed | fragments queried | PC | entity ceiling | F0.5 ceiling |
|---|---|---|---|---|---|
| 5k, e5-small | 5,000 | 22,375 | 0.99983 | 0.9994 | 0.99996 |
| 50k, e5-large | 50,000 | 222,558 | 0.99921 | 0.99732 | 0.99977 |

**research.md §3.2's Gate 1 target (PC ≥ 99% at K ≤ 30) is met at K=5.** The
blocking problem is far easier than the plan assumed, which frees Sprint 1 to
spend its time on scoring precision instead of recall.

Two things follow, and both change plans:

**Channel contribution is lopsided.** On the 50k run:

| channel | fragments reached | hits per fragment |
|---|---|---|
| dense | 100.0% | 50.0 |
| bm25 | 99.9% | 49.8 |
| char_tfidf | 93.2% | 40.5 |
| **keys** | **4.0%** | **0.1** |

The exact-key channel reaches 4% of fragments. research.md §3.2 treats keys as
"exact anchors" and §3.2's fusion rule grants them a standing exemption from the
top-K cut; at 4% reach that exemption is nearly inert. The likely cause is that
`(postcode, house#)` needs both a parsed postcode *and* a house number, and the
A0.1 ruling left address parsing regex-only. Worth an ablation before investing
further in keys.

**K can come down, which matters for compute.** Cost per K at 10M test
fragments, against the measured ceiling:

| K | PC | entity ceiling | pairs to featurise | ceiling loss |
|---|---|---|---|---|
| 10 | 0.9977 | 0.9925 | 100M | 0.0075 |
| 15 | 0.9986 | 0.9954 | 150M | 0.0046 |
| 20 | 0.9990 | 0.9967 | 200M | 0.0033 |
| 25 | 0.9992 | 0.9973 | 250M | 0.0027 |

Going from K=25 to K=15 costs 0.0019 of ceiling and saves 100M pairs of Stage-A
featurisation. On a $200 budget that trade is probably worth taking; it is
Sprint 1's call, but the numbers now exist to make it.

**India no longer lags US.** 0.99862 vs 0.99959 pair completeness — a 0.001 gap,
where the Indic-script bugs (§2.1, §2.2, §2.5) would have produced a chasm. The
dense channel reaching 100% of fragments is what closes it.

### 1.7 Blocking runtime is the real Sprint 1 blocker

Scaling from the 5k run to the 50k run: S1 grew 10x and fragments 9.9x, but
query time grew **58x**, not 9.9x. Cost scales with S1 x fragments, because every
fragment is scored against the whole S1 index.

Extrapolated to the real test run (1.73M S1 index, 9.97M fragments):

| | value |
|---|---|
| measured | 1,837 s for 222,558 fragments against 50k S1 |
| per fragment-S1 unit | 1.65e-7 s |
| **full test, single process** | **~792 h (33 days)** |

research.md §6 budgets char-TF-IDF at 30–90 min CPU / 10–20 min GPU. **The
measured path is about three orders of magnitude over that budget**, and the
probe's own `projected_minutes_for_10M_fragments` (236 min) is wrong because it
extrapolates linearly in fragments while the true cost is quadratic.

This is now the top Sprint 1 item. Options, cheapest first:

1. **GPU the sparse channels.** §3.2 already names cuML/cupy and
   `sparse_dot_topn`; the current implementation is pure scipy on CPU.
2. **Let the dense channel carry retrieval.** It reaches 100% of fragments on
   its own and FAISS is already GPU-backed. Sparse then re-ranks a shortlist
   rather than scanning the full index — the SIGMOD-2022 recipe in §2.3.
3. **Shard the S1 index.** Blocked on B0.1 (cross-country matches), and it must
   never exclude unseen labels.

---

## 2. Bugs found and fixed

### 2.1 Indic text was being destroyed (critical)

`normalise_name` reduced Indic names to a single character:

```
ഡ്രീം മീഡിയ പ്രൈവറ്റ് ലിമിറ്റഡ്   →  'ഡ'
मॉडर्न टेक्नोलॉजीज प्राइवेट लिमिटेड  →  'म'
```

Indic vowel signs and the virama are Unicode categories `Mn`/`Mc`, for which
`str.isalnum()` is `False`, so the punctuation filter replaced each with a space
and shattered every word. Given the 41%/32% rates above, this was **silently
wiping the matching signal on ~40% of India fragments** — it would have looked
like a model quality problem, not a normalisation bug. Fixed by keeping `Mn`/`Mc`
explicitly; Latin and French output is byte-identical.

### 2.2 Transliterated legal suffixes were never stripped

The legal suffix is transliterated along with the name
(`പ്രൈവറ്റ് ലിമിറ്റഡ്` = "Private Limited"), so the Latin-only suffix list never
reached `name_core` on those rows. `rules.py` now carries native-script
renderings for seven Indic scripts — transliterations of an English legal form,
not gazetteer entries, so §9-compliant.

### 2.3 Decoder returned the wrong k (numerical)

An optimisation of `best_k` deconvolved candidates out of the full
Poisson-binomial PMF by forward substitution. It was 2× faster and **silently
wrong**: repeated division by `(1-p)` amplified the near-zero Poisson tail until
a PMF that should sum to 1.0 summed to **−12531**, and the decoder picked k=1
where the exact answer was k=2. Caught only by diffing the fast path against
exact recomputation. Replaced with suffix PMFs built by a stable backward scan,
verified over 8000 random cases at max deviation 4.4e-16.

### 2.4 Vendor noise in legal suffixes

The real test data writes legal suffixes inconsistently: `Private-Limited`
(hyphenated), `(LIMITED)` (parenthesised), `PRIVATE-PRIVATE` (duplicated),
`SERVICESPRIVATE` (no space). Suffix *detection* used a substring test with
literal spaces, and since the stripper is handed only what detection found, the
hyphenated form survived into `name_core` entirely. Detection and stripping now
share one matcher that accepts spaces or hyphens between suffix words.

`SERVICESPRIVATE LIMITED` and a bare `[Private]` remain unhandled and should
stay that way — the first has no token boundary to match on, and the second is
not a legal suffix on its own.

### 2.5 Zero-width characters split words

Telugu names in the test data contain U+200C ZERO WIDTH NON-JOINER. It is
category `Cf` — invisible but *not* whitespace — so the punctuation filter
replaced it with a space and **split one word into two**. Two spellings of the
same name became different tokens and stopped matching. Zero-width characters
(ZWNJ, ZWJ, ZWSP, BOM, word-joiner, soft hyphen) are now deleted outright.

### 2.6 Earlier, smaller fixes

- Accented Latin was stripped as punctuation (`Société` → `soci t`) — the exact
  France failure io_rules.md §3 warns about.
- Multi-word suffixes lost their tail first (`pvt ltd` → stray `pvt`).
- `societe` was over-stripped from trade names, mangling "Société Générale".
- The audit probe OOMed on Colab: `read_source(path).head(N)` loads the whole
  file before slicing, costing ~3 GB across S2+S3.
- Non-Latin eyeball sampling was biased to the head of the file; replaced with a
  uniform reservoir over a bounded window.

---

## 3. What is built

### Owner A — 27/37 tasks, 58 tests

| module | purpose |
|---|---|
| `safe_io.py` | io_rules.md §1 reader + 3 sanity checks (raise, not warn); namespace + ground-truth checks |
| `normalise.py` | A2.1 contract, 16 fields. NFKC → casefold → accent-folded copy keeping both |
| `rules.py` | US/IN/FR legal suffixes, street abbreviations, landmarks, Indic suffixes |
| `placeholders.py` | io_rules.md §4 set + data-driven discovery |
| `proxy_data.py` | self-authored synthetic S1/S2/S3, France test-only |
| `audit.py` / `run_audit.py` | research.md §10 checklist, JSON output |

### Owner D — 24/64 tasks, 40 tests

| module | purpose |
|---|---|
| `scorer.py` | exact macro-F0.5 + singleton rule, slice reporting |
| `submission.py` | writer, team pre-flight validator, organiser-validator wrapper |
| `decode.py` | N2 exclusivity + NULL, N1 expected-F0.5 Poisson-binomial DP |

**Gate 0 criterion 3 is met.** All eleven rows of research.md §1.2's consequence
table reproduce exactly as parametrised tests, and both §1.3 worked examples
pass: the 0.5 single-candidate break-even, and rejection of a second candidate at
π=0.5 (`{1}` = 0.825 beats `{1,2}` = 0.728).

**Decode performance:** ~174 µs/row on realistic post-floor inputs → 6.4 min for
2.2M S1 rows single-core, 1.6 min on 4 cores. Degrades to ~31 min if candidates
do not thin out at the 0.02 floor, so **Owner B's candidate quality directly
affects decode time**.

### Owner B — 25/41 tasks, 26 tests

| module | purpose |
|---|---|
| `blocking.py` | char TF-IDF, scipy BM25, exact keys, RRF fusion |
| `dense.py` | FAISS + multilingual encoder, optional imports, licence record |
| `blocking_report.py` | PC / RR / per-entity ceiling / slices, harness entry point |

Measured on real data — see §1.6. Gate 1's PC target is met at K=5; runtime is
the open problem (§1.7).

### Owner C — 0/56 tasks. Not started.

---

## 4. Resolved questions

| # | Question | Resolution |
|---|---|---|
| A0.1 / Q2 | Is libpostal "external data"? | **Yes → excluded.** Regex-only is final, not provisional. A CRF trained only on provided addresses is the sole escalation path |
| A0.2 / Q3 | Fragment → ≤1 S1? | Holds on proxy; re-confirm on real train |
| A0.3 / Q4 | Within-vendor 1:1? | **Does not hold** → D5.3 bipartite branch stays OFF |
| A0.4 / Q5 | Singleton & orphan rates | **5.58% singleton**, mean t=3.46 |
| A0.5 | Exact country strings | **RESOLVED: `us`, `india`, `france`.** France is literally "France" and is ~14% of test fragments |
| A0.6 | Non-Latin census | **41%/32% of India rows**, 6+ scripts. Holds on test (41.7%/32.9%) |
| Q7 | `candidate_pairs.tsv` schema | **io_rules.md §5.2 list shape**, confirmed by the organiser's own file |

### Still open

| # | Question | Blocking what |
|---|---|---|
| Q8 | Can test text be used unsupervised (DAPT)? | Owner C |
| Q10 | Public LB fraction / submission limit | Leaderboard hygiene |
| Q11 | Scorer whitespace / ordering | Assumed order-insensitive, trimmed |
| — | §6 compute table vs real 24M scale | **Measured: blocking is ~33 days single-process at full scale, ~1000x over the §6 budget.** Top Sprint 1 item (§1.7) |
| Q6 / B0.1 | Cross-country matches exist? | Country sharding stays off |

---

## 5. Gate 0 status

Roadmap gate: *"pipeline runs end-to-end on proxy data and emits a validator-PASS
submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly."*

| # | Criterion | Owner | Status |
|---|---|---|---|
| 1 | End-to-end run on proxy data | D8 | **blocked** — needs Owner B blocking + Owner C scoring |
| 2 | Validator-PASS submission | D4 | code done; unexercised end-to-end |
| 3 | Scorer reproduces toy F0.5 exactly | D1 | **PASS** |

---

## 6. Known gaps

- **Dense channel is now mandatory, not optional.** research.md §3.2's go/no-go
  ("drop dense if it adds <0.3 PC points") must not be applied on a US-heavy
  slice — 41% of India names need it.
- **`metaphone` install is unverified in Colab.** The notebook cell uses
  `check=False`, so failure is silent and the built-in fallback is used. If
  `phonetic_key` matters for Owner B's key channel, confirm rather than assume.
- **No end-to-end run exists yet**, so the writer and validator have never been
  exercised against a real pipeline output.
