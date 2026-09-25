# Sprint 0 — Owner B: Blocking / Candidate Generation / PC-RR Reporting

**Roadmap Sprint 0 deliverable (verbatim):**
> Blocking harness: char-TF-IDF + BM25 + FAISS + RRF; PC / RR reporter with PC-vs-K curve

Owner B area: Blocking / candidate generation / PC-RR reporting — roadmap.md § Owners.

**Sprint 0 scope note:** this is the **harness**, not the tuned system. Dense *contrastive fine-tuning*, fusion tuning and K tuning are **Sprint 1** (roadmap.md § Sprint 1). Build the channels and the reporter so Sprint 1 has something to tune.

**Direction is fixed** (research.md §3.2): **query = each S2/S3 fragment, index = S1.** This matches the many-to-one structure and bounds candidates per fragment. Do not build a symmetric all-pairs blocker.

---

## B0. Blocking tasks (`[VERIFY]` — not coding tasks)

- [ ] **B0.1 — Cross-country matches exist? (Q6, research.md §11 + §3.2 + audit #7).** Owner B per research.md §11. Country may be used as a **shard key only if cross-country matches are verified to be 0 in training**. Unknown until real data → **Sprint 0 default: do NOT shard by country.** Build the sharding as a flag, off by default.
  - Hard constraint regardless of outcome (research.md §3.2, io_rules.md §3): **never filter unknown country values.** France fragments must query France S1 records, and anything with an unseen or missing label must **query the whole index**.
- [ ] **B0.2 — `candidate_pairs.tsv` schema (Q7, research.md §11).** Owner D owns this question, but B produces the file's content. io_rules.md §5.2 states: same schema as `matching_results.tsv` — `source1_entity_id`, `candidate_entity_ids` — same formatting rules. Research.md §11 Q7 records a competing default ("two columns S1 ID, fragment ID"). **Use io_rules.md §5.2 (the S1-row-with-comma-list form); confirm with D before wiring the writer.**
- [ ] **B0.3 — Estimated data scale (research.md §6 `[VERIFY]`).** Assumption: ~1.3 GB at ~150 bytes/row ⇒ roughly 8–9M rows total, e.g. 1–3M S1 and 5–7M fragments. All of B's memory/time budgets below inherit this estimate. **Rescale after the real audit** — do not hard-code index sizing.

---

## B1. Sparse channel 1 — char n-gram TF-IDF

Params from research.md §3.2 table, "Sparse name" row.

- [ ] **B1.1 — Char TF-IDF vectoriser:** sklearn `TfidfVectorizer(analyzer="char_wb")`, char **3–5-gram** (research.md §3.2). Fit on S1 `name_norm` / folded variant from Owner A's normaliser (A2.1 contract).
- [ ] **B1.2 — Chunked sparse top-k retrieval:** `sparse_dot_topn` or cuML/cupy (research.md §3.2). Start params: **top-50, min cos 0.2**.
- [ ] **B1.3 — Chunking for memory safety** (research.md §5 "Runtime blow-ups" + §6): stream in chunks; the Foursquare 7th place generated features and predicted in 10,000-row chunks. Budget per §6: 30–90 min CPU / 10–20 min GPU, 20–40 GB.
- [ ] **B1.4 — Emit per-channel ranks, not just scores.** Owner C needs "the ranks from each channel" as features (research.md §3.3) and RRF needs ranks (B4). Persist rank alongside score.

## B2. Sparse channel 2 — word BM25 on name+addr

Params from research.md §3.2 table, "Sparse name+addr" row.

- [ ] **B2.1 — BM25 implementation:** `bm25s` / Lucene via Pyserini, or a custom `scipy.sparse` BM25 (research.md §3.2). Start params: **k1 = 1.2, b = 0.75, top-50**.
- [ ] **B2.2 — Field variants:** run over name, addr, and name+addr (research.md §3.2 diagram shows all three). Role: **rare tokens**.
- [ ] **B2.3 — Emit BM25 score + rank** for C's feature library (research.md §3.3 lists "BM25 score, and the ranks from each channel").

## B3. Dense channel — FAISS ANN (scaffold only in Sprint 0)

Model choice is licence-constrained (research.md §9 table). Fine-tuning is Sprint 1.

- [ ] **B3.1 — Pick the encoder from the licence-cleared list** (research.md §3.2 + §9): Qwen3-Embedding-0.6B (Apache-2.0) **or** multilingual-e5-large (MIT) / bge-m3 (MIT). **Pin the commit** (research.md §9). Log licence + param count (io_rules.md §9).
- [ ] **B3.2 — FAISS index:** `IndexFlatIP` on GPU for S1 ≤ a few M, or **IVF-PQ if memory-bound** (research.md §3.2). Budget per §6: 5–20 min, S1 ≤ 3M × 2 KB ≈ 6 GB.
- [ ] **B3.3 — Embedding cache to Parquet** (research.md §6): "Precompute embeddings once, cache them to Parquet." Budget: 20–60 min for ~9M strings on 1× A100/L4-class, 1024-d fp16 ≈ 2 KB/row ≈ 18 GB.
- [ ] **B3.4 — Retrieve top-50, emit score + rank** (research.md §3.2). Role: semantic / acronym / transliteration.
- [ ] **B3.5 — CPU-only fallback path** (research.md §6): "Keep a CPU-only fallback that skips the cross-encoder" — the dense channel must degrade gracefully if no GPU is available at competition time.
- [ ] **B3.6 — Leave a hook for contrastive fine-tuning, don't build it.** Sprint 1 adds MultipleNegativesRankingLoss on training (fragment, S1) positives with S1-sibling hard negatives (research.md §3.2, roadmap.md § Sprint 1). Sprint 0 ships the off-the-shelf encoder only.

## B4. Exact-key channel

Params from research.md §3.2 table, "Keys" row.

- [ ] **B4.1 — Implement the three key types** (research.md §3.2): `(postcode, house#)`, `(postcode, rare name token)`, `(Double Metaphone of name_core, postcode prefix)`. Uses Owner A's `postcode`, `addr_numbers`, `name_core`, phonetic key (A2.5, A2.7) and token IDF (A2.3).
- [ ] **B4.2 — Cap collisions at 200 per key** (research.md §3.2 "cap 200 per key") — guards the "Keys with huge collisions" runtime blow-up in research.md §5.
- [ ] **B4.3 — Never key on a placeholder.** Owner A's `missing` flag (A4) must suppress key generation — io_rules.md §4: never let two records match *because* they share a placeholder token.

## B5. RRF fusion

- [ ] **B5.1 — Implement RRF exactly** (research.md §3.2 Fusion row): `score = Σ 1/(60 + rank)`.
- [ ] **B5.2 — Keep top-K = 20–30, plus all key hits with ≤ 5 collisions** (research.md §3.2 Fusion row). K tuning itself is Sprint 1.
- [ ] **B5.3 — Emit the fused set as the final candidate set.** This is what becomes `candidate_pairs.tsv` — and per io_rules.md §5.2 it must be the **last** candidate list the model runs inference over, **not an early blocking pass you later filter**. Coordinate with C (who consumes it) and D (who writes it).

## B6. PC / RR reporter with PC-vs-K curve

Metric definitions are exact — research.md §3.2 "Audit metrics".

- [ ] **B6.1 — Pair Completeness:** `PC = |C ∩ M| / |M|`.
- [ ] **B6.2 — Reduction Ratio:** `RR = 1 − |C| / (|S1| × |S2 ∪ S3|)`.
- [ ] **B6.3 — PC-vs-K curve** (roadmap.md Sprint 0 deliverable + research.md §3.2) plus the **distribution of candidates per fragment**.
- [ ] **B6.4 — Report per country, vendor and fold** (research.md §3.2 "report per country, vendor and fold").
- [ ] **B6.5 — Per-entity recall ceiling:** report the **share of S1 rows whose full true set lies inside C** (research.md §3.2: "The recall ceiling of the final score is per-entity"). This is the number that actually bounds the leaderboard score, distinct from pair-level PC.
- [ ] **B6.6 — Wire the reporter to grouped CV folds** from Owner D: `GroupKFold(5)` with groups = S1 entity ∪ its fragments, blocked additionally by postcode/city so neighbouring lookalikes stay together (research.md §8.3a, §5 "CV leakage").

## B7. Harness integration

- [ ] **B7.1 — Single entry point** that runs all four channels + RRF and emits the candidate set, so Sprint 1 can tune channels behind one interface.
- [ ] **B7.2 — Time and memory profile on a 10% sample** (research.md §5 "Runtime blow-ups" test column).
- [ ] **B7.3 — Record the Sprint 0 baseline PC@K on proxy data.** Not gated in Sprint 0 (Gate 1 is where PC ≥ 99% at K ≤ 30 bites), but the number is the Sprint 1 starting point.

---

## Must satisfy (io_rules.md rules applicable to Owner B)

- [ ] **Read all input through Owner A's `read_source()`** (§1) — never a bare `pd.read_csv`. Flags: `sep="\t"`, `dtype=str`, `keep_default_na=False`, `na_filter=False`, `quoting=csv.QUOTE_NONE`, `encoding="utf-8"`, `on_bad_lines="warn"`.
- [ ] **Never strip ID prefixes** (§2, §8): full IDs are the key everywhere; `S2-00001` ≠ `S3-00001`.
- [ ] **Country is an open set** (§3, §8): no filtering, no one-hot to `{US, India}`. Unseen/missing country must still flow through and appear in the submission. Sharding by country is gated on B0.1 and must never exclude unseen labels.
- [ ] **Placeholders never create candidates** (§4): `""`, `"-"`, `"0"`, `"NA"`, `"None"`, `"null"`, `"n/a"` → `missing` flag; no match *because* of a shared placeholder.
- [ ] **`candidate_pairs.tsv` is the FINAL candidate set** (§5.2): the last list the model runs inference over, not an early pass later filtered. Schema `source1_entity_id`, `candidate_entity_ids`; comma-separated, **no quoting, no spaces** around commas; no duplicate IDs in a list; no duplicate S1 rows.
- [ ] **Final matches must be a subset of candidates** (§5.2, §8): every ID in `matching_results.tsv` must appear in `candidate_pairs.tsv` for that S1 row. A violation signals a pipeline bug.
- [ ] **No transitive closure / union-find** (§8, research.md §3.4): merging two S1 entities through one fragment is metric death. Blocking emits candidates, never merged components.
- [ ] **No external data** (§9): no geocoders, registries, gazetteer-backed parsers. Postcode/house-number keys come from Owner A's regex extraction only.
- [ ] **Model licence + param count logged** (§9): final models MIT/Apache-2.0 and ≤ 8B. Avoid Qwen3-8B (8.2B total per its HF card, research.md §9). Llama-3.1-8B and gemma-2-9b are **NOT allowed**.
- [ ] **Reproducible from raw TSV** (§9): fixed seeds, pinned deps, pinned model commits.

---

## Gate 0 — pass/fail (roadmap.md § Sprint 0, verbatim criteria)

> **Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

| # | Criterion | PASS / FAIL |
|---|---|---|
| 1 | Pipeline runs **end-to-end on proxy data** | ☐ PASS ☐ FAIL |
| 2 | It emits a **validator-PASS** submission | ☐ PASS ☐ FAIL |
| 3 | Scorer **reproduces a hand-computed F0.5 on a toy set exactly** | ☐ PASS ☐ FAIL |

Gate 0 is a single shared checkpoint for all four owners. Owner B's contribution: the blocking harness produces a candidate set that C can score and D can write as `candidate_pairs.tsv`, with matches ⊆ candidates holding.

**Not gated here:** PC ≥ 99% at K ≤ 30 is **Gate 1** (roadmap.md § Sprint 1), along with the rule that if dense adds < 0.3 PC over sparse+keys it is dropped from blocking (kept as a feature).
