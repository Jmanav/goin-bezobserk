# Sprint 0 — Owner C: Matching Model (Features, LightGBM, Cross-Encoder, Calibration)

**Roadmap Sprint 0 deliverable (verbatim):**
> Feature library (RapidFuzz string metrics, IDF, number/unit features); LightGBM OOF trainer scaffold; cross-encoder trainer scaffold

Owner C area: Matching model (features, LightGBM, cross-encoder, calibration) — roadmap.md § Owners.

**Sprint 0 scope note:** this is the **feature library + two trainer scaffolds**. The *baseline GBDT run* is Sprint 1; *S1-sibling hard negatives, the full feature set, the cross-encoder on the band, and stacking* are Sprint 2 (roadmap.md § Sprint 1, § Sprint 2). Build the machinery; don't chase score yet.

Target: LightGBM on **~80 engineered features** (research.md §0, §3.3).

---

## C0. Blocking tasks (`[VERIFY]` — not coding tasks)

- [ ] **C0.1 — Can test text be used unsupervised (DAPT)? (Q8, research.md §11 + §3.6).** Owner C per research.md §11. Domain-adaptive pretraining = 1 epoch of MLM or contrastive SimCSE-style training on the *provided* unlabeled name+address strings, **test included**. Research.md §3.6 flags `[VERIFY rule wording on using test text unsupervised; conservative: train-only]`. **Default until answered: train-only.** Resolve via rules/organisers.
- [ ] **C0.2 — Does "≤ 8B" count total params incl. embeddings? (Q9, research.md §11 + §9).** Owner C per research.md §11. Default: **YES → avoid Qwen3-8B** (8.2B total / 6.95B non-embedding per its HF model card). Blocks any Stage-C LLM tier model choice.
- [ ] **C0.3 — Cross-encoder band share (research.md §3.3 `[VERIFY]`).** Pairs with `0.05 < p_A < 0.95` are **estimated 5–15% of candidates**. Unmeasurable until a Stage-A model exists (Sprint 1). Consequence now: size the cross-encoder scaffold's batching for a 5–15% slice, and keep the band bounds configurable.
- [ ] **C0.4 — Within-vendor 1:1 per S1 (Q4, owned by A → A0.3).** Affects whether C's listwise features can assume one fragment per vendor per S1. Default: **Unknown** → do not assume it.

---

## C1. Feature library — string metrics

All from research.md §3.3 "Stage A, LightGBM" feature list. Consume Owner A's normaliser contract (A2.1): `name_norm`, `name_core`, `addr_norm`, `name_tokens`, `addr_numbers`, `postcode`, `landmark_flag`, `script_flag`, folded ASCII variant.

- [ ] **C1.1 — RapidFuzz metrics (MIT, research.md §9)** on `name_norm`, `name_core` **and** `addr_norm` (research.md §3.3): `ratio`, `partial_ratio`, `token_set_ratio`, `token_sort_ratio`, Jaro-Winkler, Levenshtein-normalised.
- [ ] **C1.2 — Use `process.cpdist` for bulk computation** (research.md §6): budget 30–90 min on 32 vCPU for ~150M pairs (K=25 × 6M). **Stream in chunks** — research.md §6 and §5 "Runtime blow-ups".
- [ ] **C1.3 — Channel-score features** (research.md §3.3): char-n-gram TF-IDF cosine, dense cosine, BM25 score, **and the ranks from each channel** — consumed from Owner B's per-channel score+rank output (B1.4, B2.3, B3.4).

## C2. Feature library — IDF / rarity

- [ ] **C2.1 — IDF-weighted token Jaccard** (research.md §3.3), using Owner A's S1 token IDF (A2.3).
- [ ] **C2.2 — Sum of IDF of shared vs unshared tokens** (research.md §3.3).
- [ ] **C2.3 — Rarest shared token's IDF** (research.md §3.3) — the Fellegi–Sunter / Splink-style TF adjustment **expressed as a feature**. Per research.md §2.4, Splink's TF-adjusted log-likelihood ideas are used as *features*, **not as the matcher**.

## C3. Feature library — number / unit / distinguishing tokens

Research.md §3.3 "Distinguishing-token features". Ditto (research.md §2.4) is direct evidence that **explicit number tokens matter for business addresses**.

- [ ] **C3.1 — Numbers in name:** `"#2"`, `"Unit 5"` (research.md §3.3).
- [ ] **C3.2 — House numbers equal / different / missing** (research.md §3.3) — three-state, not a boolean; "missing" is distinct from "different" (io_rules.md §4).
- [ ] **C3.3 — Unit mismatch** (research.md §3.3) — targets the "Suite 200 vs 210" edge case (research.md §5 "Numeric/unit mismatch": tokenisers split digits, so compare the **parsed** unit/house fields exactly).
- [ ] **C3.4 — Postcode equal / prefix / different** (research.md §3.3).
- [ ] **C3.5 — Branch words:** `"north"`, `"airport"`, `"mall"` (research.md §3.3) — targets chains/franchises (research.md §5).
- [ ] **C3.6 — Legal-suffix agreement or conflict** (research.md §3.3, §3.1) — consumes Owner A's suffix feature (A3.2), which is a **conflict signal, not a deletion**.
- [ ] **C3.7 — Acronym match:** initials of one name equal a token of the other (research.md §3.3) — targets "IBM vs International Business Machines" (research.md §5).

## C4. Feature library — listwise / competition features

Research.md §3.3 "Competition features (listwise)". Ranked #3 in the novelty stack at **+0.5–2 estimated points**, low risk (research.md §4): the model sees "is there a better S1 for this fragment?"

- [ ] **C4.1 — Candidate rank among this fragment's S1 candidates** (research.md §3.3).
- [ ] **C4.2 — Score gap to the best other S1** (research.md §3.3).
- [ ] **C4.3 — Number of S1s sharing the same address** (research.md §3.3) — the shared-address/mall mitigation (research.md §5).
- [ ] **C4.4 — Number of S1s sharing the same `name_core`** (research.md §3.3) — the **chain indicator** (research.md §5 chains/franchises).
- [ ] **C4.5 — Cross-vendor triangle support** (research.md §3.3): best score to any fragment from the *other* vendor that itself strongly matches this S1. **Feature only** — research.md §3.4: "S2↔S3 fragment similarity is used only as a *feature* (triangle support), never as a transitive link."

## C5. Feature library — categoricals

- [ ] **C5.1 — Vendor ID (S2/S3) as categorical** (research.md §3.3, §3.7). Per research.md §3.7, start with vendor as a categorical feature + per-vendor calibration — cheap, captures most of the gain.
- [ ] **C5.2 — Country as categorical with an explicit `"other/unseen"` bucket** (research.md §3.3, §5; io_rules.md §3). Hard rule: country **must not be the only split path**. No one-hot to `{US, India}`.
- [ ] **C5.3 — Per-vendor noise statistics as features** (research.md §3.7): field missingness, abbreviation rate, address length. Owner A produces the stats in Sprint 1; C defines the feature slots now.
- [ ] **C5.4 — Feature registry / manifest.** ~80 features (research.md §3.3) across C1–C5 need names, dtypes and an ablation switch each — research.md §8.1 runs a drop-feature ablation ladder, and §4 rank 3 requires a listwise drop-feature ablation.

## C6. LightGBM OOF trainer scaffold

Start params are exact — research.md §3.3.

- [ ] **C6.1 — Start params:** `num_leaves = 255`, `learning_rate = 0.05`, `min_data_in_leaf = 100`, `feature_fraction = 0.7`, early stopping on grouped CV (research.md §3.3). Note: Foursquare 7th place found **much larger trees helped at 1.5M-row scale** (`num_leaves = 2^12`, lr 0.1, ~2,000 iterations, validation AUC still rising) — so **tune upward** (research.md §2.1, §3.3).
- [ ] **C6.2 — OOF wiring on grouped CV:** `GroupKFold(5)`, groups = S1 entity ∪ its fragments, **blocked additionally by postcode or city** so neighbouring lookalikes stay in the same fold (research.md §8.3a; §5 "CV leakage" — compare random vs grouped CV gap). Fold definitions come from Owner D.
- [ ] **C6.3 — Negatives come from the real blocking output** (research.md §3.6): "every non-matching candidate from the *same* blocking pipeline, so the training distribution equals the inference distribution. **This is the most common silent bug.**" Also io_rules.md §8. Wire the trainer to consume Owner B's candidate set directly.
- [ ] **C6.4 — FP-weighted training hook** (research.md §2.1, §2.5): Foursquare 7th place weighted samples by the loss a mistake would cause, because false positives hurt more than true negatives help. Average weights ~0.8 positives / ~1.0 negatives. Under macro F0.5 the asymmetry is harsher still (io_rules.md §6).
- [ ] **C6.5 — Chunked, memory-aware inference** (research.md §2.1 "Adopt", §6): 10,000-row chunks; budget 1–3 h train, 30–60 GB.
- [ ] **C6.6 — Emit OOF predictions in the shape the calibrator and decoder need** — isotonic regression is fit **on out-of-fold predictions** (research.md §3.3), and Owner D's N2/N1 consume calibrated per-candidate probabilities (research.md §3.4, §3.5).
- [ ] **C6.7 — Leave S1-sibling hard negatives as a hook, don't build them.** N4 is Sprint 2 (roadmap.md § Sprint 2), ranked #2 in the novelty stack at +1–3 points (research.md §4). Spec for later: for each positive (a, r), add r paired with S1 entities b near a (same postcode, same `name_core`, top dense neighbours of a) — **guaranteed negatives because S1 is deduplicated** (research.md §3.6).

## C7. Cross-encoder trainer scaffold

Scaffold only in Sprint 0; the band run and stacking are Sprint 2 (roadmap.md § Sprint 2).

- [ ] **C7.1 — Pick from the licence-cleared list** (research.md §3.3, §9): `microsoft/mdeberta-v3-base` (MIT, ~276M), `BAAI/bge-reranker-v2-m3` (Apache-2.0, ~568M), or `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0). **Pin the commit; log licence + param count** (io_rules.md §9).
- [ ] **C7.2 — Input format, exact** (research.md §3.3): `name_a [SEP] addr_a </s> name_b [SEP] addr_b`, with **Ditto-style tags around numbers** (`"[NUM] 221 [/NUM]"`).
- [ ] **C7.3 — Train params, exact** (research.md §3.3): 2–3 epochs, lr 2e-5, max_len 128, balanced hard negatives.
- [ ] **C7.4 — Band gating configurable:** runs **only** on `0.05 < p_A < 0.95` (research.md §3.3), estimated 5–15% of candidates per C0.3. Budget per research.md §6: ~1–3 h on 4 GPUs for ~15M pairs at len 128.
- [ ] **C7.5 — CPU-only fallback that skips the cross-encoder** (research.md §6) — required for the constrained-budget path ($200 credits/participant, research.md §6).
- [ ] **C7.6 — Do NOT build Stage C (LoRA LLM).** Optional tier, include **only if it adds ≥ 0.3 CV F0.5 points over Stage B** (research.md §3.3, §4 rank 8, roadmap.md § Priority order step 8). Gated by C0.2 on model size.

## C8. Calibration + stacking scaffold

- [ ] **C8.1 — Logistic stacker over `[p_A, p_B, p_C, margin features]`** (research.md §3.3). Scaffold now; fit in Sprint 2.
- [ ] **C8.2 — Isotonic regression per regime** = vendor × country-known/unknown (research.md §3.3), fit on **out-of-fold** predictions.
- [ ] **C8.3 — France fallback path** (research.md §3.3): use the **pooled calibrator** + the prior-shift correction from research.md §8. France is test-only, so this path must exist before it can be tested (io_rules.md §3).
- [ ] **C8.4 — Adversarial-validation hook for per-vendor modelling** (research.md §3.7): build per-vendor models **only if** adversarial validation between S2 and S3 fragments reaches **AUC > 0.8** *and* per-vendor models beat pooled by **> 0.2 CV F0.5 points**. Sprint 0 ships the hook, not the models.

---

## Must satisfy (io_rules.md rules applicable to Owner C)

- [ ] **Read all input through Owner A's `read_source()`** (§1): `sep="\t"`, `dtype=str`, `keep_default_na=False`, `na_filter=False`, `quoting=csv.QUOTE_NONE`, `encoding="utf-8"`, `on_bad_lines="warn"`.
- [ ] **Never strip ID prefixes** (§2, §8): full IDs everywhere; bare numbers collide across sources.
- [ ] **Country is an open set** (§3, §8): soft/categorical with an explicit `"other/unseen"` bucket; **never** a branch that only has US/India paths; no filtering or one-hot to `{US, India}`. Unseen/missing country must still flow through the full pipeline and appear in the submission.
- [ ] **Placeholders never create matches** (§4): consume Owner A's `missing` flag; no two records match *because* they share a placeholder token. Empty address ≠ match signal. Empty name → lean on address only and be conservative.
- [ ] **Precision-heavy metric drives every threshold** (§6): `F0.5 = 1.25·tp / (0.25·t + p)`; true singleton + any prediction = **0.0**; false merges hurt more than misses; **when in doubt, don't merge**.
- [ ] **Score only the final candidate set** (§5.2): the model runs inference over `candidate_pairs.tsv` as the **last** candidate list — not an early pass later filtered. Final matches ⊆ candidates.
- [ ] **No transitive closure / union-find** (§8): cross-vendor similarity is a **feature only** (C4.5), never a transitive link.
- [ ] **No "always predict ≥ 1 match"** (§8): Shopee-style min-k zeroes every true singleton. Never a decision rule (research.md §2.2).
- [ ] **Negatives must come from the real blocking output** (§8, research.md §3.6) — otherwise train/inference mismatch.
- [ ] **Never tune to the public leaderboard** (§6, §8): private board = full test set. Tune on the team's own CV scorer.
- [ ] **No external data** (§9): only the provided data; no registries, APIs, geocoders, internet augmentation. Any external lookup ⇒ disqualification.
- [ ] **Final model MIT/Apache-2.0 and ≤ 8B params; log licence + param count for every model** (§9). Avoid Qwen3-8B (8.2B total). Llama-3.1-8B (Llama 3.1 Community) and gemma-2-9b (Gemma Terms) are **NOT allowed** (research.md §9).
- [ ] **Reproducible from raw TSV** (§9): fixed seeds, pinned deps, pinned model commits, one command to rebuild.

---

## Gate 0 — pass/fail (roadmap.md § Sprint 0, verbatim criteria)

> **Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

| # | Criterion | PASS / FAIL |
|---|---|---|
| 1 | Pipeline runs **end-to-end on proxy data** | ☐ PASS ☐ FAIL |
| 2 | It emits a **validator-PASS** submission | ☐ PASS ☐ FAIL |
| 3 | Scorer **reproduces a hand-computed F0.5 on a toy set exactly** | ☐ PASS ☐ FAIL |

Gate 0 is a single shared checkpoint for all four owners. Owner C's contribution: the feature library computes over Owner B's candidates, and the LightGBM scaffold produces per-candidate scores that Owner D's N2/N1 can decode into a submission. **Score quality is not gated at Gate 0** — Gate 2 is where the cross-encoder must add ≥ 0.3 CV F0.5 or be dropped (roadmap.md § Sprint 2).
