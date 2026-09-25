# Sprint 0 — Owner D: Decoding / Infra / Scorer / Validator / Reproducibility

**Roadmap Sprint 0 deliverable (verbatim):**
> **Exact local macro-F0.5 scorer** (singleton rule); **submission validator wrapper**; N1/N2 code; pinned-dependency images

Owner D area: Decoding / infra / scorer / validator / reproducibility — roadmap.md § Owners.

**Owner D is the critical path.** Roadmap.md § Guiding priorities #4: "Build **safe I/O + validator + local scorer before any modeling**, so every experiment ends in a valid, self-scored file." io_rules.md §10 puts the writer, validator wrapper and scorer at steps 3–4, **before** normalisation/blocking/matching. Roadmap.md § Priority order step 1 is "Safe I/O + validator + exact scorer". Gate 0's criteria 2 and 3 are **both** D's deliverables.

---

## MEASURED FROM THE REAL TRAINING DATA (2026-09-25) — read before D5/D6

`scripts/probe_cardinality_script.py` over all 2,206,821 train S1 rows. These
numbers **replace the estimates research.md §1.2 was written against**, and they
invert the decoder's central assumption.

| quantity | research.md assumed | measured | consequence |
|---|---|---|---|
| singleton rate (t=0) | "say, 40%" (§1.2) | **5.58%** | singleton handling is worth ~0.056 of the score, not ~0.40 |
| t=1 | — | 5.40% | |
| t=2-4 | — | **63.0%** | the dominant regime |
| t>=5 | — | **26.0%** | §5's "large cluster" slice is a quarter of all rows |
| mean t | — | **3.46** (max 11) | |
| ground-truth shape | Q7 disputed | **one row per S1, list shape** | io_rules.md §5.2 is right; research.md §11 Q7's "two columns" default is wrong |
| singleton rate US / India | — | 5.583% / 5.588% | near-identical; not a country effect |

**What this changes for D5/D6:**

- **The empty set is no longer the right default.** research.md §3.5 makes `{}`
  the default and searches `best_k` upward from 0. With mean t = 3.46, the
  optimal `k` is ≥ 2 for the large majority of rows. Keep the `k = 0` branch
  exactly as specified (D6.2 is still correct), but do not tune the decoder as
  though abstention were the common case — it is 5.6% of rows.
- **λ_null falls sharply** (D5.1). 94.4% of S1 rows have at least one match, so
  the prior that a fragment is an orphan is far lower than a 40%-singleton world
  implies. Fit it from the measured orphan rate, never from the §1.2 example.
- **Precision pressure now lives in the t=2-4 band, not in singletons.** By
  §1.2's own table, at t=2 missing one match costs 0.167 while adding one wrong
  costs 0.286. That asymmetry is what the decoder must get right for 63% of rows.
- **Re-derive, do not inherit, the "no min-k" rule.** research.md §2.2 rejects
  Shopee-style min-k forcing because it "would zero every true singleton" — an
  argument worth 0.40 under the old assumption and 0.056 under the real one. The
  conclusion is probably still right, but the stated reasoning no longer carries
  it, so D must re-check it on CV rather than cite §2.2.
- **D1.4's slice reporting matters more, not less.** With 26% of rows at t >= 5,
  the `t >= 5` slice is a first-class regime, not a tail.

---

## D0. Blocking tasks (`[VERIFY]` — not coding tasks)

- [ ] **D0.1 — Real timeline & team cap (Q1, research.md §11 + roadmap.md header).** Owner D per research.md §11, **resolve by: read official rules today.** Internshala: 25 Sep 2026 9:00 AM IST – 27 Sep 2026 9:00 PM IST; Top 50 announced 2 Oct 2026. Unstop: **2–4 members** vs Internshala's **3–4** — confirm the cap. Unstop: top 10 invited to a virtual Grand Finale 7 Oct 2026 (Internshala: 10:00 AM–3:00 PM IST). Research.md § Caveats notes the official details **conflict** with the stated "more than 1 month / 4+ people / multi-GPU" assumptions. This is research.md's own "**Next action today**".
- [x] **D0.2 — `candidate_pairs.tsv` schema (Q7, research.md §11).** Owner D per research.md §11, resolve by: sample file / rules. **Conflict to settle:** io_rules.md §5.2 says same schema as `matching_results.tsv` (`source1_entity_id`, `candidate_entity_ids` — one S1 row with a comma-separated list); research.md §11 Q7 records a competing default ("two columns S1 ID, fragment ID"). Also research.md §9: `[VERIFY required column names/format]`. **RESOLVED: use io_rules.md §5.2.** **Evidence (2026-09-25):** the organiser's own `train_ground_truth.tsv` has header `source1_entity_id<TAB>matched_entity_ids` with n_data_rows == n_distinct_s1 == 2,206,821 — i.e. the io_rules.md §5.1 list shape, one row per S1. That settles Q7 in io_rules.md's favour; research.md §11's "two columns S1 ID, fragment ID" default is wrong. Tell Owner B (B0.2).
- [ ] **D0.3 — Public LB fraction / submission limit (Q10, research.md §11).** Owner D, resolve by: platform. Unknown. Affects leaderboard hygiene (roadmap.md: at most 1 exploratory submission per major ladder step) and whether the known-answer probe in D0.4 is affordable.
- [ ] **D0.4 — Scorer treatment of whitespace / list ordering (Q11, research.md §11 + io_rules.md §5.3).** Default assumption: **order-insensitive, trimmed.** Resolve by submitting a known-answer probe **only if a submission is cheap** (gated by D0.3). Keep the writer's output deterministic regardless.
- [ ] **D0.5 — Local scorer vs leaderboard agreement.** Roadmap.md § Competition go/no-go after H6: local scorer reproduces the leaderboard within **±0.002** `[VERIFY vs LB]`. Cannot be checked pre-event — record it as the first competition-phase gate, not a Sprint 0 task.
- [ ] **D0.6 — Within-vendor 1:1 per S1 (Q4, owned by A → A0.3).** Default **Unknown** ⇒ code N2's bipartite-matching branch as **off by default** (research.md §3.4 `if VERIFY(one_per_vendor_per_S1)`).

---

## D1. Exact local macro-F0.5 scorer

This is Gate 0 criterion 3. Formula and singleton rule are exact — research.md §1.2, io_rules.md §6.

- [x] **D1.1 — Implement the per-entity formula exactly:** `F0.5 = 1.25·tp / (0.25·t + p)` for `t + p > 0`, **and `F0.5 = 1.0` when `t = p = 0`** (the singleton rule). Derived in research.md §1.2 from `F_β = (1+β²)·tp / ((1+β²)·tp + β²·fn + fp)` with β = 0.5.
- [x] **D1.2 — Macro-average over **all** S1 rows, singletons included** (io_rules.md §6, research.md §1.2: "Rows are macro-averaged, so a large cluster counts the same as a singleton").
- [x] **D1.3 — Encode research.md §1.2's consequence table as unit tests** — this **is** the hand-computed toy set for Gate 0 criterion 3. Every row must reproduce exactly:

  | True t | Prediction | tp | p | Expected F0.5 |
  |---|---|---|---|---|
  | 0 | {} | 0 | 0 | **1.000** |
  | 0 | {x} (any) | 0 | 1 | **0.000** |
  | 1 | {} | 0 | 0 | 0.000 |
  | 1 | {correct} | 1 | 1 | 1.000 |
  | 1 | {correct, wrong} | 1 | 2 | 0.556 (1.25/2.25) |
  | 1 | {wrong} | 0 | 1 | 0.000 |
  | 2 | {1 correct} | 1 | 1 | 0.833 (1.25/1.5) |
  | 2 | {both} | 2 | 2 | 1.000 |
  | 2 | {both, 1 wrong} | 2 | 3 | 0.714 (2.5/3.5) |
  | 4 | {1 correct} | 1 | 1 | 0.625 (1.25/2) |
  | 4 | {3 correct} | 3 | 3 | 0.9375 (3.75/4) |

- [x] **D1.4 — Slice-aware reporting** (research.md §8.2): country, vendor, `t ∈ {0, 1, 2–4, ≥ 5}`, chain vs non-chain, shared address, landmark-only, generic-name, non-ASCII. Also report **singleton accuracy** separately (research.md §8.1). Note research.md §1.2's "up to 0.40 of the score" was an illustrative guess; the measured singleton rate is **5.58%**, so the `t=2-4` (63%) and `t>=5` (26%) slices carry the score instead.
- [x] **D1.5 — Order-insensitive, trimmed comparison** (io_rules.md §5.3 `[VERIFY]`, D0.4) so the scorer matches the assumed grader behaviour.

## D2. Grouped CV split machinery

Owners B and C both consume these folds (B6.6, C6.2), so D owns them.

- [ ] **D2.1 — `GroupKFold(5)`** with groups = **S1 entity ∪ its fragments** (research.md §8.3a).
- [ ] **D2.2 — Additionally block by postcode or city** so neighbouring lookalikes stay in the same fold (research.md §8.3a, §5 "CV leakage").
- [ ] **D2.3 — Report the random-vs-grouped CV gap** as the leakage test (research.md §5 "CV leakage" test column).
- [ ] **D2.4 — Leave-one-country-out split scaffold** (research.md §8.3b): train US → eval India and the reverse, as the **France proxy**. Used in Sprint 2 (roadmap.md § Gate 2), scaffolded now.

## D3. Submission writer

io_rules.md §5 is the binding format spec. Build this **before** any modeling (io_rules.md §10 step 3).

- [x] **D3.1 — `matching_results.tsv` writer** (io_rules.md §5.1): columns `source1_entity_id<TAB>matched_entity_ids`; **exactly one row per Source-1 entity in the test set** (a missing S1 row ⇒ rejection); comma-separated list with **no quoting, no spaces** around commas; **blank field for singletons** (the S1 row still appears).
- [x] **D3.2 — Write safely, exactly as specified** (io_rules.md §5.3): `",".join(ids)` over a **deduped list**, then `to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")`. Deterministic ordering, no trailing comma.
- [x] **D3.3 — `candidate_pairs.tsv` writer** (io_rules.md §5.2): same schema (`source1_entity_id`, `candidate_entity_ids`) and same formatting rules. Schema gated by D0.2. Consumes Owner B's **final** fused candidate set (B5.3).
- [x] **D3.4 — Both files land in `output/`** (io_rules.md §5).

## D4. Submission validator wrapper

This is Gate 0 criterion 2. The organisers ship a stdlib-only helper — D wraps it (roadmap.md Sprint 0: "**submission validator wrapper**").

- [x] **D4.1 — Wrap the shipped validator** (io_rules.md §7), invoked as:
  ```bash
  python3 utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
  ```
  It must print **PASS (exit 0)** before any upload. It checks **format only** — it does **not** compute the score (io_rules.md §7).
- [x] **D4.2 — Add the team's own pre-flight assertions** (research.md §9, io_rules.md §5 + §8): row count = `|S1_test|`; unique S1 IDs; no duplicate IDs within a list; all matched IDs exist in the S2/S3 **test** set; **only S2-/S3- IDs, no S1 self-matches**; empty string for no match; tab separator; header matches the sample.
- [x] **D4.3 — Assert matches ⊆ candidates** (io_rules.md §5.2, §8; research.md §9): every ID in `matching_results.tsv` must appear in `candidate_pairs.tsv` for that S1 row. A violation **signals a pipeline bug** — surface it loudly, not as a warning.
- [ ] **D4.4 — Make validation non-bypassable in the submit path** (io_rules.md §7, roadmap.md § Leaderboard hygiene: "never spend a submission on a format error"). Wire this into the repo's `/submit` skill.

## D5. N2 — fragment exclusivity + NULL

Ranked **#1** in the novelty stack: +1–3 estimated points, **low risk** (research.md §4). Pseudocode is given verbatim in research.md §3.4 — implement it, don't redesign it.

- [x] **D5.1 — Per-fragment odds and NULL weight** (research.md §3.4): for each fragment `r` in S2 ∪ S3 with candidates `C` and calibrated `p_rc` — `w_c = p_rc / (1 - p_rc)`; `w_null = 1.0 * lambda_null[regime]` (tuned on CV, the prior of a fragment being an orphan); `Z = w_null + Σ w_c`; `pi[r,c] = w_c / Z`; `pi[r,NULL] = w_null / Z`.
- [x] **D5.2 — Why no global solver** (research.md §3.4): with a deduplicated master the only hard cross-row constraint is "each fragment matches at most one S1", and that **decomposes per fragment**. Do not build a global optimiser.
- [ ] **D5.3 — Optional bipartite branch, OFF by default** (research.md §3.4, gated by D0.6/A0.3): if within-vendor 1:1 holds, then per vendor, per connected component of `(S1, frag_v)` edges with `pi > 0.05`, solve max-weight bipartite matching with NULL dummies via `scipy.optimize.linear_sum_assignment` on `-log-odds` (components are small); zero out `pi[r,c]` for unmatched pairs when `pi[r,c] < tau_keep`.
- [x] **D5.4 — Final guard** (research.md §3.4): if a fragment is selected by more than one S1 after decoding, keep `argmax pi`.
- [x] **D5.5 — Emit per-S1 candidate marginals** into the N1 decoder (research.md §3.4).
- [ ] **D5.6 — CC-vs-exclusivity ablation harness** (research.md §4 rank 1 falsifiable ablation): connected components vs exclusivity on grouped CV; **count S1 pairs sharing a fragment**. This is the evidence that exclusivity beats CC (research.md §3.4: CC merges S1-a and S1-b whenever one fragment links to both).

## D6. N1 — exact expected-F0.5 decoder

Ranked #4: +0.3–1.5 points, **medium risk (needs calibration)** (research.md §4). Pseudocode verbatim in research.md §3.5.

- [x] **D6.1 — Poisson-binomial DP** (research.md §3.5): `P_in = poisson_binomial_pmf(q[:k])` via **O(k²) DP**; `P_out = convolve(poisson_binomial_pmf(q[k:]), poisson_pmf(lam, max=m))`.
- [x] **D6.2 — The `k == 0` branch is the singleton rule:** `return P_out[0]` — F = 1 iff T = 0 (research.md §3.5). Get this exactly right; it is where the free 1.0 on true singletons comes from.
- [x] **D6.3 — The expectation loop** (research.md §3.5): skip `a == 0` (F = 0 when tp = 0 and p = k > 0); otherwise accumulate `pa*pb * 1.25*a / (0.25*(a+b) + k)`.
- [x] **D6.4 — `best_k = argmax_{k in 0..m}`, top-k optimal under independence** (research.md §3.5). Inputs: marginals `q_1..q_m` from N2, sorted desc, with `q_i >= floor = 0.02`.
- [x] **D6.5 — Empty set is the default** (research.md §0, §3.5; io_rules.md §6: "When in doubt, don't merge. Empty is the safe default").
- [ ] **D6.6 — Unseen-mass λ estimation** (research.md §3.5): estimate from out-of-fold data as `E[#true matches not in C]` per stratum (country × vendor × name-rarity bucket). For France, use the **pooled estimate inflated by the PC drop observed in leave-one-country-out**.
- [x] **D6.7 — Performance:** `m ≤ 10` after the floor ⇒ ~10³ ops per S1; at 1–3M S1 rows that is **seconds in numba or vectorised numpy** (research.md §3.5; §6 budgets decode at < 5 min CPU). Exact DP beats Monte-Carlo here — use MC only if modelling correlated labels.
- [x] **D6.8 — Verify the decoder against research.md §1.3's worked examples** as unit tests:
  - **A (single candidate):** π = 0.6 → predict {x} (0.6 > 0.4); π = 0.4 → predict {}. **Break-even is exactly 0.5.**
  - **B (two candidates):** π₁ = 0.9, π₂ = 0.5 → {} = 0.05, {1} = **0.825**, {1,2} = 0.728 ⇒ the second candidate at 0.5 is **rejected**. Confirms the effective threshold for adding a second match is above 0.5 — "the precision tilt that a single global threshold cannot express."
- [ ] **D6.9 — EUM robustness hedge** (research.md §3.5, §4 rank 4, per Ye et al. 2012): also fit per-regime thresholds tuned directly on out-of-fold macro-F0.5. Pick whichever wins on grouped CV **and** leave-one-country-out. Scaffold now; the bake-off is Gate 2 (roadmap.md § Sprint 2).

## D7. Pinned-dependency images

- [ ] **D7.1 — `requirements.lock` with recorded versions** (research.md §9): FAISS (MIT), RapidFuzz (MIT), Splink (MIT), LightGBM, sklearn, pandas/Polars. Log **licence + param count of every model** used (io_rules.md §9).
- [ ] **D7.2 — Prove no network calls at inference** (research.md §9): run with outbound network blocked (e.g. `docker --network none`) **and prove it in the README**.
- [ ] **D7.3 — Fixed seeds everywhere** (research.md §9, io_rules.md §9).
- [ ] **D7.4 — One command rebuilds both output files from raw TSV** (`make submit`) with **logged hashes of inputs and outputs** (research.md §9, io_rules.md §9). Full reproducibility run is Gate 3; the command exists in Sprint 0.
- [ ] **D7.5 — AWS image with pinned deps** (research.md §7 P0 row). Budget context (research.md §6): $200 free AWS credits per participant, +$100 for Top 500 teams at the 48-hour mark — ~$800 for a team of 4, i.e. **tens of A10G/L4 GPU-hours, not unlimited multi-GPU**. Use spot instances; cache embeddings to Parquet.
- [ ] **D7.6 — CPU-only fallback profile** (research.md §6): a path that skips the cross-encoder entirely.

## D8. End-to-end proxy run (owns Gate 0)

- [ ] **D8.1 — Wire A → B → C → D into one runnable pipeline** on Owner A's proxy data (roadmap.md § Sprint 0 "Proxy data").
- [ ] **D8.2 — Produce both output files and run the validator to PASS** (Gate 0 criterion 2).
- [ ] **D8.3 — Run the scorer against the hand-computed toy set and confirm exact reproduction** (Gate 0 criterion 3).
- [ ] **D8.4 — Record the Sprint 0 baseline CV macro-F0.5** as the first rung of the research.md §8.1 ablation ladder.

---

## Must satisfy (io_rules.md rules applicable to Owner D)

- [ ] **Read all input through Owner A's `read_source()`** (§1): `sep="\t"`, `dtype=str`, `keep_default_na=False`, `na_filter=False`, `quoting=csv.QUOTE_NONE`, `encoding="utf-8"`, `on_bad_lines="warn"`.
- [ ] **Never strip ID prefixes** (§2, §8): full IDs are the key everywhere; `S2-00001` ≠ `S3-00001`.
- [ ] **Country is an open set** (§3, §8): any record with an unseen or missing country label must still **appear in the submission**. No filtering, no one-hot to `{US, India}`.
- [ ] **`matching_results.tsv` format, exactly** (§5.1): exactly one row per test S1 entity (missing row ⇒ **rejection**); comma-separated `matched_entity_ids` with **no quoting, no spaces**; **blank for singletons**; only S2-/S3- IDs that **exist in the test set**; **no S1 self-matches**; no duplicate IDs in a list; no duplicate `source1_entity_id` rows.
- [ ] **`candidate_pairs.tsv`** (§5.2): same schema and formatting; it is the **final** candidate set the model ran inference over, not an early pass.
- [ ] **Final matches ⊆ candidates** (§5.2, §8): assert it; a violation is a pipeline bug.
- [ ] **Write safely** (§5.3): `to_csv(sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")`; deduped lists; deterministic; no trailing comma.
- [ ] **Exact metric, exact singleton rule** (§6): `F0.5 = 1.25·tp / (0.25·t + p)`; macro-averaged over all S1 rows including singletons; true singleton + empty = 1.0, true singleton + any prediction = 0.0.
- [ ] **Validator must PASS (exit 0) before every upload** (§7, §8) — never spend a submission on a format error.
- [ ] **Build the team's own local scorer** (§7) so a submission is never spent just to learn a score.
- [ ] **No transitive closure / union-find** (§8): N2 exclusivity replaces connected components.
- [ ] **No "always predict ≥ 1 match"** (§8): min-k forcing zeroes every true singleton.
- [ ] **Never tune to the public leaderboard** (§6, §8): public = subset, private = full test set. Pick the final model from CV. At most 1 exploratory submission per major ladder step; submit two finals — best-CV and a higher-abstention hedge (roadmap.md § Leaderboard hygiene).
- [ ] **No external data** (§9): only provided data; no registries, APIs, geocoders, internet augmentation ⇒ disqualification risk.
- [ ] **Final model MIT/Apache-2.0, ≤ 8B params; log licence + param count of every model** (§9).
- [ ] **Reproducible from raw TSV** (§9): fixed seeds, pinned deps, **one command** to rebuild both output files; packages get audited.

---

## Gate 0 — pass/fail (roadmap.md § Sprint 0, verbatim criteria)

> **Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

| # | Criterion | Owner D task | PASS / FAIL |
|---|---|---|---|
| 1 | Pipeline runs **end-to-end on proxy data** | D8.1 | ☐ PASS ☐ FAIL |
| 2 | It emits a **validator-PASS** submission | D4, D8.2 | ☐ PASS ☐ FAIL |
| 3 | Scorer **reproduces a hand-computed F0.5 on a toy set exactly** | D1.3, D8.3 | ☐ PASS ☐ FAIL |

Gate 0 is a single shared checkpoint for all four owners. **Two of the three criteria are Owner D's deliverables** — D is the critical path for clearing the gate.
