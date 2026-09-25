# Roadmap — Business Entity Resolution

Sprint plan, owners, and go/no-go gates. Companion to `research.md` (full design) and `io_rules.md` (I/O rules). `[VERIFY]` = confirm against real data/rules.

**Two tracks:** a **Build phase** (the weeks before the event — build a *data-agnostic* pipeline on proxy/synthetic data) and a **Competition phase** (the 72-hour execution window on the real data). `[VERIFY]` the real event window and team cap from the official rules today — research flags Internshala's 25–27 Sep 2026 window and a 2–4 member cap.

---

## Owners

| Owner | Area |
|---|---|
| **A** | Normalisation / data audit / rule lists (US, IN, FR) |
| **B** | Blocking / candidate generation / PC-RR reporting |
| **C** | Matching model (features, LightGBM, cross-encoder, calibration) |
| **D** | Decoding / infra / scorer / validator / reproducibility |

---

## Guiding priorities

1. **Blocking recall** sets the ceiling — you can't match what you never retrieve.
2. **Lookalike precision** wins F0.5 — false merges cost more than misses.
3. **Singletons + calibration** — one wrong match on a true singleton = a full 1.0 lost.
4. Build **safe I/O + validator + local scorer before any modeling**, so every experiment ends in a valid, self-scored file.

---

## Build phase (pre-event — you have this time now)

### Sprint 0 — Foundations (do these first, in parallel)
| Owner | Deliverable |
|---|---|
| A | `read_source()` safe TSV reader + the 3 sanity checks; normaliser skeleton with US/IN/FR suffix & abbreviation rule lists; placeholder detector |
| B | Blocking harness: char-TF-IDF + BM25 + FAISS + RRF; PC / RR reporter with PC-vs-K curve |
| C | Feature library (RapidFuzz string metrics, IDF, number/unit features); LightGBM OOF trainer scaffold; cross-encoder trainer scaffold |
| D | **Exact local macro-F0.5 scorer** (singleton rule); **submission validator wrapper**; N1/N2 code; pinned-dependency images |

**Proxy data:** generate synthetic S1/S2/S3 by corrupting business lists *you create yourself* (do **not** ship external data into training for the real event). Use it to exercise the whole pipeline end-to-end.

**Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

### Sprint 1 — Blocking to target
- B: dense encoder contrastive fine-tune (S1-sibling hard negatives), fusion, K tuning.
- A: vendor noise stats, learned noise-channel augmentation.
- C: baseline GBDT on sparse channel; add IDF + number + listwise features.
- D: N2 exclusivity + empty-default decoding; calibration per regime.

**Gate 1:** PC ≥ 99% at K ≤ 30 on grouped CV (proxy). If dense adds < 0.3 PC over sparse+keys, drop it from blocking (keep as feature).

### Sprint 2 — Matching + decoding
- C: S1-sibling hard negatives in training; full feature set; cross-encoder on the 0.05–0.95 band; stacking.
- D: N1 expected-F0.5 decoder vs tuned per-regime thresholds (EUM); leave-one-country-out (France proxy); prior-shift correction.
- A: FR rule check; transliteration/Unicode handling.

**Gate 2:** cross-encoder must add ≥ 0.3 CV F0.5 over the GBDT stack, or it's dropped. Plug-in decoder must beat EUM thresholds on grouped CV *and* leave-one-country-out.

### Sprint 3 — Hardening
- All: slice analysis (chains, shared address, generic names, landmark-only, non-ASCII, t∈{0,1,2–4,≥5}).
- D: reproducibility run (one command, raw TSV → both outputs); compliance checklist; audit bundle.
- A: methodology doc draft.

**Gate 3:** every edge-case slice has a mitigation with a passing test; `make submit` rebuilds both files from raw data; compliance checklist green.

---

## Competition phase (72-hour execution on real data)

| Window | A: Data | B: Blocking | C: Matching | D: Decode/Infra |
|---|---|---|---|---|
| **H0–6** | run full data audit (§10 of research) | first blocking run, PC@K curve | baseline GBDT on sparse channel | scorer on train folds; **first safe submission** |
| **H6–24** | vendor noise stats, noise channel | dense fine-tune, fusion, K tuning | sibling negatives, full features | N2 + N1, calibration per regime |
| **H24–48** | FR rule check vs test sample | final `candidate_pairs.tsv` | cross-encoder on band, stacking | leave-one-country-out, prior-shift |
| **H48–72** | methodology doc | reproducibility run | freeze models | **final 2 submissions**, audit bundle |

**Competition go/no-go:**
- After **H6**: local scorer reproduces the leaderboard within ±0.002 on the first submission `[VERIFY vs LB]`.
- After **H24**: PC ≥ 99% at K ≤ 30 on real data.
- After **H48**: cross-encoder adds ≥ 0.3 CV points or is dropped; final variant chosen from **CV, not the public board**.

---

## Priority order if time runs short

Ship in this order — each step alone yields a valid, scored submission:

1. Safe I/O + validator + exact scorer
2. Sparse blocking + exact keys at PC ≥ 98%
3. LightGBM with listwise + number features + S1-sibling negatives
4. N2 exclusivity + empty-default thresholds
5. Calibration + N1 expected-F0.5 decoder
6. Dense blocking channel
7. Cross-encoder on the ambiguous band
8. LoRA LLM tier / N3 alias bootstrapping (only if they clear their ablation gate)

---

## Leaderboard hygiene
- At most **1 exploratory submission per major ladder step**.
- Pick the final model from **your CV**, not the public leaderboard (private board = full test set).
- Submit two: (i) best-CV model, (ii) a more conservative higher-abstention hedge.
- Always run the validator to **PASS** before uploading — never spend a submission on a format error.

---

## Standing risks to watch
France calibration (unseen country) · pandas NA traps · libpostal/offline-parser compliance `[VERIFY]` · the 72-hour crunch · negatives that don't match the real blocking distribution · transitive/union-find false merges.