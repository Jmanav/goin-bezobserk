# Sprint 0 — Foundations

Build-phase sprint 0 of the Amazon ML Challenge 2026 Business Entity Resolution build. Companion to `docs/research.md` (full design), `docs/roadmap.md` (sprints + gates) and `docs/io_rules.md` (binding I/O rules).

**Only Sprint 0 is planned here.** Sprints 1–3 and the competition-phase (H0–72) plan are deliberately not broken down yet — they come after Gate 0 clears.

---

## Sprint goal

Roadmap.md § Sprint 0 — Foundations: *"do these first, in parallel"*.

Stand up a **data-agnostic** pipeline skeleton on **proxy data** that runs end to end and produces a format-valid, self-scored submission — before any modeling begins. Per roadmap.md § Guiding priorities #4:

> Build **safe I/O + validator + local scorer before any modeling**, so every experiment ends in a valid, self-scored file.

Sprint 0 is explicitly **not** about score. No PC target, no CV F0.5 target — those are Gates 1 and 2. Sprint 0 is about having working machinery and a trustworthy scorer.

**Proxy data** (roadmap.md § Sprint 0): generate synthetic S1/S2/S3 by corrupting business lists **you create yourself**. Do **not** ship external data into training for the real event (io_rules.md §9 — any external lookup ⇒ disqualification).

---

## Owners and task files

| Owner | Area (roadmap.md § Owners) | Task file | Task groups |
|---|---|---|---|
| **A** | Normalisation / data audit / rule lists (US, IN, FR) | [`owner-A-tasks.md`](owner-A-tasks.md) | A0–A5 |
| **B** | Blocking / candidate generation / PC-RR reporting | [`owner-B-tasks.md`](owner-B-tasks.md) | B0–B7 |
| **C** | Matching model (features, LightGBM, cross-encoder, calibration) | [`owner-C-tasks.md`](owner-C-tasks.md) | C0–C8 |
| **D** | Decoding / infra / scorer / validator / reproducibility | [`owner-D-tasks.md`](owner-D-tasks.md) | D0–D8 |

Each file opens with a `[VERIFY]` **blocking-task** section (A0/B0/C0/D0). Those are decisions and confirmations, **not coding tasks** — flagged separately so nobody writes dependent code on an unresolved assumption.

---

## Dependencies between owners

**Critical path: D → everyone.** Owner D owns two of Gate 0's three criteria (validator-PASS, scorer-exact) and the shared CV folds. io_rules.md §10 puts the writer, validator wrapper and scorer at steps 3–4, *before* normalisation/blocking/matching.

```
A2.1 normaliser output contract ──┬──> B1/B2/B4  (blocking keys & channel inputs)
  (name_norm, name_core,          └──> C1–C3     (string / number / suffix features)
   name_tokens, addr_norm,
   addr_numbers, postcode,
   landmark_flag, script_flag,
   folded ASCII)

A1 read_source() ─────────────────────> B, C, D  (all input goes through it; never bare read_csv)
A2.3 S1 token IDF ────────────────┬──> B4.1  (rare-name-token key)
                                  └──> C2    (IDF / rarity features)
A2.7 Double Metaphone ────────────────> B4.1  (phonetic key)
A3.2 suffix agree/conflict ───────────> C3.6  (suffix conflict feature)
A4 placeholder `missing` flag ────┬──> B4.3  (never key on a placeholder)
                                  └──> C     (never match on a placeholder)
A5 proxy data ────────────────────────> D8.1  (the end-to-end run needs data)

B1.4/B2.3/B3.4 per-channel score+rank ─> C1.3  (channel-score & rank features)
B5.3 final fused candidate set ────┬──> C6.3  (negatives MUST come from real blocking output)
                                   └──> D3.3  (candidate_pairs.tsv content)
B6 PC/RR reporter ────────────────────> (Gate 1 input, not Gate 0)

C6.6 calibrated OOF probabilities ────> D5.1  (N2 needs calibrated p_rc)
C8.2 per-regime isotonic ─────────────> D6    (N1 is "medium risk — needs calibration")

D2 GroupKFold(5) folds ───────────┬──> B6.6  (PC/RR reported per fold)
                                  └──> C6.2  (OOF training)
D3 submission writer ─────────────────> Gate 0 criterion 2
D1 exact scorer ──────────────────────> Gate 0 criterion 3, and every owner's experiments
D5 N2 marginals ──────────────────────> D6 N1 decoder
```

### Cross-owner `[VERIFY]` hand-offs

| Question | Raised in | Owner | Consumer who needs the answer |
|---|---|---|---|
| Q1 real timeline & team cap | research.md §11 | **D** (D0.1) | everyone — re-plan if 72 h is confirmed |
| Q2 offline parser = external data? | research.md §11, §3.1 | **A** (A0.1) | A3 address parsing stays regex-only until answered |
| Q3 fragment → ≤ 1 S1? | research.md §11, §1.1 | **A** (A0.2) | D — the whole asymmetric framing rests on it |
| Q4 within-vendor 1:1 per S1? | research.md §11, §1.1 | **A** (A0.3) | **D5.3** — bipartite branch stays **off by default** |
| Q6 cross-country matches? | research.md §11, §3.2 | **B** (B0.1) | B — country sharding stays **off by default** |
| Q7 `candidate_pairs.tsv` schema | research.md §11 | **D** (D0.2) | **RESOLVED** — io_rules.md §5.2 list shape, confirmed by the real ground-truth file |
| Q8 test text for DAPT? | research.md §11, §3.6 | **C** (C0.1) | C — default train-only |
| Q9 "≤ 8B" incl. embeddings? | research.md §11, §9 | **C** (C0.2) | C — default yes ⇒ avoid Qwen3-8B |

**RESOLVED (D0.2 / B0.2):** `candidate_pairs.tsv` uses the io_rules.md §5.2 shape — one S1 row with a comma-separated `candidate_entity_ids` list. **Evidence (2026-09-25):** the organiser's own `train_ground_truth.tsv` has header `source1_entity_id<TAB>matched_entity_ids` with n_data_rows == n_distinct_s1 == 2,206,821 — i.e. the io_rules.md §5.1 list shape, one row per S1. That settles Q7 in io_rules.md's favour; research.md §11's "two columns S1 ID, fragment ID" default is wrong.

### Parallelism

A1 (`read_source()`) and A2.1 (the normaliser output contract) are the only true blockers for B and C — publish both early and the other three owners run in parallel. D1–D4 (scorer, folds, writer, validator) depend on **nothing** from A/B/C and should start immediately.

---

## Gate 0 — the single checkpoint

Verbatim from roadmap.md § Sprint 0:

> **Gate 0:** pipeline runs end-to-end on proxy data and emits a validator-PASS submission. Scorer reproduces a hand-computed F0.5 on a toy set exactly.

| # | Criterion | Primary owner | PASS / FAIL |
|---|---|---|---|
| 1 | Pipeline runs **end-to-end on proxy data** | D8.1 (needs A5 proxy data, B5.3 candidates, C6 scores) | ☐ PASS ☐ FAIL |
| 2 | It emits a **validator-PASS** submission | D4 + D8.2 | ☐ PASS ☐ FAIL |
| 3 | Scorer **reproduces a hand-computed F0.5 on a toy set exactly** | D1.3 + D8.3 | ☐ PASS ☐ FAIL |

One shared gate for all four owners — nobody clears Gate 0 individually. The toy set for criterion 3 is research.md §1.2's consequence table, reproduced as unit tests in D1.3, plus the §1.3 worked examples in D6.8.

**Explicitly not gated here:**
- PC ≥ 99% at K ≤ 30 → **Gate 1** (roadmap.md § Sprint 1)
- Cross-encoder ≥ 0.3 CV F0.5 uplift; plug-in decoder vs EUM thresholds → **Gate 2** (roadmap.md § Sprint 2)
- Slice mitigations, `make submit` reproducibility, compliance checklist green → **Gate 3** (roadmap.md § Sprint 3)

---

## Next action today

Research.md's own closing line ("**Next action today**") and roadmap.md's header `[VERIFY]` agree: **read the official Unstop rules and FAQ for event dates, team size, submission limits and external-data wording** — Owner D, task **D0.1**. Research.md § Caveats notes the official details conflict with the assumed "more than 1 month / 4+ people / multi-GPU" plan, so this single answer can re-scope the entire build phase.
