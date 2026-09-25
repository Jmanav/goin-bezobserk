# I/O Rules & Things to Remember — Before We Code

Pin this. Every rule below either comes straight from the problem statement or is a trap that silently costs a submission. `[VERIFY]` = confirm against the real data/rules before relying on it.

---

## 1. Reading the input (get this wrong and everything downstream is corrupt)

**All files are TSV.** Read with an explicit tab separator, everything as string, and NA handling **off**.

```python
import csv
import pandas as pd

def read_source(path):
    return pd.read_csv(
        path,
        sep="\t",              # tabs, not commas — commas live inside addresses & ID lists
        dtype=str,             # never let pandas infer numeric IDs
        keep_default_na=False, # don't turn "NA"/"None"/"null"/"" into NaN
        na_filter=False,       # same, and faster
        quoting=csv.QUOTE_NONE,# fields are not quoted; don't let a stray " eat a row
        encoding="utf-8",
        on_bad_lines="warn",   # surface malformed rows instead of dropping silently
    )
```

Why each flag matters:
- **`sep="\t"`** — without it, pandas returns a single column with the whole line. The brief calls this out explicitly.
- **`dtype=str`** — IDs like `S2-00047` are strings; but even the numeric part must never be cast to int (leading zeros, and cross-source collisions).
- **`keep_default_na=False, na_filter=False`** — a business literally named `"NA"`, `"None"`, or an empty address would otherwise become `NaN` and either vanish or wrongly match other `NaN`s.
- **`quoting=csv.QUOTE_NONE`** — addresses contain commas and may contain stray quotes; default quoting can merge/split fields.

**Sanity checks right after reading (do these every run):**
- `len(df) == wc -l(file) - 1` (row count matches minus header).
- Every row has exactly **4 fields** (`entity_id, business_name, business_address, country`). Count tabs per raw line before trusting the parse.
- No unexpected `NaN` anywhere (`df.isna().sum().sum() == 0`).

---

## 2. Columns & the ID namespace

Each source file has exactly these columns:

| Column | Notes |
|---|---|
| `entity_id` | Prefix is the source: `S1-`, `S2-`, `S3-`. **Never strip the prefix.** |
| `business_name` | Noisy: abbreviations, legal suffixes, typos, transliterations. |
| `business_address` | Noisy: partial, abbreviations, missing components, landmark refs. |
| `country` | Open set of string labels (see §3). |

**ID rules:**
- The **prefix defines the namespace.** The bare numbers may collide across sources (`S2-00001` and `S3-00001` are different records). Keep full IDs as the key everywhere.
- Source is determined by prefix **and** by which file the record is in. There is no separate `source` column.
- `[VERIFY]` assert the three test ID sets are disjoint and every ID carries the expected prefix for its file.

---

## 3. Country rules (this is where France breaks pipelines)

- Training covers **US** and **India**. The **test set adds France**, which is **not in training**.
- Treat `country` as an **open set of string labels**. Do **not** hard-code, filter, or one-hot to `{US, India}`.
- Any record with an unseen or missing country label must still flow through the full pipeline and appear in the submission.
- Use country as a *soft/categorical* feature with an explicit `"other/unseen"` bucket — never as a branch that only has US/India paths.
- `[VERIFY]` exact string values in the data (`"France"` vs `"FR"` vs `"france"`; `"US"` vs `"USA"` vs `"United States"`). Normalize casing but keep the label.

---

## 4. Placeholders & missing values

- After turning NA-handling off, some fields will be genuine junk: `""`, `"-"`, `"0"`, `"NA"`, `"None"`, `"null"`, `"n/a"`.
- Build a **placeholder detector** → set a `missing` flag; **never** let two records match *because* they share a placeholder token.
- Empty address ≠ match signal. Empty name → lean on address only, and be conservative.

---

## 5. Output files (only one is scored — but both are audited)

You produce **two** tab-separated files in `output/`:

| File | Scored on leaderboard? | Purpose |
|---|---|---|
| `matching_results.tsv` | **Yes — the only scored file** | your final matches |
| `candidate_pairs.tsv` | No | the **final** candidate set fed to the matching model (last blocking stage), for audit |

### 5.1 `matching_results.tsv`

Columns (single tab between them):

```
source1_entity_id<TAB>matched_entity_ids
S1-00001<TAB>S2-00047,S2-00193,S3-00812
S1-00002<TAB>S3-00004
S1-00003<TAB>
```

Rules:
- **Exactly one row per Source-1 entity in the test set.** Missing an S1 row ⇒ rejection.
- `matched_entity_ids` is a **comma-separated list, no quoting, no spaces** around commas.
- **Empty** for singletons (leave the field blank — the S1 row still appears).
- **Only S2-/S3- IDs**, and only ones that **exist in the test set**. No self-matches to S1.
- **No duplicate IDs** within a list. **No duplicate `source1_entity_id` rows.**

### 5.2 `candidate_pairs.tsv`

- Same schema (`source1_entity_id`, `candidate_entity_ids`) and same formatting rules.
- This is the **last** candidate list your model runs inference over — not an early blocking pass you later filter.
- **Final matches must be a subset of candidates.** Every ID in `matching_results.tsv` must also appear in `candidate_pairs.tsv` for that S1 row. (Validator warns if not — signals a pipeline bug.)

### 5.3 Writing safely

```python
# order-insensitive but keep it deterministic; no spaces; no trailing comma
df_out["matched_entity_ids"] = df_out["ids"].apply(lambda ids: ",".join(ids))  # ids: list, deduped
df_out.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
```

`[VERIFY]` how the scorer treats list ordering and whitespace — assume **order-insensitive, trimmed**, but submit a known-answer probe if a submission is cheap.

---

## 6. Metric reminders (macro F0.5 — precision-heavy)

Per S1 entity: **F0.5 = 1.25·tp / (0.25·t + p)**, then **averaged over all S1 rows** (singletons included).

Burn these into your head:
- **True singleton + empty prediction = 1.0.** True singleton + *any* prediction = **0.0**.
- One false match on an otherwise-correct single pair drops it 1.0 → 0.556.
- Missing one of two true matches: 1.0 → 0.833. Adding one wrong to a correct pair: 1.0 → 0.714. ⇒ **false merges hurt more than misses.**
- Every S1 row counts equally: a correct singleton (1.0) outscores 3-of-4 on a big cluster (0.9375).
- **When in doubt, don't merge.** Empty is the safe default.
- Public leaderboard = subset; **private = full test set** revealed after. Tune on your **own CV scorer**, not the public board.

---

## 7. Validate before every submission

- Run `utils/validate_submission.py` (stdlib-only helper the organizers ship):

```bash
python3 utils/validate_submission.py \
  --matching  output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir  dataset/test
```

- It must print **PASS (exit 0)** before you upload. It checks format only — it does **not** compute your score.
- Build your **own** local macro-F0.5 scorer (exact singleton rule) on a held-out split so you never spend a submission just to learn your score.

---

## 8. Silent killers (checklist)

- [ ] Read without `sep="\t"` → single-column garbage.
- [ ] `NA`/`None`/empty coerced to NaN → dropped rows or fake matches.
- [ ] Stripped ID prefixes → cross-source ID collisions.
- [ ] Country filtered/one-hot to {US, India} → France silently degrades or disappears.
- [ ] Missing S1 rows in `matching_results.tsv` → rejection.
- [ ] Duplicate IDs in a list, or duplicate S1 rows → rejection.
- [ ] A matched ID not present in `candidate_pairs.tsv` → pipeline bug flag.
- [ ] Matched ID that isn't in the test set, or an S1 self-match → rejection.
- [ ] Transitive closure / union-find merging two S1 entities through one fragment → false merges (metric death).
- [ ] "Always predict ≥1 match" (Shopee-style min-k) → zeroes every true singleton.
- [ ] Tuning to the public leaderboard → private-board shake-up.
- [ ] Negatives in training that don't come from the real blocking output → train/inference mismatch.

---

## 9. Compliance rules to keep in view while coding

- **Pure ML.** No external databases, APIs, geocoders, business-registry lookups, or internet data augmentation. Only the provided data. Any external lookup ⇒ disqualification.
- Offline gazetteer-backed parsers (e.g., libpostal) are a **`[VERIFY]` risk** — treat as external data by default; don't wire them into the scored pipeline without a written organizer OK.
- Final model must be **MIT/Apache-2.0** licensed and **≤ 8B parameters**. Log the license + param count of every model you use.
- Keep the pipeline **reproducible** from raw TSV (fixed seeds, pinned deps, one command to rebuild both output files) — top teams' packages get audited.

---

## 10. Minimal pre-coding TODO

1. `read_source()` helper + the 3 sanity checks (§1).
2. ID/namespace + country audit (§2, §3) → confirm the `[VERIFY]`s.
3. Submission **writer** + **validator wrapper** (§5, §7) — build these *before* any modeling so every experiment ends in a valid file.
4. Local **macro-F0.5 scorer** with the singleton rule (§6).
5. Only then: normalization → blocking → matching → decoding.