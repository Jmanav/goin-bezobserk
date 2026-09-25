# Amazon ML Challenge 2026 — Business Entity Resolution: Build Plan (research.md)

The pipeline most likely to reach the top 10 treats this as a fragment-to-master assignment problem, not as clustering. For every S2/S3 record, retrieve a small set of S1 candidates, score each pair with a calibrated GBDT plus a cross-encoder, pick at most one S1 per fragment (with an explicit NULL option), and then choose each S1's output set to maximise expected F0.5, with an empty set as the default. One scheduling fact overrides the rest: Internshala's listing puts the ML Challenge at 25 September 2026, 9:00 AM IST to 27 September 2026, 9:00 PM IST, and Unstop says each team must have a minimum of 2 and a maximum of 4 members (Internshala's listing says 3–4, so confirm the cap). That makes your "more than a month" the build window, not the competition window.

**TL;DR**
- **Winning recipe:** normalise text with rules only; block by querying each fragment against the S1 master using BM25/char-n-gram, a dense ANN index and exact keys, merged with RRF. Score pairs with LightGBM on string and rarity features, then run a permissively licensed cross-encoder only on ambiguous pairs. Normalise probabilities per fragment with a NULL option, then decode each S1 row with an exact expected-F0.5 rule where empty is the default. Skip union-find transitive closure and Shopee-style "min-k" forcing; both lose points under this metric.
- **What decides the leaderboard:** (1) recall of blocking at a small candidate set, (2) precision on lookalikes (branches, chains, shared addresses, near-identical names), trained with S1-sibling hard negatives, and (3) calibrated probabilities feeding singleton-aware decoding. Under macro F0.5, one false match on a true singleton costs a full 1.0 for that row.
- **Your deciders:** check the real event timeline and team cap. Internshala lists the challenge as 25 Sep 2026 9:00 AM IST – 27 Sep 2026 9:00 PM IST, with the Top 50 announced on 2 Oct 2026; Unstop says 2–4 members and that the top 10 teams are invited to a virtual Grand Finale on 7 October 2026 (Internshala gives 10:00 AM–3:00 PM IST). Get a written ruling on offline address parsers (default: don't use libpostal), and keep all final models ≤8B and MIT/Apache-2.0. Qwen3-8B's Hugging Face model card lists 8.2B parameters (6.95B non-embedding), so prefer Qwen3-4B or smaller.

---

## 0. TL;DR of the plan (winning recipe + 3 deciders)

1. Frame the task as asymmetric linkage. Each S2/S3 fragment belongs to at most one S1 entity or to none. S1 records are mutual cannot-links.
2. Blocking is done fragment → S1 index, as a union of BM25 on name and address char n-grams, a dense ANN index (Qwen3-Embedding-0.6B or multilingual-e5, fine-tuned contrastively on the training pairs), and exact keys (postcode, house number, rare name token). RRF merges them; target ≥99% pair completeness at ≤30 S1 candidates per fragment.
3. Scoring is a cascade. LightGBM runs on around 80 engineered features (string similarities, IDF-weighted token overlap, number/unit agreement, rank and margin versus the other S1 candidates, vendor ID). The cross-encoder (mDeBERTa-v3-base or bge-reranker-v2-m3) sees only the 0.05–0.95 band. The two are stacked, with isotonic calibration per vendor and country regime.
4. Inference: a per-fragment softmax over {S1 candidates ∪ NULL}, exclusivity enforcement, then an exact expected-F0.5 set decoder per S1 that accounts for the chance a true match was blocked out.
5. The three deciders are blocking recall, lookalike precision, and calibration plus singleton handling. Everything else is secondary.

---

## 1. Problem analysis

### 1.1 Structure (verified from the task brief; inferences flagged)
- There are three sources. S1 is a clean, deduplicated master. S2 and S3 are noisy vendor fragments. There is no shared ID. The only fields are `business_name`, `business_address` and `country`.
- Output is one row per S1 entity listing matched S2/S3 IDs, which may be empty. The leaderboard scores only `matching_results.tsv`. `candidate_pairs.tsv` is used to audit the recall ceiling and reduction ratio, and final matches must be a subset of it.
- **Inference (asymmetric framing):** because S1 is deduplicated, a fragment should correspond to at most one S1 entity. This makes the problem a many-to-one assignment (fragments → S1 ∪ {NULL}), not general clustering. [VERIFY: in train, count fragments whose ID appears in more than one S1 row of the ground truth; expect 0.]
- **Inference (possible within-vendor uniqueness):** a vendor may also hold at most one record per S1 entity. If so, the problem becomes one-to-one per vendor and bipartite matching applies. [VERIFY: for each S1 entity in train, count matched S2 IDs and matched S3 IDs separately; if the max is 1, the within-vendor one-to-one constraint holds.]
- Country is an open set. Test adds France, which is absent from training. No code path may filter, one-hot encode, or hard-code {US, India}.

### 1.2 Metric math (exact)
Let T be the true set for an S1 entity (t = |T|), P the predicted set (p = |P|), and tp = |T ∩ P|. With β = 0.5, F_β = (1+β²)·tp / ((1+β²)·tp + β²·fn + fp). Substituting fn = t − tp and fp = p − tp:

**F0.5 = 1.25·tp / (0.25·t + p)** for t + p > 0, and **F0.5 = 1.0 when t = p = 0** (singleton rule).

Consequences (derived, exact):
| True t | Prediction | tp | p | F0.5 |
|---|---|---|---|---|
| 0 | {} | 0 | 0 | **1.000** |
| 0 | {x} (any) | 0 | 1 | **0.000** |
| 1 | {} | 0 | 0 | 0.000 |
| 1 | {correct} | 1 | 1 | 1.000 |
| 1 | {correct, wrong} | 1 | 2 | 1.25/2.25 = 0.556 |
| 1 | {wrong} | 0 | 1 | 0.000 |
| 2 | {1 correct} | 1 | 1 | 1.25/1.5 = 0.833 |
| 2 | {both} | 2 | 2 | 1.000 |
| 2 | {both, 1 wrong} | 2 | 3 | 2.5/3.5 = 0.714 |
| 4 | {1 correct} | 1 | 1 | 1.25/2 = 0.625 |
| 4 | {3 correct} | 3 | 3 | 3.75/4 = 0.9375 |

What this means:
- **Singletons are all-or-nothing.** A true singleton predicted as empty earns a free 1.0; one spurious link costs the full 1.0. **[MEASURED 2026-09-25 — this assumption was wrong.]** The real singleton rate is **5.58%**, not the ~40% this paragraph hypothesised, so singleton decisions swing ~0.056 of the score rather than ~0.40. The real distribution is t=0 5.58%, t=1 5.40%, **t=2-4 63.0%**, **t>=5 26.0%**, mean t=3.46, max 11, and US/India are near-identical (5.583%/5.588%). This is a **many-matches** problem: the empty set is the wrong answer for 94% of rows, so the decoder is mostly choosing a set *size*. Precision pressure lives in the t=2-4 band, where missing one match costs 0.167 and adding one wrong costs 0.286.
- **Asymmetry in numbers:** for t = 2, missing one match costs 0.167, while adding one wrong match to a correct pair costs 0.286. For t = 1, adding a wrong match costs 0.444.
- **Rows are macro-averaged**, so a large cluster counts the same as a singleton. Getting 3 of 4 right on a big entity (0.9375) is worth less than one correctly empty singleton (1.0).

### 1.3 Expected F0.5 of a decision (the basis for N1)
With per-candidate match probabilities π_i and U = number of true matches outside the candidate set, predicting a set S of size k gives E[F] = E[1.25·TP_S / (0.25·(TP_S + R) + k)], where TP_S counts true matches inside S and R counts true matches outside S (including U). The empty set gives E[F] = P(T = 0).

**Worked example A (one candidate, no unseen mass):** π = 0.6. Predicting {} gives 0.4; predicting {x} gives 0.6·1.25/1.25 = 0.6, so predict. With π = 0.4 the choice flips to empty. The single-candidate break-even is exactly 0.5.

**Worked example B (two independent candidates):** π₁ = 0.9, π₂ = 0.5.
- {}: 0.1·0.5 = 0.05
- {1}: 0.45·(1.25/1.25) + 0.45·(1.25/1.5) = 0.45 + 0.375 = **0.825**
- {1,2}: 0.45·1 + 0.45·(1.25/2.25) + 0.05·(1.25/2.25) = 0.45 + 0.25 + 0.028 = 0.728

So the second candidate at 0.5 is rejected. Once one match is already predicted, the effective threshold for adding another is above 0.5. This is the precision tilt that a single global threshold cannot express.

**Worked example C (blocked-out mass):** if CV shows that 1% of true matches for this stratum fall outside the candidates, P(T = 0) for a "nothing retrieved" S1 drops slightly. Predicting empty is still optimal, because nothing can be predicted outside the candidates, but the expected score of the row falls. Unseen mass matters mostly because it lowers the value of {} relative to weak candidates.

### 1.4 Hidden traps
- `pandas.read_csv` turns "NA", "None", "NaN", "null" and "" into NaN by default. A business called "NA" or a blank address can vanish or merge. Always pass `sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE`.
- TSV fields may contain stray quotes or tabs. Count fields per line before parsing.
- ID prefixes (S1-/S2-/S3-) define the namespace. Never strip the prefix, since the bare numeric IDs may collide across sources.
- The public leaderboard is a subset, and a separate private leaderboard is computed on the complete test set after the event. Tuning to the public board overfits.
- France is only in test. Any rule, feature or dictionary tuned only to US and India will quietly degrade there.

---

## 2. Competitor landscape & lessons from similar competitions

### 2.1 Kaggle Foursquare Location Matching (2022): the closest precedent
**Setting.** The Kaggle data held more than 1.5M place entries (per the hobbitlab repository's summary of the data page), noised and duplicated. Matching used name, address, coordinates and other attributes. The metric was mean per-row IoU (Jaccard) between predicted and true match sets. The Kaggle competition page lists 1,079 teams and 22,050 submissions, and Foursquare's blog counts 1,290 data scientists.

**What top teams did (verified):**
- **1st place (team re:waiwai: Takoi, Pao, Charmq):** a three-stage process plus post-processing and an XLM-RoBERTa model. Stage 1 generated candidates by text similarity and geographic proximity, then used a limited-feature LightGBM to cut candidates memory-efficiently. The top 40 candidates per ID went to a stage-2 LightGBM with a richer feature set: Levenshtein and Jaro-Winkler distances, statistics and ratios, Euclidean distance on coordinates, and SVD-reduced name embeddings. Stages 3–4 added BERT-style transformers, iteratively tuned on CV and leaderboard.
- **Yuki Uehara (top finisher, highlighted by Foursquare):** four stages. Candidates came from text similarity and geography, followed by LightGBM filtering, pairwise classification with two transformer models, and GNN node-classification post-processing. He reported moving from 0.907 to 0.946 with this framework.
- **7th place (Future Architect write-up):** LightGBM binary classification on query–candidate pairs, 5-fold CV split by POI ID, and num_leaves = 2^12 with learning rate 0.1 for about 2,000 iterations. Validation AUC was still rising at that point. They **weighted samples by the IoU loss a mistake would cause**, stating explicitly that false positives hurt the score more than true negatives help it, because many rows had no pair. Average weights came out at about 0.8 for positives and 1.0 for negatives. For memory, they generated features and predicted in 10,000-row chunks and held sparse BoW matrices on the GPU.
- **Summaries of top solutions (Zhihu digest):** candidate generation combined several sources: nearest neighbours, name similarity, shared words, category, cleaned phone, same address, and TF-IDF, which was noted as especially useful for airports and Indian places. Teams kept about 20 candidates per pattern (around 40 in total) and used RAPIDS Forest Inference to speed up GBDT inference.
- **4th place (Viel/Matiounine/S.):** an ensemble of LightGBM, CatBoost and XGBoost. The published repository notes the fuller pipeline ran out of memory at inference, so the submitted version used fewer candidates. Runtime and memory limits shaped the final candidate count.

**Adopt:** multi-source candidate generation; a cheap first-stage GBDT to prune; a feature-rich second-stage GBDT; a transformer only on survivors; group K-fold by entity; FP-weighted training; chunked, memory-aware inference.
**Reject or modify:** Foursquare's symmetric dedup produced a graph, and many teams merged it transitively or with GNNs. Here S1 is already deduplicated, so any merge through a fragment that joins two S1 entities is by construction a false merge. Replace connected components with the fragment → S1 assignment in §3.4. The IoU metric also counted a row's own ID, while ours gives 0 to any prediction on a true singleton, so the precision pressure is harsher.

### 2.2 Kaggle Shopee Price Match Guarantee (2021)
**Setting.** Product listings matched by image and title; per-item F1.
**What worked (verified from public solution repos):** ArcFace (ArcMarginProduct) metric-learning heads on image and text encoders, KNN in embedding space, and a cosine threshold. Two widely used post-processing tricks were "min2" (always predict at least two items, since each item's group contains itself) and INB (iterative neighbourhood blending, a form of query expansion). One public reproduction reports test F1 of 0.8211 → 0.8285 after min2 → 0.8345 after INB.
**Transfer:** metric learning for the dense blocking encoder; threshold tuning on CV.
**Reject:** "min-k" forcing. Its equivalent here, "always predict at least one," would zero every true singleton. **[MEASURED: this argument is much weaker than written.]** At a 5.58% singleton rate it costs ~0.056, not ~0.40. The conclusion is probably still right, but it must be **re-derived on CV rather than inherited from this paragraph**. Use INB-style neighbourhood expansion only as a *candidate-generation* feature (see N3), never as a decision rule.

### 2.3 SIGMOD Programming Contests (2020–2022)
- **2022 (blocking):** submissions were ranked by average recall at a fixed candidate budget; the output file had to contain 3,000,000 pairs, 1M for X1 and 2M for X2, with running time as a tiebreaker. The winner was team WBSG (Brinkmann and Peeters, Mannheim), out of 55 teams. They embedded records with a transformer pre-trained by supervised contrastive learning, indexed the embeddings in FAISS, ran nearest-neighbour search, and **re-ranked the retrieved pairs with a symbolic similarity metric**.
- **Transfer:** this maps directly onto our `candidate_pairs.tsv` audit. Contrastively fine-tuned dense retrieval plus a lexical re-rank is a proven recipe for recall at a fixed budget.

### 2.4 Research benchmarks (WDC Products, Magellan/DeepMatcher, company data)
- **Ditto (VLDB 2021):** fine-tuning pre-trained LMs as sequence-pair classifiers improved F1 by up to 29% over the prior state of the art on benchmark datasets. On a real task matching company datasets of 789K and 412K records it reached 96.5% F1. Its "domain knowledge" injection tagged the first number in the address (street number) and the last four digits of the phone. That is direct evidence that **explicit number tokens matter for business addresses**.
- **Sparkly (VLDB 2023):** top-k TF/IDF (BM25, via Lucene) blocking outperformed 8 state-of-the-art blockers. TF had minimal effect on short attributes such as names and titles. A strong sparse baseline is non-negotiable.
- **UniBlocker (2024):** a self-supervised dense blocker was comparable to Sparkly and complementary to it. Ensembling the two improved pair completeness by up to 5% on one dataset. This is the argument for sparse + dense fusion.
- **LLM matchers:** Peeters, Steiner and Bizer (EDBT 2025) and Steiner et al. (ICDE-W 2025) studied prompting and fine-tuning LLMs, including open 8B models, for entity matching on WDC Products and related sets. I have not verified their specific numbers for this document. Treat LLM matchers as a candidate cross-encoder tier, not a proven winner on name+address data.
- **Splink (Fellegi–Sunter, MIT):** supports term-frequency adjustments and claims about a minute per million records on a laptop. Its own docs say it is **not designed** for a single "bag of words" company-name column. Use its TF-adjusted log-likelihood ideas as *features*, not as the matcher.

### 2.5 What most competing teams will do, and where they lose points (inference)
| Typical approach | Where it loses |
|---|---|
| TF-IDF or sentence-embedding cosine with a global threshold | Lookalikes (branches, "XYZ Pharmacy #2"); no singleton logic; poor calibration on France |
| Union-find over all pairs above threshold | Transitive bridges merging two S1 entities through one noisy fragment |
| pandas defaults | Drops "NA"/"None" names; corrupted rows; missing S1 rows in submission |
| Tuning on the public leaderboard | Private-board shake-up |
| Country one-hot for {US, India} | France breaks features or gets filtered out |
| Heavy LLM on all pairs | Runs out of time in 72 h; no calibration |

---

## 3. Recommended architecture

```
 TSV (S1,S2,S3)  --safe read (dtype=str, na_filter=False)-->  Arrow/Polars
        |
 [3.1 Normalise]  NFKC->casefold->accent-fold copy; legal-suffix/descriptor tagging;
        |          number/unit/postcode extraction; landmark phrase tagging; token IDF
        v
 [3.2 Block]  query = each S2/S3 fragment ; index = S1 (per country-agnostic shard)
   ├─ BM25/char-3-5gram TF-IDF (name, addr, name+addr)       top-50
   ├─ Dense ANN (fine-tuned embedder, FAISS GPU)              top-50
   ├─ Exact keys (postcode+house#, rare-name-token, phonetic)  all hits (capped)
   └─ RRF fusion -> top-K (K≈20-30)  ==> candidate_pairs.tsv (final audited set)
        v
 [3.3 Score]  Stage A: LightGBM (~80 feats) on all candidates
              Stage B: cross-encoder on 0.05<p<0.95 band ; stack + isotonic calib
        v
 [3.4 Collective]  per-fragment softmax over {S1 cands ∪ NULL}; exclusivity;
                   optional per-vendor 1:1 matching (if VERIFY holds)
        v
 [3.5 Decode]  per-S1 exact expected-F0.5 set selection (default = {})
        v
 matching_results.tsv  (+ validator: 1 row/S1, subset of candidates, no dups)
```

### 3.1 Normalisation & parsing
**Inputs:** raw strings. **Outputs:** `name_norm`, `name_core` (legal suffixes and generic descriptors removed), `name_tokens`, `addr_norm`, `addr_numbers` (house number, unit, floor), `postcode`, `landmark_flag`, `script_flag`, and a folded ASCII variant.
- **Unicode:** NFKC, casefold, then an accent-folded copy (keep both; French "Société Générale" vs "Societe Generale"). Normalise punctuation variants (’ ' ` “ ”) and ampersands (& ↔ and/et).
- **Legal suffixes (hand-written rule lists, which are domain knowledge and not an external lookup; document them in the methodology doc):** US Inc/Incorporated, LLC, L.L.C., Corp, Co, Ltd, LLP, PLLC. India Pvt Ltd, Private Limited, Pvt. Ltd., (P) Ltd, LLP, "M/s"/"M/S" prefix, "& Sons", "Enterprises", "Traders". France SARL, SAS, SASU, SA, EURL, SCI, SNC, "Société", "Ets"/"Établissements". Store suffixes as a *feature* (suffix agreement or conflict), not only strip them.
- **Descriptor stripping (learned, not hard-coded):** compute each token's document frequency in S1 names. Tokens above a DF percentile (e.g., "restaurant", "pharmacy", "store", "traders") get down-weighted by IDF rather than deleted. Learn from training positives which tokens often *differ* between matched pairs, e.g., vendors appending "Store" or "Branch".
- **Addresses:** rule-based extraction of numbers, units ("Ste 200", "Flat 3B", "#12", "Apt"), US ZIP (5 or 9 digits), India PIN (6 digits), France postcode (5 digits, "CEDEX"). Tag landmark phrases ("near", "opp", "opposite", "behind", "beside", "nr", "b/h", "next to", "en face de", "près de"). Expand abbreviations (St↔Street, Rd↔Road, Ave↔Avenue, Bd/Blvd↔Boulevard, Nagar, Marg).
- **Parsers:** libpostal is an MIT-licensed C library whose parser was trained on over 1B OSM/OpenAddresses examples, with a reported 99.45% full-parse accuracy on held-out data. It also uses GeoNames as a place-name and postcode gazetteer, and its model data licence is not clearly stated. **[VERIFY] ruling:** because the model bundles knowledge derived from external geographic data (gazetteers of places and postcodes), a strict reviewer could call it "external data." Conservative default: **do not use libpostal or any gazetteer-backed parser in the scored pipeline** unless organisers confirm in writing. deepparse is LGPL-3.0, which is not MIT/Apache; avoid it for the same reasons. usaddress and pyap are rule/CRF-based and US-centric; their licences are unverified here, so check them before use. Recommendation: regex plus a small CRF/token tagger **trained only on the provided training addresses** if needed.
- **Transliteration:** Indian names show spelling variants (Shri/Shree/Sri, Enterprises/Ent., Aggarwal/Agarwal). Handle them with character n-gram similarity and a phonetic key (Double Metaphone); both are algorithmic, not data. **[MEASURED 2026-09-25 — larger and broader than anticipated.]** 41.3% of S2 and 32.4% of S3 India rows carry non-Latin script (41.7%/32.9% on test, so no train->test shift), spanning **at least six scripts**: Devanagari, Malayalam, Gujarati, Tamil, Telugu, Bengali. Two patterns: a fully transliterated name (zero char-n-gram overlap with a Latin S1 record, and Double Metaphone is Latin-only, so **only the dense channel bridges these**), and a Latin name with a native-script state in the address tail. Consequences for this plan: legal suffixes are transliterated too, so the suffix lists need native-script entries; Indic vowel signs and the virama are Unicode categories Mn/Mc and must survive punctuation stripping; zero-width characters (U+200C) appear in Telugu names and must be deleted, not blanked, or they split words.

### 3.2 Blocking (target ≥99% pair completeness, small K)
**Direction:** query = each S2/S3 fragment, index = S1. This matches the many-to-one structure and bounds candidates per fragment.
| Channel | Implementation | Start params | Role |
|---|---|---|---|
| Sparse name | char 3–5-gram TF-IDF (sklearn `TfidfVectorizer(analyzer="char_wb")`) + chunked sparse top-k (sparse_dot_topn or cuML/cupy) | top-50, min cos 0.2 | typos, abbreviations |
| Sparse name+addr | word BM25 (bm25s / Lucene via Pyserini, or a custom scipy.sparse BM25) | k1 = 1.2, b = 0.75, top-50 | rare tokens |
| Dense | fine-tuned Qwen3-Embedding-0.6B (Apache-2.0) or multilingual-e5-large (MIT) / bge-m3 (MIT); FAISS GPU `IndexFlatIP` (S1 ≤ a few M) or IVF-PQ if memory-bound | top-50 | semantic/acronym/transliteration |
| Keys | (postcode, house#), (postcode, rare name token), (Double Metaphone of name_core, postcode prefix) | cap 200 per key | exact anchors |
| Fusion | RRF: score = Σ 1/(60 + rank) | keep top-K = 20–30 plus all key hits with ≤ 5 collisions | shrink |

- **Dense fine-tuning:** contrastive (MultipleNegativesRankingLoss) on training (fragment, S1) positives, with **S1-sibling hard negatives** (see N4). The SIGMOD 2022 winner's recipe was contrastive pre-training plus FAISS plus a symbolic re-rank.
- **Country handling:** use country as a *soft* feature and a shard key only if cross-country matches are verified to be 0 in training [VERIFY]. Never filter unknown country values: France fragments must query France S1 records, and anything with an unseen or missing label must query the whole index.
- **Audit metrics** (report per country, vendor and fold): Pair Completeness PC = |C ∩ M| / |M|; Reduction Ratio RR = 1 − |C| / (|S1| × |S2 ∪ S3|); plus the PC-vs-K curve and the distribution of candidates per fragment. The **recall ceiling of the final score** is per-entity. Also report the share of S1 rows whose full true set lies inside C.
- **Go/no-go:** PC ≥ 99.0% at K ≤ 30 on grouped CV. ~~If the dense channel adds less than 0.3 points of PC over sparse + keys, drop it from blocking (keep it as a feature).~~ **[MEASURED — do not apply this rule as written.]** 41% of S2 India names are non-Latin, where sparse channels retrieve *nothing*; a pooled PC delta would hide that behind the US majority. The dense channel is **mandatory**. If this rule is evaluated at all, evaluate it per country.
- **[MEASURED 2026-09-25] Blocking results on real train** (`scripts/probe_blocking.py`, S1-first samples, dense active): PC **0.99983** at 5k S1 / 22k fragments, **0.99921** at 50k S1 / 223k fragments; per-entity ceiling 0.9994 and 0.9973. **The Gate 1 target is met at K=5**, so blocking is far easier than this section assumed and Sprint 1 should spend its budget on scoring precision, not recall. India no longer lags US (0.99862 vs 0.99959).
- **[MEASURED] The exact-key channel reaches only 4.0% of fragments** (0.1 hits/fragment). The "all key hits with <= 5 collisions" exemption in the Fusion row is therefore nearly inert. Likely cause: `(postcode, house#)` needs both a parsed postcode and a house number, and the Q2 ruling left address parsing regex-only. Ablate before investing further.
- **[MEASURED] K can be reduced.** At 10M test fragments, K=25 costs 250M pairs for a ceiling of 0.9973; K=15 costs 150M for 0.9954. Trading 0.0019 of ceiling for 100M fewer pairs of Stage-A featurisation is likely worth it on a $200 budget.

### 3.3 Pair scoring
**Stage A, LightGBM** (start: `num_leaves = 255`, `learning_rate = 0.05`, `min_data_in_leaf = 100`, `feature_fraction = 0.7`, early stopping on grouped CV; Foursquare 7th place found much larger trees helped at 1.5M-row scale, so tune upward). Features:
- String metrics (RapidFuzz, MIT): ratio, partial_ratio, token_set_ratio, token_sort_ratio, Jaro-Winkler, Levenshtein-normalised, on `name_norm`, `name_core` and `addr_norm`.
- Char-n-gram TF-IDF cosine, dense cosine, BM25 score, and the ranks from each channel.
- IDF-weighted token Jaccard; the sum of IDF of shared vs unshared tokens; the rarest shared token's IDF (Fellegi–Sunter/Splink-style TF adjustment expressed as a feature).
- **Distinguishing-token features:** numbers in name ("#2", "Unit 5"), house numbers equal/different/missing, unit mismatch, postcode equal/prefix/different, branch words ("north", "airport", "mall").
- Legal-suffix agreement or conflict; acronym match (initials of one name equal a token of the other).
- **Competition features (listwise):** candidate rank among this fragment's S1 candidates, score gap to the best other S1, number of S1s sharing the same address, number of S1s sharing the same name_core (chain indicator).
- **Cross-vendor features:** best score to any fragment from the *other* vendor that itself strongly matches this S1 (triangle support).
- Vendor ID (S2/S3) and country as categorical. Country uses an "other/unseen" bucket and must not be the only split path; see §5.

**Stage B, cross-encoder** on pairs with 0.05 < p_A < 0.95 (estimated 5–15% of candidates [VERIFY]). Options, all licence-checked: `microsoft/mdeberta-v3-base` (MIT, ~276M), `BAAI/bge-reranker-v2-m3` (Apache-2.0, ~568M), `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0). Input: `name_a [SEP] addr_a </s> name_b [SEP] addr_b`, with Ditto-style tags around numbers ("[NUM] 221 [/NUM]"). Train 2–3 epochs, lr 2e-5, max_len 128, balanced hard negatives.
**Optional Stage C, LoRA LLM** (Qwen3-4B Apache-2.0, or Phi-4-mini-instruct MIT 3.8B) as a yes/no classifier with token-probability output, on the hardest ~1% only. Include it only if it adds ≥ 0.3 points of CV F0.5 over Stage B.
**Stacking and calibration:** a logistic stacker over [p_A, p_B, p_C, margin features], then isotonic regression fit on out-of-fold predictions **per regime** (vendor × country-known/unknown). For France, fall back to the pooled calibrator and apply the prior-shift correction in §8.

### 3.4 Collective inference (N2, stress-tested)
**Finding:** with a deduplicated master, the only hard cross-row constraint is "each fragment matches at most one S1." That constraint decomposes *per fragment*, so no global solver is needed. If within-vendor one-to-one also holds [VERIFY], the problem becomes a bipartite matching per vendor per connected component.

```
# N2: fragment-level exclusive assignment with NULL
for fragment r in S2 ∪ S3:
    C = candidates(r)                     # S1 ids, calibrated pairwise p_rc
    # "at most one" posterior (independence-corrected), or listwise softmax with learned NULL logit
    w_c    = p_rc / (1 - p_rc)            # odds
    w_null = 1.0 * lambda_null[regime]    # tuned on CV (prior of fragment being orphan)
    Z = w_null + sum(w_c for c in C)
    pi[r, c]    = w_c / Z                 # marginal P(r belongs to c)
    pi[r, NULL] = w_null / Z
if VERIFY(one_per_vendor_per_S1):
    for each vendor v, each connected component G of (S1, frag_v) edges with pi>0.05:
        solve max-weight bipartite matching with NULL dummies
        (scipy.optimize.linear_sum_assignment on -log-odds; components are small)
        set pi[r,c] = 0 for pairs not in the matching when pi[r,c] < tau_keep
emit pi as per-S1 candidate marginals -> N1 decoder
final guard: if a fragment is selected by >1 S1 after decoding, keep argmax pi
```
- **Why this beats connected components:** CC merges S1-a and S1-b whenever one fragment links to both. Exclusivity makes the two compete for the fragment, and NULL lets the fragment go unmatched.
- **Cross-source consistency:** S2↔S3 fragment similarity is used only as a *feature* (triangle support), never as a transitive link.

### 3.5 Metric-aware decoding (N1)
Honest novelty check: exact expected-F maximisation is well established. Lewis (1995) characterised it; Chai (2005) gave an O(n³) exact algorithm under label independence and Jansche (2007) an O(n⁴) one; Dembczyński et al. (NeurIPS 2011) gave GFM, which is exact without independence given n² + 1 distribution parameters. Ye et al. (ICML 2012) showed that decision-theoretic (plug-in) and empirical-utility (tuned threshold) approaches are asymptotically equivalent. Empirically, tuned thresholds held up better under model misspecification, while the plug-in approach did better on rare classes and in a domain-adaptation scenario. **What is specific to us:** macro averaging over S1 rows, the singleton-equals-1 rule, and unseen (blocked-out) mass. That is an application, not a new algorithm.

```
# N1: exact expected-F0.5 decoder for one S1 entity
# inputs: marginals q_1..q_m (from N2, sorted desc, q_i >= floor=0.02), unseen mass lam (CV per stratum)
def expected_F(k, q, lam):
    # TP_S ~ PoissonBinomial(q[:k]); R ~ PoissonBinomial(q[k:]) (+) Poisson(lam)
    P_in  = poisson_binomial_pmf(q[:k])            # O(k^2) DP
    P_out = convolve(poisson_binomial_pmf(q[k:]), poisson_pmf(lam, max=m))
    if k == 0: return P_out[0]                     # singleton rule: F=1 iff T=0
    E = 0
    for a, pa in enumerate(P_in):                  # a = true positives inside S
        if a == 0: continue                        # F=0 (tp=0, p=k>0)
        for b, pb in enumerate(P_out):
            E += pa*pb * 1.25*a / (0.25*(a+b) + k)
    return E
best_k = argmax_{k in 0..m} expected_F(k, q, lam)   # top-k optimal under independence
return q_ids[:best_k]
```
- **Cost:** m ≤ 10 after the floor (estimate) gives about 10³ operations per S1. At 1–3M S1 rows that is seconds in numba or vectorised numpy. Exact DP beats Monte-Carlo here, so use MC only if you model correlated labels, e.g., two fragments that are near-duplicates of each other.
- **Robustness hedge (per Ye et al.):** also fit an EUM decoder, i.e., per-regime thresholds tuned directly on out-of-fold macro-F0.5. Pick whichever wins on grouped CV *and* the leave-one-country-out split. Expect the plug-in decoder to win when calibration is good; otherwise the thresholds.
- **Unseen mass λ:** estimate from out-of-fold data as E[#true matches not in C] per stratum (country × vendor × name-rarity bucket). For France, use the pooled estimate inflated by the PC drop observed in leave-one-country-out.

### 3.6 Training-data strategy
- **Positives:** all ground-truth (S1, fragment) pairs. **Negatives:** every non-matching candidate from the *same* blocking pipeline, so the training distribution equals the inference distribution. This is the most common silent bug.
- **S1-sibling hard negatives (N4):** for each positive (a, r), add r paired with S1 entities b that are near a (same postcode or same name_core, top dense neighbours of a). Because S1 is deduplicated, these are guaranteed negatives.
- **Learned noise channel (N4b):** from training positives, estimate empirical edit operations per vendor (token drop rate, abbreviation map, suffix drop, digit transpositions, address truncation) and use them to synthesise extra positives from singleton S1 records. This is augmentation from provided data only; it is compliant, but document it.
- **Domain-adaptive pretraining:** 1 epoch of MLM or contrastive SimCSE-style training on the *provided* unlabeled name+address strings, test included, before fine-tuning. This is allowed as "provided data" [VERIFY rule wording on using test text unsupervised; conservative: train-only].
- **Pseudo-labelling on test:** only for p > 0.99 one-to-one assignments, at most one round, and only if leave-one-country-out shows a gain. It is high risk for France.

### 3.7 Vendor-aware modelling (N5)
- Start with vendor as a categorical feature plus per-vendor calibration; this is cheap and captures most of the gain (inference).
- Build per-vendor models only if adversarial validation between S2 and S3 fragments reaches AUC > 0.8 *and* per-vendor models beat the pooled model by more than 0.2 points of CV F0.5.
- Per-vendor noise statistics (field missingness, abbreviation rate, address length) also become features.

---

## 4. Novelty stack (ranked by expected gain × feasibility ÷ risk)

Gains are **my estimates** of macro-F0.5 points over a strong GBDT + global-threshold baseline, to be validated.
| Rank | Idea | Why differentiating | Prior art | Est. gain | Risk | Falsifiable ablation |
|---|---|---|---|---|---|---|
| 1 | **N2 fragment exclusivity + NULL** | Removes transitive false merges; most teams use CC | assignment/matching ER; Foursquare GNN post-proc | +1–3 | low | CC vs exclusivity on grouped CV; count S1 pairs sharing a fragment |
| 2 | **N4 S1-sibling hard negatives** | Directly attacks lookalike precision | hard-negative mining (standard); Foursquare FP-weighting | +1–3 | low | same model ± siblings; precision on same-postcode slice |
| 3 | **Listwise/margin features** (new) | Model sees "is there a better S1 for this fragment?" | Foursquare ID-level features (7th) | +0.5–2 | low | drop-feature ablation |
| 4 | **N1 expected-F0.5 decoder** | Singleton-aware, set-size-aware thresholds | Chai 2005; Jansche 2007; Dembczyński 2011; Ye 2012 | +0.3–1.5 | medium (needs calibration) | vs tuned per-regime thresholds (EUM) |
| 5 | **N5 vendor-aware calibration** | S2/S3 noise differ | domain-specific calibration (standard) | +0.2–1 | low | pooled vs per-vendor isotonic |
| 6 | **Noise-channel augmentation** (N4b) | More positives for rare patterns and France-like shift | Ditto data augmentation | +0–1 | medium | ± augmentation, leave-one-country-out |
| 7 | **N3 alias bootstrapping** | Adds matched-fragment strings to S1 profile for retrieval | query expansion / INB (Shopee) | +0–0.5 | **high** (error propagation) | only for blocking; compare PC and final F |

**N3 guardrails:** use only fragments assigned with π > 0.98. Expand the *retrieval query* only, never the scoring features. Allow one iteration. The expanded profile may never pull in a fragment that is closer to another S1 (exclusivity still applies).

**What we can honestly claim in the methodology doc:** "We cast linkage against a deduplicated master as exclusive fragment-to-master assignment with an abstain option, and decode each master's link set by maximising expected per-entity F0.5 using exact Poisson-binomial DP (following Chai/Jansche/Dembczyński), extended with a blocked-out-mass term and singleton rule." Do **not** claim a new F-measure algorithm.

---

## 5. Edge cases & failure modes

| Case | Why models fail | Mitigation | How to test |
|---|---|---|---|
| Chains/franchises (same name, different addresses) | Name similarity dominates | house#/postcode features; chain-count feature; sibling negatives | slice: S1 name_core with ≥ 3 S1 entities |
| Shared addresses (malls, office towers) | Address similarity dominates | name-core weight; count-of-S1-at-address feature | slice: addresses with ≥ 2 S1 |
| Generic names ("City Pharmacy") | Low information | IDF-weighted overlap; require address agreement | slice: max token IDF < threshold |
| One distinguishing token ("Store #12" vs "#14") | Fuzzy ratio ≈ 0.95 | explicit number-mismatch features; cross-encoder number tags | synthetic pairs from S1 siblings |
| Numeric/unit mismatch (Suite 200 vs 210) | Tokenisers split digits | parsed unit/house fields; exact compare | slice: both have numbers |
| Landmark-only addresses ("near bus stand") | No parseable street | landmark flag; lean on name + PIN | slice: landmark_flag = 1 |
| Legal-suffix confusion (X Pvt Ltd vs X LLP) | Stripped suffix loses signal | suffix as a conflict feature, not deletion | slice: both have differing suffixes |
| DBA vs legal names | Low lexical overlap | dense embedder; address weight; cross-vendor triangle | inspect FN at high address similarity |
| Acronyms (IBM vs International Business Machines) | Char n-grams fail | acronym feature; dense channel | synthetic acronym tests |
| Transliteration/Unicode (Shree/Sri; é/e) | Tokens differ | NFKC + fold; char n-grams; phonetic key | slice: non-ASCII or variant list |
| Placeholders/missing ("NA", "-", "0", empty) | Spurious matches on placeholders | placeholder detector → missing flag; never match on placeholder tokens | count strings with DF > 1% of length ≤ 3 |
| Singletons & orphans | Forced matches | NULL option; decoder default {}; λ | per-row F on singletons |
| Large clusters (S1 with many fragments) | Recall drops at small K | K is per-fragment, so it's not capped per S1 | slice: t ≥ 5 |
| Within-source dupes & transitive bridges | CC merges entities | exclusivity; no transitive links | count S1 pairs sharing fragments |
| Train→test shift (France/unseen) | Suffix/abbreviation rules missing; calibration off | FR rules; leave-one-country-out; pooled calibrator; prior-shift EM | adversarial AUC train vs test-FR |
| TSV/pandas NA traps | Rows dropped or merged | `keep_default_na=False, na_filter=False, dtype=str` | row count equals `wc -l` minus 1 |
| ID namespace collisions | Stripped prefixes collide | keep full IDs; assert prefixes | assert set disjointness |
| Submission-format errors | Missing S1 rows, dups, IDs not in test | validator script (§9) | run on every submission |
| CV leakage | Same S1 or chain in both folds | GroupKFold by S1 component + postcode block | compare random vs grouped CV gap |
| Alias-propagation errors (N3) | One wrong alias cascades | π > 0.98, retrieval-only, 1 iteration | PC/F with and without N3 |
| Runtime blow-ups | Keys with huge collisions; O(n²) fuzzy | caps per key; RRF top-K; chunking | time and memory profile on a 10% sample |

---

## 6. Compute & runtime plan (all numbers are estimates)
~~Assumption [VERIFY]: 1.3 GB at about 150 bytes per row suggests roughly 8-9M rows in total, e.g., 1-3M S1 and 5-7M fragments.~~

**[MEASURED 2026-09-25] The real scale is ~24M rows, roughly 3x this estimate:**
S1 2,206,821 train / 1,732,544 test; S2 5,034,616 / 4,887,273; S3 5,285,603 / 5,082,316.

**[MEASURED] The char-TF-IDF row below is wrong by about three orders of
magnitude.** Scaling the blocking harness from 5k to 50k S1 (fragments x9.9)
raised query time **x58**, because cost scales with S1 x fragments: every
fragment is scored against the whole S1 index. At 1.65e-7 s per fragment-S1
unit, the full test run is **~792 h (33 days) single-process**, against the
30-90 min CPU / 10-20 min GPU budgeted in the table. Any linear-in-fragments
projection understates this badly.

Fixes, cheapest first: (1) move the sparse channels to GPU (cuML/cupy or
`sparse_dot_topn`, both already named in section 3.2 -- the current
implementation is pure scipy on CPU); (2) let the FAISS dense channel carry
retrieval, since it reaches 100% of fragments alone, and use sparse only to
re-rank a shortlist (the SIGMOD-2022 recipe in section 2.3); (3) shard the S1
index, which is blocked on Q6 and must never exclude unseen labels.
| Stage | Hardware | Time (est.) | Memory (est.) |
|---|---|---|---|
| Read + normalise (Polars) | 16–32 vCPU | 5–15 min | 10–20 GB |
| Char-n-gram TF-IDF top-50 (chunked sparse matmul) | 32 vCPU or 1 GPU (cupy) | 30–90 min CPU / 10–20 min GPU | 20–40 GB |
| Dense embedding, 0.6B model, fp16, ~40 tokens | 1× A100/L4-class | 20–60 min for ~9M strings | 1024-d fp16 ≈ 2 KB/row → ~18 GB |
| FAISS GPU flat IP, S1 index, top-50 | 1 GPU | 5–20 min | S1 ≤ 3M × 2 KB ≈ 6 GB |
| RRF + keys | CPU | 10 min | — |
| Features on ~150M pairs (K = 25 × 6M) | 32 vCPU, RapidFuzz `process.cpdist` | 30–90 min | stream in chunks |
| LightGBM train (OOF 5-fold) | CPU/GPU | 1–3 h | 30–60 GB |
| Cross-encoder on ~10% band (~15M pairs) | 1–4 GPUs, mdeberta-base, len 128 | ~1–3 h on 4 GPUs | — |
| Decode (numba) | CPU | < 5 min | — |

Budget reality: the AWS Builder Center prep guide says every participant gets **$200 in free AWS credits** (with 79,000+ students registered), and mirrored copies of the Unstop listing add that the Top 500 teams receive another $100 in credits at the 48-hour mark. For a team of 4 that is about $800, i.e., tens of A10G/L4 GPU-hours, not unlimited multi-GPU. Precompute embeddings once, cache them to Parquet, and use spot instances. Keep a CPU-only fallback that skips the cross-encoder.

---

## 7. Roadmap (4-person team, parallelised)

**Critical scheduling fact [VERIFY today]:** Internshala's listing gives the ML Challenge window as 25 September 2026, 9:00 AM IST to 27 September 2026, 9:00 PM IST, and Unstop says teams get the problem statement and dataset on Day 1 and must submit by Day 3. Submissions need a 1–2 page approach document plus zipped code. Internshala lists the Top 50 announcement for 2 October 2026, and Unstop says the top 10 teams from Round 1 are invited to a virtual Grand Finale on **7 October 2026** (10:00 AM–3:00 PM IST per Internshala). So the next weeks are for building a **data-agnostic** pipeline on proxy data (e.g., synthetic S1/S2/S3 generated by corrupting public-domain-style business lists that you create yourself; do not ship external data into training for the real event). The 72 hours are then for adaptation.

| Phase | Owner A: Normalisation/Data | Owner B: Blocking | Owner C: Matching model | Owner D: Decoding/Infra |
|---|---|---|---|---|
| P0 (pre-event) | safe TSV reader, normaliser with US/IN/FR rule lists, placeholder detector | BM25 + char-TF-IDF + FAISS + RRF code, PC/RR reporter | feature library (RapidFuzz), LightGBM OOF trainer, cross-encoder trainer | exact local scorer, submission validator, N1/N2 code, AWS images with pinned deps |
| H0–6 | data audit (§10) | first blocking run, PC@K curve | baseline GBDT on the sparse channel | scorer on train folds, first submission (safe) |
| H6–24 | vendor noise stats, noise channel | dense fine-tune, fusion, K tuning | sibling negatives, full features | N2 + N1, calibration per regime |
| H24–48 | FR rule check vs test sample | final candidate_pairs.tsv | cross-encoder on band, stacking | leave-one-country-out, prior-shift |
| H48–72 | methodology doc | reproducibility run | freeze models | final 2 submissions, audit bundle |

**Priority order if time runs short:** (1) safe I/O + validator + exact scorer → (2) sparse blocking + keys at PC ≥ 98% → (3) LightGBM with listwise and number features + sibling negatives → (4) N2 exclusivity + empty-default thresholds → (5) calibration + N1 → (6) dense channel → (7) cross-encoder → (8) LLM/N3.
**Go/no-go:** after H6, the scorer must reproduce leaderboard ±0.002 on a first submission [VERIFY vs LB]; after H24, PC ≥ 99% at K ≤ 30; after H48, the cross-encoder must add ≥ 0.3 CV points or be dropped.

---

## 8. Experiment & ablation plan
1. **Ladder:** exact-key only → + sparse → + dense → GBDT(basic) → + IDF/number → + listwise → + siblings → + N2 → + calibration → + N1 → + cross-encoder → + N5 → + augmentation → + N3. Log CV macro-F0.5, PC, precision, recall, and singleton accuracy at each step.
2. **Slices:** country, vendor, t ∈ {0, 1, 2–4, ≥ 5}, chain vs non-chain, shared address, landmark-only, generic-name, non-ASCII.
3. **Validation splits:** (a) GroupKFold(5) with groups = S1 entity ∪ its fragments, blocked additionally by postcode or city so that neighbouring lookalikes stay together; (b) leave-one-country-out (train US → eval India, and the reverse) as a France proxy; (c) adversarial validation (a LightGBM classifier on pair features, train vs test) with AUC reported by country.
4. **Prior-shift correction:** estimate the test singleton rate via EM on test posteriors (a Saerens-style prior adjustment; standard technique, my knowledge rather than a sourced claim). Then adjust λ_null and decoder priors per country, with France especially important.
5. **Leaderboard hygiene:** at most 1 exploratory submission per major ladder step. Pick the final from CV, not the public board, since the private board uses the full test set. Submit (i) a best-CV model and (ii) a more conservative variant with higher abstention as the hedge.

---

## 9. Compliance & audit checklist
- [ ] No network calls at inference: run with outbound network blocked (e.g., `docker --network none`) and prove it in the README.
- [ ] No external data: no registries, geocoders, ER APIs, or scraped lists. Hand-written rule lists (suffixes, abbreviations) are documented as domain knowledge.
- [ ] **Offline parser ruling [VERIFY]:** libpostal (code MIT, but model data derived from OSM/OpenAddresses/GeoNames with an unclear licence) is **excluded by default**. deepparse (LGPL-3.0) is excluded. Ask organisers in writing before any use.
- [ ] Model licence table (verified on HF model cards, 24 Sep 2026; pin commits):

| Model | License | Params | Use |
|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | 0.6B | dense blocking |
| Qwen/Qwen3-Reranker-0.6B | Apache-2.0 | 0.6B | cross-encoder |
| BAAI/bge-m3 | MIT | ~568M | dense alt |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | ~568M | cross-encoder alt |
| intfloat/multilingual-e5-large(-instruct) | MIT | ~560M | dense alt |
| microsoft/mdeberta-v3-base | MIT | ~276M | cross-encoder |
| FacebookAI/xlm-roberta-base/large | MIT | ~278M/~560M | backbone |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | ~118M | fast dense |
| Qwen/Qwen3-4B | Apache-2.0 | 4.0B | LoRA LLM tier |
| microsoft/Phi-4-mini-instruct / Phi-3.5-mini-instruct | MIT | 3.8B | LoRA LLM alt |
| mistralai/Mistral-7B-Instruct-v0.3 | Apache-2.0 | ~7.25B | LLM alt |
| Qwen/Qwen3-8B | Apache-2.0 | **8.2B total (6.95B non-embedding, per HF model card)** | **risky vs "≤8B"; avoid** |
| meta-llama/Llama-3.1-8B | Llama 3.1 Community | ~8B | **NOT allowed** |
| google/gemma-2-9b | Gemma Terms | ~9B | **NOT allowed** |

- [ ] Libraries: FAISS (MIT), RapidFuzz (MIT), Splink (MIT); record versions in `requirements.lock`.
- [ ] Reproducibility: fixed seeds, a single `make submit` that rebuilds from raw TSV, logged hashes of the inputs and outputs.
- [ ] `candidate_pairs.tsv` = exactly the pairs scored by the final model; assert matches ⊆ candidates [VERIFY required column names/format].
- [ ] Submission validator: row count = |S1_test|; unique S1 IDs; no duplicate IDs within a list; all IDs exist in S2/S3 test; empty string for no match; tab separator; header matches the sample.
- [ ] Methodology doc (1–2 pages): pipeline diagram, licences, no-external-data statement, the N1/N2 description.

---

## 10. Data-audit checklist (run FIRST, H0–H6)
1. **Parsing:** fields per line = 4 for all lines; row counts vs `wc -l`; count of literal "NA"/"None"/"NaN"/"null"/"" per column.
2. **Namespace:** every ID has a prefix consistent with its file; the three ID sets are disjoint; ground-truth IDs ⊆ train IDs.
3. **Structure:** fragments matched to more than one S1 (expect 0); per-S1 matched counts per vendor (tests within-vendor one-to-one); S1 duplicates by exact normalised name + address.
4. **Singleton rate:** share of S1 with an empty truth set, overall and per country. This sets the stakes of §1.2.
5. **Cardinality distribution:** histogram of t per S1; share of S1 rows with t ≥ 5.
6. **Orphans:** share of fragments matched to no S1, per vendor. This sets λ_null.
7. **Country mix:** counts per source; cross-country matches in train (expect 0); unseen labels in test; exact string values (e.g., "France" vs "FR").
8. **Vendor noise:** missingness, length, abbreviation rate, and share with numbers, per vendor and country.
9. **Adversarial validation:** train vs test records (character n-gram features) by country; AUC and top features. France expected to be high; check US/India AUC too.
10. **Scale:** rows per source to finalise §6 estimates.

---

## 11. Open questions / assumptions register
| # | Question | Default assumption | Owner | Resolve by |
|---|---|---|---|---|
| Q1 | Real timeline & team cap | 25 Sep 2026 9:00 AM IST – 27 Sep 2026 9:00 PM IST (Internshala); 2–4 members per Unstop vs 3–4 per Internshala | D | read official rules today |
| Q2 | Is an offline parser (libpostal) "external data"? | **RESOLVED: yes → excluded.** Address parsing is regex-only and final | A | resolved 2026-09-25 |
| Q3 | Fragments map to ≤ 1 S1? | Holds on proxy; re-confirm on real train | A | audit #3 |
| Q4 | Within-vendor 1:1 per S1? | **RESOLVED: does NOT hold** → N2 bipartite branch stays off | A | resolved 2026-09-25 |
| Q5 | Singleton & orphan rates | **RESOLVED: 5.58% singleton, mean t=3.46** (not ~40%) — see §1.2 | A | resolved 2026-09-25 |
| Q6 | Cross-country matches exist? | No | B | audit #7 |
| Q7 | candidate_pairs.tsv schema | **RESOLVED: io_rules.md §5.2 list shape** (one row per S1, comma-separated). The organiser's own ground-truth file confirms it; this row's old default was wrong | D | resolved 2026-09-25 |
| Q8 | Can test text be used unsupervised (DAPT)? | Conservative: no | C | rules/organisers |
| Q9 | "≤ 8B" counts total params incl. embeddings? | Yes → avoid Qwen3-8B | C | rules/organisers |
| Q10 | Public LB fraction / submission limit | Unknown | D | platform |
| Q11 | Scoring treatment of whitespace/ordering in lists | order-insensitive, trimmed | D | submit a known-answer test |

---

## 12. Reference list
Sources are cited by name inline throughout: Foursquare's blog on the winners, the Future Architect 7th-place write-up, the Theo Viel 4th-place repository, the Zhihu solution digest, the Shopee public solution repositories, the SIGMOD 2022 contest archive and the Mannheim WBSG announcement, the papers on Ditto, Sparkly, UniBlocker, Chai/Jansche/Dembczyński/Waegeman and Ye et al. on F-measures, the Splink documentation, libpostal's README and Mapzen posts, the Qwen3 Embedding release, the Hugging Face model cards, and the Unstop and AWS Builder Center event pages. The source register is supplied separately by the citation layer.

---

## Caveats
- The official event details (Internshala: 25 Sep 2026 9:00 AM IST – 27 Sep 2026 9:00 PM IST; Unstop: 2–4 members, Internshala: 3–4; AWS Builder Center: $200 in free AWS credits per participant) conflict with your stated constraints ("more than 1 month," "4+ people," "multi-GPU"). I planned for the stricter official version while keeping the ambitious components as optional tiers.
- Every gain, runtime, and memory figure above is an estimate until the data audit.
- I did not verify specific benchmark numbers for the LLM matchers (Peeters/Steiner/Bizer); treat them as a tier to test.
- Kaggle write-ups for Foursquare 1st/2nd/3rd place were not directly readable; their details come from Foursquare's own summary and secondary digests.

## Chat summary
1. Frame the task as fragment → S1 assignment with NULL, not clustering; connected components and "min-k" forcing lose points under macro F0.5 with singletons.
2. Block from fragments into an S1 index with sparse + dense + keys fused by RRF; audit PC ≥ 99% at K ≤ 30.
3. Score with LightGBM (listwise, number and IDF features, S1-sibling hard negatives), then a permissive cross-encoder only on ambiguous pairs, with calibration per vendor and regime.
4. Decode with an exact expected-F0.5 rule where empty is the default. This is established theory applied to our setting, not a new algorithm.
5. Main risks: France calibration, pandas NA traps, libpostal compliance, and the 72-hour window.

**Top 3 decisions:** (1) confirm the real timeline and team cap and re-plan around 72 hours if confirmed; (2) whether to request an organiser ruling on offline parsers or simply exclude them; (3) the LLM tier: Qwen3-4B/Phi-4-mini (safe) versus Qwen3-8B (8.2B total and 6.95B non-embedding per its HF model card; licence-fine but size-risky).

**Next action today:** read the official Unstop rules and FAQ for the event dates, team size, submission limits and external-data wording, then assign owners A–D to the P0 pre-event build.