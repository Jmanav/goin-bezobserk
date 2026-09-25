"""Owner A focused probe: singleton rate, cardinality, and a non-Latin eyeball.

Memory-safe rewrite for a standard (non-High-RAM) Colab runtime, sized against
the real data: S1 2,206,821 / S2 5,034,616 / S3 5,285,603 rows.

What changed vs. the previous version, and why:
  1. Section 2 called read_source(path).head(N) -- read_csv loads the WHOLE file
     before .head() slices it, so S2+S3 cost ~3 GB before a single row was
     dropped. Now chunked via read_csv(chunksize=...), so only one chunk is live.
  2. Section 2 now filters each chunk to India BEFORE the script check, and caps
     each source at WINDOW_ROWS rows instead of scanning whole files.
  2b. The sample is a uniform reservoir over that window. Stopping at the first
     20 hits biased it to the head of the file, since a ~35-40% mixed rate fills
     the quota inside one chunk; collecting every hit instead would cost ~0.83 GB
     across both sources, so the reservoir holds exactly 20 rows either way.
  3. Every source read passes usecols, so unused columns never materialise.
  4. Section 1b builds entity_id->country from 200k-row chunks and deletes each
     chunk immediately, rather than holding a full S1 DataFrame.
  5. Both per-S1 dicts are deleted as soon as the aggregate is computed, so the
     three sections do not accumulate peak memory across the script.
  6. train_ground_truth.tsv is still read whole -- it is one small file and
     io_rules.md section 1's flags are mandatory for it.

Each section is independently wrapped, so one failure never loses the others.
"""

import csv
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- config ----
DATA_ROOT = "/content/drive/MyDrive/6ab10eb3b23ba_student_resource/student_resource/dataset"
TRAIN = Path(DATA_ROOT) / "train"

N_EYEBALL = 20          # rows in the final sample (reservoir size)
WINDOW_ROWS = 1_000_000 # rows scanned per source before sampling stops
SAMPLE_SEED = 42        # fixed, so the sample is reproducible
S1_CHUNK = 200_000      # rows per chunk for the S1 country join
FRAG_CHUNK = 50_000     # rows per chunk for the S2/S3 eyeball

# io_rules.md section 1: these flags are mandatory and must not be re-derived.
READ_FLAGS = dict(
    sep="\t", dtype=str, keep_default_na=False, na_filter=False,
    quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="warn",
)

RESULTS = {}
STATUS = {}


def read_source(path, **kw):
    """The io_rules.md section 1 contract, with usecols/chunksize passed through."""
    return pd.read_csv(path, **READ_FLAGS, **kw)


def script_flag(text):
    """Coarse script label (mirrors src/ber/normalise.py script_flag).

    Only alphabetic characters count, so punctuation, symbols and emoji can
    never produce 'mixed', and accented Latin stays 'latin'. A pure non-Latin
    string returns its own script name, so filtering on 'mixed' alone would
    miss exactly the rows that matter most.
    """
    scripts = set()
    for ch in text or "":
        if not ch.isalpha():
            continue
        try:
            scripts.add(unicodedata.name(ch).split()[0].lower())
        except ValueError:
            scripts.add("unknown")
    if not scripts:
        return "none"
    if scripts == {"latin"}:
        return "latin"
    if "latin" in scripts:
        return "mixed"
    return sorted(scripts)[0]


def show(title, obj):
    bar = "=" * 72
    print("\n" + bar + "\n" + title + "\n" + bar)
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def find_train_dir():
    if TRAIN.is_dir():
        return TRAIN
    base = Path("/content/drive/MyDrive")
    if base.is_dir():
        for hit in base.rglob("dataset/train"):
            return hit
    raise FileNotFoundError(f"no train dir at {TRAIN}")


print(__doc__.split("Each section")[0].strip())

# Resolved once, outside the numbered sections: if this fails, every section
# reports its own cause instead of dying on a NameError from section 1.
try:
    train_dir = find_train_dir()
    print("\ntrain dir:", train_dir)
    print("files:", sorted(p.name for p in train_dir.glob("*.tsv")))
except Exception as exc:
    train_dir = None
    print(f"could not locate train dir: {type(exc).__name__}: {exc}", file=sys.stderr)


# ================================== 1. singleton rate + cardinality =========
# Ground truth only. Nothing else is loaded here, so this can never OOM.
try:
    if train_dir is None:
        raise FileNotFoundError(f"train dir not found (looked at {TRAIN})")
    gt_path = next(train_dir.glob("*ground_truth*.tsv"))

    gt = read_source(gt_path)
    key_col, val_col = gt.columns[0], gt.columns[1]

    # Row count vs distinct keys settles the file's shape: io_rules.md does not
    # pin it, and a key count alone cannot tell the list shape from pairwise.
    n_rows = len(gt)
    n_keys = gt[key_col].nunique()

    counts = Counter()
    if n_rows == n_keys:
        shape = "one row per S1 (io_rules.md 5.1 list shape)"
        for raw in gt[val_col]:
            counts[len([v for v in str(raw).split(",") if v.strip()])] += 1
    else:
        shape = "pairwise (S1 repeated across rows)"
        per_s1 = Counter()
        for k, raw in zip(gt[key_col], gt[val_col]):
            key = k.strip()
            if str(raw).strip():
                per_s1[key] += 1
            else:
                per_s1.setdefault(key, 0)
        counts = Counter(per_s1.values())
        del per_s1

    n_s1 = sum(counts.values())
    buckets = {"t=0": 0, "t=1": 0, "t=2-4": 0, "t>=5": 0}
    for t, c in counts.items():
        if t == 0:
            buckets["t=0"] += c
        elif t == 1:
            buckets["t=1"] += c
        elif t <= 4:
            buckets["t=2-4"] += c
        else:
            buckets["t>=5"] += c

    RESULTS["1_cardinality"] = {
        "file": str(gt_path),
        "header": list(gt.columns),
        "n_data_rows": n_rows,
        "n_distinct_s1": n_keys,
        "detected_shape": shape,
        "n_s1": n_s1,
        "singleton_rate_overall": buckets["t=0"] / (n_s1 or 1),
        "cardinality_buckets": buckets,
        "cardinality_pct": {k: v / (n_s1 or 1) for k, v in buckets.items()},
        "histogram_t_0_to_10": {str(t): counts.get(t, 0) for t in range(11)},
        "max_t": max(counts) if counts else 0,
        "mean_t": sum(t * c for t, c in counts.items()) / (n_s1 or 1),
    }
    del gt
    STATUS["1_cardinality"] = "OK"
    show("1. SINGLETON RATE + CARDINALITY", RESULTS["1_cardinality"])
except Exception as exc:
    STATUS["1_cardinality"] = f"FAILED: {type(exc).__name__}: {exc}"
    print(f"\n[1] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)


# ========================= 1b. optional country breakdown (chunked S1) ======
try:
    if train_dir is None:
        raise FileNotFoundError(f"train dir not found (looked at {TRAIN})")
    if "1_cardinality" not in RESULTS:
        raise RuntimeError("section 1 did not complete; nothing to break down")

    # t per S1, honouring whichever shape section 1 detected. Counting raw rows
    # here would inflate n_s1 on a pairwise file (one row per match, not per S1).
    gt2 = read_source(gt_path, usecols=[0, 1])
    kc, vc = gt2.columns[0], gt2.columns[1]
    t_per_s1 = {}
    if RESULTS["1_cardinality"]["detected_shape"].startswith("one row"):
        for k, raw in zip(gt2[kc], gt2[vc]):
            t_per_s1[k.strip()] = len([v for v in str(raw).split(",") if v.strip()])
    else:
        for k, raw in zip(gt2[kc], gt2[vc]):
            key = k.strip()
            t_per_s1[key] = t_per_s1.get(key, 0) + (1 if str(raw).strip() else 0)
    del gt2

    # Aggregate per country as each S1 chunk arrives, so neither the S1 frame
    # nor an entity_id->country dict is ever fully materialised.
    per_country = {}
    s1_path = next(train_dir.glob("*source1*.tsv"))
    n_chunks = 0
    for chunk in read_source(s1_path, usecols=["entity_id", "country"],
                             chunksize=S1_CHUNK):
        n_chunks += 1
        for eid, raw_country in zip(chunk["entity_id"], chunk["country"]):
            t = t_per_s1.get(eid.strip())
            if t is None:
                continue
            label = str(raw_country).strip().casefold() or "<missing>"
            e = per_country.setdefault(
                label, {"n_s1": 0, "t=0": 0, "t=1": 0, "t=2-4": 0, "t>=5": 0}
            )
            e["n_s1"] += 1
            if t == 0:
                e["t=0"] += 1
            elif t == 1:
                e["t=1"] += 1
            elif t <= 4:
                e["t=2-4"] += 1
            else:
                e["t>=5"] += 1
        del chunk

    del t_per_s1
    for e in per_country.values():
        e["singleton_rate"] = e["t=0"] / (e["n_s1"] or 1)

    RESULTS["1b_by_country"] = {"s1_chunks_read": n_chunks, "by_country": per_country}
    STATUS["1b_by_country"] = "OK"
    show("1b. SINGLETON RATE BY COUNTRY (chunked join)", RESULTS["1b_by_country"])
except Exception as exc:
    STATUS["1b_by_country"] = f"FAILED: {type(exc).__name__}: {exc}"
    print(f"\n[1b] SKIPPED: {type(exc).__name__}: {exc}", file=sys.stderr)


# ============ 2. non-Latin eyeball (windowed scan, reservoir sample) ========
# Stopping at the first N matches biased the sample to the top of the file:
# at a ~35-40% mixed rate the quota filled inside the first chunk, so the sample
# only reflected whatever ordering the file happens to have (upload batch,
# region). Now each source is scanned over a fixed WINDOW_ROWS window and the
# sample is drawn uniformly across it.
#
# The sample is drawn by reservoir sampling rather than by collecting every hit
# and sampling at the end: collecting ~400k hits per source as dicts measured
# ~1043 B/row => ~0.83 GB across both sources, which is the kind of number that
# caused the original OOM. A reservoir holds exactly N_EYEBALL rows regardless
# of how many hits are seen, and is uniform over the window (verified).
NON_LATIN = {"mixed", "devanagari", "tamil", "bengali", "telugu", "gujarati",
             "kannada", "malayalam", "oriya", "gurmukhi", "arabic", "han",
             "hiragana", "katakana", "cyrillic", "unknown"}
try:
    if train_dir is None:
        raise FileNotFoundError(f"train dir not found (looked at {TRAIN})")

    rng = random.Random(SAMPLE_SEED)
    reservoir, n_seen = [], 0          # n_seen = hits seen, across both sources
    per_source, label_counts, flag_counts = {}, Counter(), Counter()

    for vendor in ("source2", "source3"):
        path = next(train_dir.glob(f"*{vendor}*.tsv"))
        reader = read_source(
            path,
            usecols=["entity_id", "business_name", "business_address", "country"],
            chunksize=FRAG_CHUNK,
        )
        scanned = india = hits = 0
        for chunk in reader:
            scanned += len(chunk)
            cl = chunk["country"].str.strip().str.casefold()
            label_counts.update(cl.value_counts().to_dict())

            # Filter to India BEFORE the per-string script check, to cut volume.
            ind = chunk[cl.str.startswith("ind")]
            india += len(ind)
            if len(ind):
                blob = (ind["business_name"].astype(str) + " "
                        + ind["business_address"].astype(str))
                flags = blob.map(script_flag)
                flag_counts.update(flags.value_counts().to_dict())

                sel = ind[flags.isin(NON_LATIN)]
                hits += len(sel)
                for eid, nm, ad, ctry, fl in zip(
                    sel["entity_id"], sel["business_name"],
                    sel["business_address"], sel["country"],
                    flags[flags.isin(NON_LATIN)],
                ):
                    n_seen += 1
                    row = {"source": vendor, "entity_id": eid, "country": ctry,
                           "script_flag": fl, "business_name": nm,
                           "business_address": ad}
                    if len(reservoir) < N_EYEBALL:
                        reservoir.append(row)
                    else:
                        j = rng.randrange(n_seen)
                        if j < N_EYEBALL:
                            reservoir[j] = row
                del blob, flags, sel
            del chunk, cl, ind

            if scanned >= WINDOW_ROWS:
                break
        del reader

        per_source[vendor] = {
            "rows_scanned": scanned,
            "india_rows": india,
            "n_non_latin": hits,
            "non_latin_rate_of_india": hits / (india or 1),
            "window_exhausted_file": scanned < WINDOW_ROWS,
        }
        print(f"  {vendor}: scanned {scanned:,} rows, {india:,} India, "
              f"{hits:,} non-Latin ({hits / (india or 1):.1%} of India)")

    RESULTS["2_script_eyeball"] = {
        "window_rows_per_source": WINDOW_ROWS,
        "sample_seed": SAMPLE_SEED,
        "per_source": per_source,
        "country_labels_seen": dict(label_counts.most_common(10)),
        "script_flag_counts_in_scan": dict(flag_counts),
        "n_non_latin_seen_total": n_seen,
        "sampling": ("uniform reservoir over the whole window, so the sample is "
                     "not biased to the head of the file"),
        "note": ("script_flag only inspects alphabetic chars, so emoji, '&', '#' "
                 "and accented Latin cannot produce 'mixed' -- a hit here is a "
                 "real non-Latin letter. Pure non-Latin returns its own script "
                 "name (e.g. 'devanagari'), not 'mixed'."),
        "sample": reservoir,
    }
    STATUS["2_script_eyeball"] = "OK"
    show("2. NON-LATIN SCRIPT EYEBALL", RESULTS["2_script_eyeball"])
except Exception as exc:
    STATUS["2_script_eyeball"] = f"FAILED: {type(exc).__name__}: {exc}"
    print(f"\n[2] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)


# ============================================================ 3. summary ====
summary = {"status": STATUS, "key_numbers": {}}
if "1_cardinality" in RESULTS:
    r = RESULTS["1_cardinality"]
    summary["key_numbers"].update({
        "ground_truth_shape": r["detected_shape"],
        "n_s1": r["n_s1"],
        "singleton_rate_overall": round(r["singleton_rate_overall"], 6),
        "buckets": r["cardinality_buckets"],
        "max_t": r["max_t"],
    })
if "1b_by_country" in RESULTS:
    summary["key_numbers"]["singleton_rate_by_country"] = {
        k: round(v["singleton_rate"], 6)
        for k, v in RESULTS["1b_by_country"]["by_country"].items()
    }
if "2_script_eyeball" in RESULTS:
    r = RESULTS["2_script_eyeball"]
    summary["key_numbers"]["non_latin"] = {
        "script_flag_counts": r["script_flag_counts_in_scan"],
        "n_seen_total": r["n_non_latin_seen_total"],
        "per_source_rate_of_india": {
            k: round(v["non_latin_rate_of_india"], 4)
            for k, v in r["per_source"].items()
        },
    }
show("3. SUMMARY", summary)
