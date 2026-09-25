"""Data audit reports and token IDF.

Covers Owner A tasks A1.5 (country audit), A2.3 (S1 token IDF) and the
data-derived half of the A0 [VERIFY] probes. Reports only -- nothing here
filters or mutates data, per the docs/data-audit skill contract.
"""

from __future__ import annotations

import math
from collections import Counter

from .normalise import normalise_country, has_non_ascii, script_flag
from .placeholders import discover_placeholders


# --- A2.3 token IDF over S1 names -------------------------------------------


def token_document_frequency(token_lists):
    """Document frequency per token."""
    df = Counter()
    for tokens in token_lists:
        df.update(set(tokens))
    return df


def token_idf(token_lists, smooth=True):
    """IDF per token over S1 names (A2.3).

    Consumed by Owner B's rare-name-token key (B4.1) and Owner C's IDF features
    (C2). Computed on S1 only, per research.md section 3.1.
    """
    token_lists = list(token_lists)
    n_docs = len(token_lists)
    df = token_document_frequency(token_lists)
    if not n_docs:
        return {}, df
    idf = {}
    for token, count in df.items():
        if smooth:
            idf[token] = math.log((n_docs + 1) / (count + 1)) + 1.0
        else:
            idf[token] = math.log(n_docs / count)
    return idf, df


def high_df_tokens(df, n_docs, percentile=0.99):
    """Tokens above a DF percentile (A2.4).

    research.md section 3.1 down-weights these by IDF rather than deleting
    them, so this returns a report, not a stoplist to subtract.
    """
    if not df or not n_docs:
        return []
    counts = sorted(df.values())
    cutoff_index = min(len(counts) - 1, int(len(counts) * percentile))
    cutoff = counts[cutoff_index]
    return sorted(
        ((t, c, c / n_docs) for t, c in df.items() if c >= cutoff),
        key=lambda row: -row[1],
    )


def differing_tokens_in_positives(pairs):
    """Tokens that often differ between matched pairs (A2.4).

    `pairs` yields (s1_tokens, fragment_tokens). research.md section 3.1 learns
    these from training positives -- vendors appending "Store"/"Branch" is the
    canonical case.
    """
    added = Counter()
    dropped = Counter()
    for s1_tokens, frag_tokens in pairs:
        a, b = set(s1_tokens), set(frag_tokens)
        added.update(b - a)
        dropped.update(a - b)
    return added, dropped


# --- A1.5 country audit ------------------------------------------------------


def country_report(frames):
    """Report country labels per source (A1.5, research.md audit #7).

    Reports exact distinct string values so the A0.5 [VERIFY] ("France" vs "FR"
    vs "france") can be settled from data. Never filters.
    """
    report = {}
    for source, df in frames.items():
        raw_counts = Counter(df["country"])
        norm_counts = Counter(normalise_country(c) for c in df["country"])
        report[source] = {
            "n_rows": len(df),
            "raw_values": dict(raw_counts),
            "normalised_values": dict(norm_counts),
            "n_distinct_raw": len(raw_counts),
            "n_distinct_normalised": len(norm_counts),
        }
    return report


def cross_country_matches(s1_df, frag_df, truth):
    """Count matched pairs whose countries disagree (research.md audit #7, Q6).

    Expected 0. Owner B's country sharding (B0.1) is gated on this being 0.
    """
    s1_country = dict(zip(s1_df["entity_id"], s1_df["country"].map(normalise_country)))
    frag_country = dict(zip(frag_df["entity_id"], frag_df["country"].map(normalise_country)))

    mismatches = []
    for s1_id, frag_ids in truth.items():
        a = s1_country.get(s1_id)
        for frag_id in frag_ids:
            b = frag_country.get(frag_id)
            if b is not None and a is not None and a != b:
                mismatches.append((s1_id, frag_id, a, b))
    return mismatches


def unseen_test_labels(train_frames, test_frames):
    """Country labels present in test but not train (io_rules.md section 3)."""
    train_labels = set()
    for df in train_frames.values():
        train_labels |= {normalise_country(c) for c in df["country"]}
    test_labels = set()
    for df in test_frames.values():
        test_labels |= {normalise_country(c) for c in df["country"]}
    return test_labels - train_labels


# --- A0.2 / A0.3 structural probes ------------------------------------------


def fragments_in_multiple_s1(truth):
    """Fragments claimed by more than one S1 (A0.2, research.md Q3, audit #3).

    Expected empty: the whole asymmetric framing in research.md section 1.1
    rests on a fragment belonging to at most one S1 entity.
    """
    owners = {}
    for s1_id, frag_ids in truth.items():
        for frag_id in frag_ids:
            owners.setdefault(frag_id, []).append(s1_id)
    return {f: ids for f, ids in owners.items() if len(ids) > 1}


def per_vendor_match_counts(truth):
    """Matched fragment counts per S1, split by vendor (A0.3, Q4, audit #3).

    If the max per vendor is 1, the within-vendor one-to-one constraint holds
    and Owner D may enable N2's bipartite branch (D5.3). Default is unknown, so
    that branch stays off until this says otherwise.
    """
    per_vendor = {}
    for s1_id, frag_ids in truth.items():
        counts = Counter(f.split("-", 1)[0] for f in frag_ids)
        per_vendor[s1_id] = dict(counts)

    maxima = {}
    for counts in per_vendor.values():
        for vendor, n in counts.items():
            maxima[vendor] = max(maxima.get(vendor, 0), n)
    return per_vendor, maxima


def cardinality_report(truth):
    """Histogram of t per S1 and the singleton share (audit #4, #5).

    research.md section 1.2: the singleton rate sets the stakes of the whole
    metric, since a true singleton predicted empty is a free 1.0.
    """
    sizes = Counter(len(ids) for ids in truth.values())
    n = len(truth) or 1
    n_singletons = sizes.get(0, 0)
    buckets = {"t=0": 0, "t=1": 0, "t=2-4": 0, "t>=5": 0}
    for size, count in sizes.items():
        if size == 0:
            buckets["t=0"] += count
        elif size == 1:
            buckets["t=1"] += count
        elif size <= 4:
            buckets["t=2-4"] += count
        else:
            buckets["t>=5"] += count
    return {
        "histogram": dict(sorted(sizes.items())),
        "buckets": buckets,
        "singleton_rate": n_singletons / n,
        "n_s1": len(truth),
    }


def orphan_report(truth, frag_frames):
    """Share of fragments matched to no S1, per vendor (audit #6).

    This sets lambda_null in Owner D's N2 (research.md section 3.4).
    """
    matched = {f for ids in truth.values() for f in ids}
    out = {}
    for vendor, df in frag_frames.items():
        ids = set(df["entity_id"])
        orphans = ids - matched
        out[vendor] = {
            "n_fragments": len(ids),
            "n_orphans": len(orphans),
            "orphan_rate": len(orphans) / (len(ids) or 1),
        }
    return out


# --- A0.6 script census, A4.2 placeholder discovery -------------------------


def script_census(frames, sample=None):
    """Non-Latin character census per country (A0.6, research.md section 3.1).

    Rule-based transliteration is only added if a non-Latin script actually
    appears and the library ships no external dictionaries. Vectorised, since
    this runs over every row of a multi-million-row source; pass `sample` to cap
    rows per source (reported as `n_sampled`) when a full pass is too slow.
    """
    out = {}
    for source, df in frames.items():
        n_total = len(df)
        if sample is not None and n_total > sample:
            df = df.sample(n=sample, random_state=0)
        blob = df["business_name"].astype(str) + " " + df["business_address"].astype(str)
        key = df["country"].map(normalise_country)
        flags = blob.map(script_flag)
        non_ascii = blob.map(has_non_ascii)

        per_country = {}
        for country, idx in key.groupby(key).groups.items():
            per_country[country] = {
                "n": int(len(idx)),
                "n_non_ascii": int(non_ascii.loc[idx].sum()),
                "scripts": {k: int(v) for k, v in flags.loc[idx].value_counts().items()},
            }
        if sample is not None and n_total > sample:
            for entry in per_country.values():
                entry["n_sampled_from"] = n_total
        out[source] = per_country
    return out


def placeholder_report(frames, max_len=3, min_df_share=0.01):
    """Data-driven placeholder discovery per source and column (A4.2)."""
    out = {}
    for source, df in frames.items():
        out[source] = {
            "business_name": discover_placeholders(
                df["business_name"], max_len=max_len, min_df_share=min_df_share
            ),
            "business_address": discover_placeholders(
                df["business_address"], max_len=max_len, min_df_share=min_df_share
            ),
        }
    return out


def literal_na_counts(frames):
    """Count literal "NA"/"None"/"NaN"/"null"/"" per column (audit #1).

    These survive only because `read_source` turns NA handling off; seeing them
    here is proof the io_rules.md section 1 flags worked.
    """
    targets = {"", "na", "none", "nan", "null", "n/a", "-", "0"}
    out = {}
    for source, df in frames.items():
        per_col = {}
        for col in ("business_name", "business_address", "country"):
            counts = Counter(
                v.strip().casefold() for v in df[col] if v.strip().casefold() in targets
            )
            per_col[col] = dict(counts)
        out[source] = per_col
    return out


# --- audit #10 scale --------------------------------------------------------


def scale_report(frames):
    """Rows per source, to finalise the research.md section 6 estimates."""
    return {source: len(df) for source, df in frames.items()}


def cardinality_by_country(truth, s1_df):
    """Singleton rate and t-buckets per country (audit #4, #5 per country).

    research.md section 1.2 makes the singleton rate the single biggest lever on
    the score, and section 1.4 warns that France behaves differently -- so the
    overall rate is not enough on its own.
    """
    country_of = dict(zip(s1_df["entity_id"], s1_df["country"]))
    per_country = {}
    for s1_id, frag_ids in truth.items():
        key = normalise_country(country_of.get(s1_id, ""))
        entry = per_country.setdefault(
            key, {"n_s1": 0, "t=0": 0, "t=1": 0, "t=2-4": 0, "t>=5": 0}
        )
        entry["n_s1"] += 1
        t = len(frag_ids)
        if t == 0:
            entry["t=0"] += 1
        elif t == 1:
            entry["t=1"] += 1
        elif t <= 4:
            entry["t=2-4"] += 1
        else:
            entry["t>=5"] += 1

    for entry in per_country.values():
        entry["singleton_rate"] = entry["t=0"] / (entry["n_s1"] or 1)
    return per_country


def ground_truth_shape(path):
    """Report the on-disk shape of a ground-truth TSV without assuming one.

    io_rules.md does not pin this file's format, so the row count is reported
    alongside the distinct-key count: equal means one row per S1 (the section
    5.1 list shape), more rows than keys means the pairwise shape.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r")
        n_data_lines = sum(1 for _ in handle)
    return {"path": str(path), "header": header.split("\t"), "n_data_rows": n_data_lines}
