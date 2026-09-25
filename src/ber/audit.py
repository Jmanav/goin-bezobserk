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


def script_census(frames):
    """Non-Latin character census per country (A0.6, research.md section 3.1).

    Rule-based transliteration is only added if a non-Latin script actually
    appears and the library ships no external dictionaries.
    """
    out = {}
    for source, df in frames.items():
        per_country = {}
        for row in df.itertuples(index=False):
            key = normalise_country(row.country)
            entry = per_country.setdefault(
                key, {"n": 0, "n_non_ascii": 0, "scripts": Counter()}
            )
            entry["n"] += 1
            blob = f"{row.business_name} {row.business_address}"
            if has_non_ascii(blob):
                entry["n_non_ascii"] += 1
            entry["scripts"][script_flag(blob)] += 1
        out[source] = {
            k: {**v, "scripts": dict(v["scripts"])} for k, v in per_country.items()
        }
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
