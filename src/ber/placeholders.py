"""Placeholder detection (io_rules.md section 4, Owner A tasks A4.1-A4.3).

The hard rule this module exists to enforce: two records must never match
*because* they share a placeholder token. Callers get a `missing` flag; it is
the caller's job to suppress key generation and similarity credit when it is
set (see B4.3 and Owner C).
"""

from __future__ import annotations

import unicodedata

from .rules import PLACEHOLDER_TOKENS


def _squash(value):
    text = unicodedata.normalize("NFKC", value or "").strip().casefold()
    return " ".join(text.split())


def is_placeholder(value, extra=None):
    """True when a field carries no usable signal (A4.1).

    Matches the io_rules.md section 4 set, plus any data-driven additions
    discovered by `discover_placeholders` (A4.2).
    """
    squashed = _squash(value)
    if squashed in PLACEHOLDER_TOKENS:
        return True
    if extra and squashed in extra:
        return True
    # Punctuation-only or zero-only junk: "--", "...", "000".
    if squashed and all(not ch.isalnum() for ch in squashed):
        return True
    if squashed and set(squashed) <= {"0"}:
        return True
    return False


def discover_placeholders(values, max_len=3, min_df_share=0.01):
    """Find likely placeholders from the data itself (A4.2).

    research.md section 5 gives the test as: count strings with DF > 1% of
    length <= 3. Returns the offending normalised strings so they can be folded
    into `is_placeholder` via `extra`.
    """
    total = 0
    counts = {}
    for value in values:
        squashed = _squash(value)
        total += 1
        if squashed and len(squashed) <= max_len:
            counts[squashed] = counts.get(squashed, 0) + 1

    if not total:
        return {}

    threshold = total * min_df_share
    return {
        text: count / total
        for text, count in counts.items()
        if count > threshold
    }


def missing_flags(df, name_col="business_name", addr_col="business_address", extra=None):
    """Return per-row missing flags for name and address (A4.1).

    io_rules.md section 4: an empty address is not a match signal, and an empty
    name means leaning on the address only, conservatively.
    """
    name_missing = df[name_col].map(lambda v: is_placeholder(v, extra))
    addr_missing = df[addr_col].map(lambda v: is_placeholder(v, extra))
    return name_missing, addr_missing
