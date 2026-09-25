"""Pair feature library (Owner C, tasks C1-C5).

research.md section 3.3 lists ~80 features. This is the ~25 highest-value
subset, chosen for a 48-hour build: the string, rarity, number and listwise
groups that the section 4 novelty stack ranks #2 and #3, plus the categoricals.
Dropped for time: cross-vendor triangle support (C4.5), per-vendor noise stats
(C5.3), and everything downstream of the cross-encoder.

RapidFuzz (MIT, research.md section 9) is used when available, with pure-Python
fallbacks so the pipeline never hard-depends on an optional wheel. The fallbacks
are slower but produce the same feature names, so a model trained in one
environment still scores in the other.
"""

from __future__ import annotations

import math
import re
from collections import Counter

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import JaroWinkler as _JW
    HAVE_RAPIDFUZZ = True
except ImportError:          # pragma: no cover - environment dependent
    _fuzz = None
    _JW = None
    HAVE_RAPIDFUZZ = False

BRANCH_WORDS = frozenset({
    "north", "south", "east", "west", "airport", "mall", "branch", "annex",
    "central", "downtown", "uptown", "main", "new", "old",
})

_NUM = re.compile(r"\d+")


# --- C1: string metrics -----------------------------------------------------


def _ratio(a, b):
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return _fuzz.ratio(a, b) / 100.0
    return _seq_ratio(a, b)


def _token_set_ratio(a, b):
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return _fuzz.token_set_ratio(a, b) / 100.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _token_sort_ratio(a, b):
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return _fuzz.token_sort_ratio(a, b) / 100.0
    return _seq_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _partial_ratio(a, b):
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return _fuzz.partial_ratio(a, b) / 100.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return 0.0
    best = 0.0
    for i in range(max(1, len(long) - len(short) + 1)):
        best = max(best, _seq_ratio(short, long[i:i + len(short)]))
    return best


def _jaro_winkler(a, b):
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return _JW.similarity(a, b)
    return _seq_ratio(a, b)          # coarse stand-in; RapidFuzz is preferred


def _seq_ratio(a, b):
    """Character-bag similarity. Not Levenshtein, but monotone in overlap and
    cheap; only used when RapidFuzz is absent."""
    if a == b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    shared = sum((ca & cb).values())
    total = len(a) + len(b)
    return 2.0 * shared / total if total else 0.0


# --- C2: IDF / rarity -------------------------------------------------------


def _idf_features(tokens_a, tokens_b, idf, default_idf):
    sa, sb = set(tokens_a), set(tokens_b)
    shared = sa & sb
    union = sa | sb
    if not union:
        return {"idf_jaccard": 0.0, "idf_shared_sum": 0.0,
                "idf_unshared_sum": 0.0, "idf_rarest_shared": 0.0}

    def w(t):
        return idf.get(t, default_idf)

    shared_sum = sum(w(t) for t in shared)
    union_sum = sum(w(t) for t in union)
    return {
        "idf_jaccard": shared_sum / union_sum if union_sum else 0.0,
        "idf_shared_sum": shared_sum,
        "idf_unshared_sum": union_sum - shared_sum,
        # Fellegi-Sunter / Splink-style TF adjustment expressed as a feature
        # (research.md 2.4: use the idea as a feature, never as the matcher).
        "idf_rarest_shared": max((w(t) for t in shared), default=0.0),
    }


# --- C3: number / unit / distinguishing tokens ------------------------------


def _three_state(a, b):
    """0 missing, 1 equal, 2 different. research.md 5 and io_rules.md 4 both
    insist missing is distinct from different: a blank address is not evidence
    of a mismatch."""
    if not a or not b:
        return 0
    return 1 if a == b else 2


def _postcode_state(a, b):
    if not a or not b:
        return 0
    if a == b:
        return 1
    return 2 if a[:3] == b[:3] else 3      # 2 = shared prefix, 3 = different


def _acronym_match(name_a, name_b):
    """"IBM" vs "International Business Machines" (research.md 5)."""
    for src, dst in ((name_a, name_b), (name_b, name_a)):
        tokens = [t for t in dst.split() if t]
        if len(tokens) < 2:
            continue
        initials = "".join(t[0] for t in tokens)
        if initials and initials in src.split():
            return 1
    return 0


# --- feature assembly -------------------------------------------------------

FEATURE_NAMES = [
    # C1 string
    "name_ratio", "name_core_ratio", "name_token_set", "name_token_sort",
    "name_partial", "name_jaro", "addr_ratio", "addr_token_set",
    # C1.3 channel scores and ranks
    "rrf_score", "rrf_rank",
    # C2 rarity
    "idf_jaccard", "idf_shared_sum", "idf_unshared_sum", "idf_rarest_shared",
    # C3 numbers and distinguishing tokens
    "house_state", "unit_state", "postcode_state", "name_number_match",
    "suffix_state", "acronym", "branch_word_conflict",
    # C4 listwise
    "cand_count", "score_gap_to_best", "is_best_candidate",
    "s1_share_name_core", "s1_share_address",
    # C5 categoricals
    "vendor_is_s3", "country_known", "script_mixed",
]


def pair_features(frag, s1, *, idf, default_idf, rrf_score=0.0, rrf_rank=0,
                  cand_count=1, best_score=0.0, name_core_counts=None,
                  address_counts=None):
    """Features for one (fragment, S1) candidate pair.

    `frag` and `s1` are Owner A normalised records. Returns a dict keyed by
    FEATURE_NAMES so the caller can vectorise in a stable order.
    """
    name_core_counts = name_core_counts or {}
    address_counts = address_counts or {}

    f = {
        "name_ratio": _ratio(frag.name_norm, s1.name_norm),
        "name_core_ratio": _ratio(frag.name_core, s1.name_core),
        "name_token_set": _token_set_ratio(frag.name_core, s1.name_core),
        "name_token_sort": _token_sort_ratio(frag.name_core, s1.name_core),
        "name_partial": _partial_ratio(frag.name_core, s1.name_core),
        "name_jaro": _jaro_winkler(frag.name_core, s1.name_core),
        "addr_ratio": _ratio(frag.addr_norm, s1.addr_norm),
        "addr_token_set": _token_set_ratio(frag.addr_norm, s1.addr_norm),
        "rrf_score": float(rrf_score),
        "rrf_rank": float(rrf_rank),
    }
    f.update(_idf_features(frag.name_tokens, s1.name_tokens, idf, default_idf))

    fa = frag.addr_numbers or {}
    sa = s1.addr_numbers or {}
    f["house_state"] = _three_state(fa.get("house_number"), sa.get("house_number"))
    f["unit_state"] = _three_state(",".join(fa.get("units", [])),
                                   ",".join(sa.get("units", [])))
    f["postcode_state"] = _postcode_state(frag.postcode, s1.postcode)

    nums_a, nums_b = set(_NUM.findall(frag.name_norm)), set(_NUM.findall(s1.name_norm))
    f["name_number_match"] = _three_state(
        ",".join(sorted(nums_a)), ",".join(sorted(nums_b)))

    # Suffix conflict, not deletion (research.md 3.1 / A3.2): "X Pvt Ltd" and
    # "X LLP" share a name_core but are different companies.
    sfa, sfb = set(frag.legal_suffixes or ()), set(s1.legal_suffixes or ())
    if not sfa or not sfb:
        f["suffix_state"] = 0
    elif sfa & sfb:
        f["suffix_state"] = 1
    else:
        f["suffix_state"] = 2

    f["acronym"] = _acronym_match(frag.name_core, s1.name_core)

    ba = {t for t in frag.name_core.split() if t in BRANCH_WORDS}
    bb = {t for t in s1.name_core.split() if t in BRANCH_WORDS}
    f["branch_word_conflict"] = 1 if (ba ^ bb) else 0

    f["cand_count"] = float(cand_count)
    f["score_gap_to_best"] = float(best_score) - float(rrf_score)
    f["is_best_candidate"] = 1 if rrf_rank == 0 else 0
    f["s1_share_name_core"] = float(name_core_counts.get(s1.name_core, 1))
    f["s1_share_address"] = float(address_counts.get(s1.addr_norm, 1))

    f["vendor_is_s3"] = 0.0      # set by the caller from the id prefix
    # Country is a soft signal with an explicit unseen bucket; it must never be
    # the only split path (io_rules.md 3, research.md 3.3).
    f["country_known"] = 1 if frag.country_norm == s1.country_norm else 0
    f["script_mixed"] = 1 if "mixed" in (frag.script_flag, s1.script_flag) else 0
    return f


def features_to_row(feature_dict):
    return [float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES]


def corpus_counts(records):
    """Chain and shared-address counts over S1 (C4.3, C4.4)."""
    name_core = Counter()
    address = Counter()
    for rec in records.values():
        if rec.name_core:
            name_core[rec.name_core] += 1
        if rec.addr_norm:
            address[rec.addr_norm] += 1
    return dict(name_core), dict(address)


def default_idf_value(idf):
    """IDF for a token unseen in S1: treat it as maximally rare."""
    return max(idf.values(), default=1.0) if idf else 1.0


def token_idf(token_lists):
    df = Counter()
    n = 0
    for tokens in token_lists:
        n += 1
        df.update(set(tokens))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
