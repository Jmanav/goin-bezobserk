"""Hand-written rule lists: legal suffixes, street abbreviations, landmarks.

These are domain knowledge, not an external lookup (research.md section 3.1 and
the section 9 compliance checklist). No entry here may come from a gazetteer,
registry or scraped list. Owner A, Sprint 0 tasks A3.1-A3.5.

Every list below is reproduced from research.md section 3.1. When adding an
entry, add it to the methodology-doc appendix too (A3.6).
"""

from __future__ import annotations

# --- A3.1 legal suffixes, per country, verbatim from research.md 3.1 ---------

LEGAL_SUFFIXES = {
    "US": [
        "incorporated",
        "inc",
        "llc",
        "l.l.c.",
        "corp",
        "co",
        "ltd",
        "llp",
        "pllc",
    ],
    "IN": [
        "private limited",
        "pvt ltd",
        "pvt. ltd.",
        "(p) ltd",
        "llp",
        "and sons",
        "enterprises",
        "traders",
    ],
    "FR": [
        "sarl",
        "sas",
        "sasu",
        "sa",
        "eurl",
        "sci",
        "snc",
        "societe anonyme",
        "ets",
        "etablissements",
    ],
}

# research.md 3.1 lists "Societe" among the French designators, but it is also
# the first word of ordinary trade names -- research.md's own example is
# "Societe Generale". Bare "societe" is therefore NOT a strippable suffix; only
# the full legal form "societe anonyme" is. Kept here so the token still reads
# as a French-country signal for the suffix_countries feature (A3.2).
FR_DESIGNATOR_TOKENS = {"societe", "etablissements", "ets"}

# "M/s" / "M/S" is an India-specific *prefix*, handled separately from suffixes.
LEGAL_PREFIXES = {"IN": ["m/s"]}

ALL_LEGAL_SUFFIXES = sorted(
    {s for group in LEGAL_SUFFIXES.values() for s in group},
    key=len,
    reverse=True,
)

_SUFFIX_COUNTRY = {}
for _country, _suffixes in LEGAL_SUFFIXES.items():
    for _s in _suffixes:
        _SUFFIX_COUNTRY.setdefault(_s, set()).add(_country)

SUFFIX_COUNTRIES = {k: frozenset(v) for k, v in _SUFFIX_COUNTRY.items()}


# --- A3.3 street abbreviations, verbatim from research.md 3.1 ----------------
# Bidirectional pairs are expanded to the long form so both spellings collapse
# to one token.

STREET_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "bd": "boulevard",
    "blvd": "boulevard",
}

# Tokens that are already canonical and must not be rewritten. research.md 3.1
# names Nagar and Marg explicitly as India address vocabulary.
CANONICAL_ADDRESS_TOKENS = {"nagar", "marg", "street", "road", "avenue", "boulevard"}


# --- A2.6 landmark phrases, verbatim from research.md 3.1 -------------------

LANDMARK_PHRASES = [
    "opposite",
    "opp",
    "behind",
    "beside",
    "near",
    "nr",
    "b/h",
    "next to",
    "en face de",
    "pres de",
]


# --- A3.5 transliteration variants, from research.md 3.1 and 5 --------------
# Algorithmic handling (char n-grams + Double Metaphone) does the heavy lifting;
# this map only collapses the variants research.md names explicitly.

TRANSLITERATION_VARIANTS = {
    "shree": "shri",
    "sree": "shri",
    "sri": "shri",
    "ent": "enterprises",
    "aggarwal": "agarwal",
}


# --- io_rules.md section 4 placeholders --------------------------------------

PLACEHOLDER_TOKENS = {"", "-", "0", "na", "none", "null", "n/a"}


# --- research.md 3.3 branch words (chain/franchise signal) ------------------

BRANCH_WORDS = {"north", "south", "east", "west", "airport", "mall", "branch", "store"}
