"""Normalisation and rule-based parsing.

Implements the research.md section 3.1 output contract (Owner A task A2.1),
which Owner B blocks on and Owner C builds features from:

    name_norm, name_core, name_tokens, addr_norm, addr_numbers, postcode,
    landmark_flag, script_flag, plus a folded ASCII variant

Address handling is regex-only. No gazetteer-backed parser is wired in here:
research.md section 3.1 and io_rules.md section 9 treat libpostal-style model
data as external data by default, pending the A0.1 ruling.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .rules import (
    ALL_LEGAL_SUFFIXES,
    CANONICAL_ADDRESS_TOKENS,
    LANDMARK_PHRASES,
    LEGAL_PREFIXES,
    STREET_ABBREVIATIONS,
    SUFFIX_COUNTRIES,
    TRANSLITERATION_VARIANTS,
)

# --- A2.2 unicode layer ------------------------------------------------------

_PUNCT_MAP = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
}

_AMPERSAND = re.compile(r"\s*&\s*")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z\s'/#-]", re.IGNORECASE)


def _map_punct(text):
    return "".join(_PUNCT_MAP.get(ch, ch) for ch in text)


def nfkc_casefold(text):
    """NFKC then casefold (research.md 3.1)."""
    return unicodedata.normalize("NFKC", text or "").casefold()


def fold_ascii(text):
    """Accent-folded copy, kept *alongside* the accented form.

    research.md 3.1 keeps both so "Societe Generale" and "Societe Generale"
    reconcile without losing the original French spelling.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def has_non_ascii(text):
    return any(ord(ch) > 127 for ch in (text or ""))


def script_flag(text):
    """Coarse script label for the A2.1 contract.

    Used by the A0.6 non-Latin census and as a slice key in research.md 8.2.
    """
    scripts = set()
    for ch in text or "":
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            scripts.add("unknown")
            continue
        scripts.add(name.split()[0].lower())
    if not scripts:
        return "none"
    if scripts == {"latin"}:
        return "latin"
    if "latin" in scripts:
        return "mixed"
    return sorted(scripts)[0]


# --- name normalisation ------------------------------------------------------


def normalise_name(raw):
    """Produce `name_norm` (A2.2): unicode-normalised, punctuation-regularised.

    Accents are preserved here; `name_core_ascii` carries the folded variant.
    The alnum filter runs over the folded text so an accented letter is never
    mistaken for punctuation and blanked out -- io_rules.md section 3 calls
    France the place pipelines break, and "Societe Generale" losing its vowels
    is exactly that failure.
    """
    text = nfkc_casefold(raw)
    text = _map_punct(text)
    text = _AMPERSAND.sub(" and ", text)
    text = text.replace(".", " ").replace(",", " ")
    text = _strip_non_alnum_accent_safe(text)
    return _WS.sub(" ", text).strip()


# Unicode categories for combining marks: Mn (nonspacing, e.g. virama),
# Mc (spacing combining, e.g. Indic vowel signs). Both are part of the
# letter they attach to and must survive punctuation stripping.
_COMBINING_CATEGORIES = frozenset({"Mn", "Mc"})


def _strip_non_alnum_accent_safe(text):
    """Drop punctuation while keeping letters (accented or Indic) and digits.

    Combining marks are kept explicitly. Indic vowel signs and the virama are
    categories Mn/Mc, for which str.isalnum() is False, so testing isalnum alone
    replaced every one of them with a space and shattered Indic words into
    single letters: a Malayalam name reduced to its first character. That is
    ~40% of India fragments, so the category test is load-bearing, not a nicety.
    """
    out = []
    for ch in text:
        if (
            ch.isalnum()
            or ch.isspace()
            or ch in "'/#-"
            or unicodedata.category(ch) in _COMBINING_CATEGORIES
        ):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def _strip_legal_prefix(text):
    for prefixes in LEGAL_PREFIXES.values():
        for prefix in prefixes:
            if text.startswith(prefix + " "):
                return text[len(prefix) + 1 :].strip(), prefix
    return text, None


def find_legal_suffixes(name_norm):
    """Return the legal suffixes present in a normalised name (A3.1, A3.2).

    Returned as a set so Owner C can compute agreement/conflict rather than
    relying on deletion (research.md 3.1: store suffixes as a *feature*).
    """
    found = set()
    padded = f" {name_norm} "
    for suffix in ALL_LEGAL_SUFFIXES:
        needle = f" {suffix} ".replace(".", " ")
        needle = _WS.sub(" ", needle)
        if needle in padded:
            found.add(suffix)
    return found


def suffix_countries(suffixes):
    """Which countries a name's suffixes point at (A3.1)."""

    countries = set()
    for suffix in suffixes:
        countries |= SUFFIX_COUNTRIES.get(suffix, frozenset())
    return countries


def strip_legal_suffixes(name_norm, suffixes=None):
    """Remove legal suffixes to build `name_core` (A2.1).

    The suffixes themselves are retained by the caller as a feature (A3.2);
    stripping here only serves matching on the distinctive part of the name.

    Longest suffix first, so "pvt ltd" is removed whole rather than losing
    "ltd" and leaving a stray "pvt" behind. Never strips the entire name: a
    business called only "Enterprises" keeps it, since an empty name_core
    carries no matching signal at all.
    """
    text, _ = _strip_legal_prefix(name_norm)

    candidates = ALL_LEGAL_SUFFIXES if suffixes is None else suffixes
    for suffix in sorted(candidates, key=lambda s: (-len(s.split()), -len(s))):
        stripped = re.sub(
            rf"(?:^|\s){re.escape(suffix)}(?=\s|$)", " ", text
        )
        stripped = _WS.sub(" ", stripped).strip()
        if stripped:
            text = stripped
    return _WS.sub(" ", text).strip()


def apply_transliteration_variants(tokens):
    """Collapse the spelling variants research.md 3.1 names (A3.5)."""
    return [TRANSLITERATION_VARIANTS.get(t, t) for t in tokens]


def tokenise(text):
    return [t for t in text.split() if t]


# --- A2.5 address parsing (regex only) --------------------------------------

# US ZIP: 5 digits or ZIP+4. India PIN: 6 digits. France: 5 digits, optional
# CEDEX. All three are digit-shape rules, not gazetteer lookups.
_ZIP_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_PIN_IN = re.compile(r"\b(\d{6})\b")
_CP_FR = re.compile(r"\b(\d{5})\b(?:\s*cedex\b)?", re.IGNORECASE)
_CEDEX = re.compile(r"\bcedex\b", re.IGNORECASE)

_UNIT = re.compile(
    r"\b(?:ste|suite|apt|apartment|unit|flat|shop|office|room|fl|floor|no)\b\.?\s*"
    r"([0-9]+[a-z]?|[a-z]-?[0-9]+)",
    re.IGNORECASE,
)
_HASH_UNIT = re.compile(r"#\s*([0-9]+[a-z]?)")
_HOUSE_NUMBER = re.compile(r"\b(\d+[a-z]?(?:/\d+[a-z]?)?)\b")


def normalise_address(raw):
    """Produce `addr_norm` (A2.2, A3.3): normalised with abbreviations expanded."""
    text = nfkc_casefold(raw)
    text = _map_punct(text)
    text = _AMPERSAND.sub(" and ", text)
    text = text.replace(",", " ").replace(".", " ")
    text = _strip_non_alnum_accent_safe(text)
    text = _WS.sub(" ", text).strip()

    tokens = []
    for token in text.split():
        if token in CANONICAL_ADDRESS_TOKENS:
            tokens.append(token)
        else:
            tokens.append(STREET_ABBREVIATIONS.get(token, token))
    return " ".join(tokens)


def extract_postcode(addr_norm, country=None):
    """Extract a postcode by digit shape (A2.5).

    `country` only picks which shape to try first; an unknown or unseen label
    falls through to every shape, because io_rules.md section 3 forbids a code
    path that only handles a closed country set.
    """
    label = (country or "").strip().casefold()

    def _india():
        m = _PIN_IN.search(addr_norm)
        return m.group(1) if m else None

    def _us():
        m = _ZIP_US.search(addr_norm)
        return m.group(1) if m else None

    def _france():
        m = _CP_FR.search(addr_norm)
        return m.group(1) if m else None

    if label in {"in", "ind", "india"}:
        order = (_india, _us, _france)
    elif label in {"us", "usa", "united states", "united states of america"}:
        order = (_us, _india, _france)
    elif label in {"fr", "fra", "france"}:
        order = (_france, _us, _india)
    else:
        # Unseen or missing label: try the most specific shape first.
        order = (_india, _us, _france)

    for finder in order:
        value = finder()
        if value:
            return value
    return None


def extract_numbers(addr_norm):
    """Extract house number, unit and floor (A2.5).

    Returns a dict so Owner C can compare parsed fields exactly (C3.2, C3.3)
    rather than relying on fuzzy ratios over digits.
    """
    units = [m.group(1) for m in _UNIT.finditer(addr_norm)]
    units += [m.group(1) for m in _HASH_UNIT.finditer(addr_norm)]

    house = None
    for m in _HOUSE_NUMBER.finditer(addr_norm):
        candidate = m.group(1)
        if candidate in units:
            continue
        house = candidate
        break

    all_numbers = _HOUSE_NUMBER.findall(addr_norm)
    return {
        "house_number": house,
        "units": units,
        "all_numbers": all_numbers,
        "has_cedex": bool(_CEDEX.search(addr_norm)),
    }


def landmark_flag(addr_norm):
    """Tag landmark-reference addresses (A2.6)."""
    padded = f" {addr_norm} "
    for phrase in LANDMARK_PHRASES:
        if f" {phrase} " in padded:
            return True
    return False


# --- A2.7 phonetic key -------------------------------------------------------


def double_metaphone(text):
    """Phonetic key for `name_core` (A2.7).

    Algorithmic, with no bundled dictionary, so it stays clear of the
    io_rules.md section 9 external-data rule. Uses the `metaphone` package when
    available and falls back to a conservative built-in reduction otherwise, so
    the pipeline never hard-depends on an optional wheel in Colab.
    """
    try:
        from metaphone import doublemetaphone
    except ImportError:
        return _fallback_phonetic(text)
    primary, secondary = doublemetaphone(text or "")
    return primary or secondary or _fallback_phonetic(text)


_VOWELS = set("aeiou")


def _fallback_phonetic(text):
    """Soundex-flavoured reduction used when `metaphone` is absent."""
    folded = fold_ascii(nfkc_casefold(text or ""))
    letters = [ch for ch in folded if ch.isalpha()]
    if not letters:
        return ""

    groups = {
        **{c: "1" for c in "bfpv"},
        **{c: "2" for c in "cgjkqsxz"},
        **{c: "3" for c in "dt"},
        "l": "4",
        **{c: "5" for c in "mn"},
        "r": "6",
    }

    head = letters[0].upper()
    codes = []
    previous = groups.get(letters[0])
    for ch in letters[1:]:
        code = groups.get(ch)
        if code and code != previous:
            codes.append(code)
        if ch not in _VOWELS and ch not in {"h", "w"}:
            previous = code
        elif ch in _VOWELS:
            previous = None
    return (head + "".join(codes)).ljust(4, "0")[:6]


# --- the A2.1 contract -------------------------------------------------------

NORMALISED_FIELDS = [
    "name_norm",
    "name_core",
    "name_tokens",
    "name_core_ascii",
    "addr_norm",
    "addr_ascii",
    "addr_numbers",
    "postcode",
    "landmark_flag",
    "script_flag",
    "legal_suffixes",
    "suffix_countries",
    "phonetic_key",
    "country_norm",
    "name_missing",
    "addr_missing",
]


@dataclass
class Normalised:
    """One record's normalised form -- the A2.1 contract B and C code against."""

    name_norm: str = ""
    name_core: str = ""
    name_tokens: list = field(default_factory=list)
    name_core_ascii: str = ""
    addr_norm: str = ""
    addr_ascii: str = ""
    addr_numbers: dict = field(default_factory=dict)
    postcode: str | None = None
    landmark_flag: bool = False
    script_flag: str = "none"
    legal_suffixes: frozenset = frozenset()
    suffix_countries: frozenset = frozenset()
    phonetic_key: str = ""
    country_norm: str = ""
    name_missing: bool = False
    addr_missing: bool = False

    def as_dict(self):
        return {name: getattr(self, name) for name in NORMALISED_FIELDS}


def normalise_country(raw):
    """Casefold the country label but keep it (io_rules.md section 3).

    Never maps to a closed set and never drops an unseen label. Owner C buckets
    unseen values as "other/unseen" at feature time (C5.2); that decision does
    not belong here, because the raw label must survive to the submission.
    """
    return _WS.sub(" ", nfkc_casefold(raw)).strip()


def normalise_record(name, address, country="", extra_placeholders=None):
    """Normalise one record into the A2.1 contract.

    Pure and deterministic, per the io_rules.md section 9 reproducibility rule.
    """
    from .placeholders import is_placeholder

    name_missing = is_placeholder(name, extra_placeholders)
    addr_missing = is_placeholder(address, extra_placeholders)

    name_norm = "" if name_missing else normalise_name(name)
    addr_norm = "" if addr_missing else normalise_address(address)

    suffixes = find_legal_suffixes(name_norm)
    name_core = strip_legal_suffixes(name_norm, suffixes)
    tokens = apply_transliteration_variants(tokenise(name_core))

    numbers = extract_numbers(addr_norm) if addr_norm else {
        "house_number": None,
        "units": [],
        "all_numbers": [],
        "has_cedex": False,
    }

    return Normalised(
        name_norm=name_norm,
        name_core=name_core,
        name_tokens=tokens,
        name_core_ascii=fold_ascii(name_core),
        addr_norm=addr_norm,
        addr_ascii=fold_ascii(addr_norm),
        addr_numbers=numbers,
        postcode=extract_postcode(addr_norm, country) if addr_norm else None,
        landmark_flag=landmark_flag(addr_norm) if addr_norm else False,
        script_flag=script_flag(f"{name} {address}"),
        legal_suffixes=frozenset(suffixes),
        suffix_countries=frozenset(suffix_countries(suffixes)),
        phonetic_key=double_metaphone(name_core) if name_core else "",
        country_norm=normalise_country(country),
        name_missing=name_missing,
        addr_missing=addr_missing,
    )


def normalise_frame(df, extra_placeholders=None):
    """Normalise a whole source frame, returning the A2.1 columns.

    The input frame keeps `entity_id` untouched: io_rules.md section 2 forbids
    stripping the source prefix.
    """
    records = [
        normalise_record(
            name=row.business_name,
            address=row.business_address,
            country=row.country,
            extra_placeholders=extra_placeholders,
        ).as_dict()
        for row in df.itertuples(index=False)
    ]

    import pandas as pd

    out = pd.DataFrame.from_records(records, index=df.index)
    return df.join(out)
