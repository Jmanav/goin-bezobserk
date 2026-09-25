"""Proxy data generator (Owner A tasks A5.1-A5.3).

roadmap.md Sprint 0: generate synthetic S1/S2/S3 by corrupting business lists
*you create yourself*. Every seed string below is invented for this repo. No
external registry, gazetteer or scraped list is involved, per io_rules.md
section 9.

The generator deliberately seeds the research.md section 5 edge cases so the
pipeline meets them before the real data does, and exposes the singleton rate
as a parameter because it is an unknown (A0.4, research.md Q5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .rules import LEGAL_SUFFIXES

# --- A5.1 self-authored seed vocabulary -------------------------------------

_NAME_HEADS = [
    "Brightleaf", "Copper Kettle", "Ironwood", "Silver Birch", "Morning Star",
    "Blue Lantern", "Red Panda", "Golden Arch", "Quiet Harbor", "Clever Fox",
    "Sunder", "Marigold", "Northwind", "Pebble Lane", "Tall Cedar",
    "Anand", "Vishwa", "Shree Ganesh", "Krishna", "Lotus Bay",
    # Accented heads keep the non-ASCII slice non-empty (research.md 5), so
    # A0.6's census and the fold-vs-accent path are actually exercised.
    "Petit Jardin", "Belle Rivière", "Côte Dorée", "Maison Verte", "Café Lumière",
]

_NAME_DESCRIPTORS = [
    "Pharmacy", "Restaurant", "Traders", "Store", "Bakery", "Hardware",
    "Textiles", "Motors", "Clinic", "Cafe", "Grocers", "Electronics",
]

_STREETS = [
    "Maple Street", "Cedar Road", "Oakline Avenue", "Harbour Boulevard",
    "Gandhi Marg", "Nehru Nagar", "Rue des Peupliers", "Avenue Clairval",
    "Willow Lane", "Sunset Ridge Road", "Rue Émile Zola", "Allée des Châtaigniers",
]

_CITIES = {
    "US": ["Fairhaven", "Northport", "Cedar Falls", "Lakeshore"],
    "India": ["Nashik", "Indore", "Coimbatore", "Vadodara"],
    "France": ["Saint-Aubin", "Valmont", "Beaulieu", "Montclair"],
}

_COUNTRY_WEIGHTS = {"US": 0.45, "India": 0.45, "France": 0.10}


@dataclass
class ProxyConfig:
    """Knobs for the proxy generator.

    `singleton_rate` is a parameter, not a constant: research.md section 1.2
    notes a 40% singleton share would swing up to 0.40 of the leaderboard
    score, and the true rate is unknown until the real audit (A0.4).
    """

    n_entities: int = 400
    singleton_rate: float = 0.40
    orphan_rate: float = 0.15
    seed: int = 20260925
    france_test_only: bool = True
    placeholder_rate: float = 0.04
    max_fragments_per_vendor: int = 2


# --- A5.3 corruption operators ----------------------------------------------


def _abbreviate_street(text):
    for long, short in (
        ("Street", "St"),
        ("Road", "Rd"),
        ("Avenue", "Ave"),
        ("Boulevard", "Blvd"),
    ):
        text = text.replace(long, short)
    return text


def _drop_token(text, rng):
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    tokens.pop(rng.randrange(len(tokens)))
    return " ".join(tokens)


def _typo(text, rng):
    if len(text) < 4:
        return text
    i = rng.randrange(len(text) - 1)
    chars = list(text)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _case_noise(text, rng):
    return text.upper() if rng.random() < 0.5 else text.lower()


def _strip_accents_noisily(text):
    table = str.maketrans("éèêëàâîïôöùûüç", "eeeeaaiioouuuc")
    return text.translate(table)


def _add_landmark(text, country, rng):
    if country == "India":
        phrase = rng.choice(["near", "opp", "behind", "b/h"])
        anchor = rng.choice(["bus stand", "railway station", "main market"])
        return f"{text}, {phrase} {anchor}"
    if country == "France":
        return f"{text}, {rng.choice(['en face de', 'pres de'])} la mairie"
    return f"{text}, near {rng.choice(['the mall', 'city park'])}"


_CORRUPTIONS = [
    lambda t, c, r: _abbreviate_street(t),
    lambda t, c, r: _abbreviate_street(t),
    lambda t, c, r: _drop_token(t, r),
    lambda t, c, r: _drop_token(t, r),
    lambda t, c, r: _typo(t, r),
    lambda t, c, r: _case_noise(t, r),
    # Weighted low on purpose: some fragments must keep their accents so the
    # accent-vs-fold path (A2.2) sees both spellings of the same entity.
    lambda t, c, r: _strip_accents_noisily(t),
]


def _pick_country(rng):
    roll = rng.random()
    cumulative = 0.0
    for country, weight in _COUNTRY_WEIGHTS.items():
        cumulative += weight
        if roll <= cumulative:
            return country
    return "US"


def _suffix_for(country, rng):
    key = {"US": "US", "India": "IN", "France": "FR"}[country]
    return rng.choice(LEGAL_SUFFIXES[key])


def _title(suffix):
    return " ".join(part.upper() if len(part) <= 3 else part.title() for part in suffix.split())


def generate(config=None):
    """Build proxy S1/S2/S3 records plus ground truth.

    Returns `(s1, s2, s3, truth)` where each of the first three is a list of
    dicts with the io_rules.md section 2 columns, and `truth` maps an S1
    entity_id to the set of fragment ids that belong to it.
    """
    cfg = config or ProxyConfig()
    rng = random.Random(cfg.seed)

    s1, s2, s3 = [], [], []
    truth = {}

    # A5.3 chain and shared-address seeding: a handful of name cores and
    # addresses are reused across distinct S1 entities on purpose.
    chain_cores = [
        f"{rng.choice(_NAME_HEADS)} {rng.choice(_NAME_DESCRIPTORS)}"
        for _ in range(max(3, cfg.n_entities // 40))
    ]
    shared_addresses = [
        f"{rng.randrange(10, 900)} {rng.choice(_STREETS)}"
        for _ in range(max(3, cfg.n_entities // 50))
    ]

    for idx in range(1, cfg.n_entities + 1):
        country = _pick_country(rng)
        s1_id = f"S1-{idx:05d}"

        roll = rng.random()
        if roll < 0.12 and chain_cores:
            # chains/franchises: same name_core, different address
            core = rng.choice(chain_cores)
        elif roll < 0.20:
            # generic, low-information name
            core = f"City {rng.choice(_NAME_DESCRIPTORS)}"
        elif roll < 0.26:
            # acronym-vs-expansion case
            core = rng.choice(["International Business Systems", "National Grain Exchange"])
        else:
            core = f"{rng.choice(_NAME_HEADS)} {rng.choice(_NAME_DESCRIPTORS)}"

        name = f"{core} {_title(_suffix_for(country, rng))}"

        if rng.random() < 0.10 and shared_addresses:
            street = rng.choice(shared_addresses)  # malls / office towers
        else:
            street = f"{rng.randrange(10, 900)} {rng.choice(_STREETS)}"

        city = rng.choice(_CITIES[country])
        postcode = {
            "US": lambda: f"{rng.randrange(10000, 99999)}",
            "India": lambda: f"{rng.randrange(100000, 999999)}",
            "France": lambda: f"{rng.randrange(10000, 99999)}",
        }[country]()

        unit = ""
        if rng.random() < 0.25:
            unit = rng.choice([f"Ste {rng.randrange(100, 400)}", f"Flat {rng.randrange(1, 40)}B", f"#{rng.randrange(1, 60)}"])

        address = ", ".join(part for part in [street, unit, city, postcode] if part)
        if rng.random() < 0.10:
            address = _add_landmark(address, country, rng)

        s1.append(
            {
                "entity_id": s1_id,
                "business_name": name,
                "business_address": address,
                "country": country,
            }
        )

        is_singleton = rng.random() < cfg.singleton_rate
        matched = set()

        # research.md 5 wants a t>=5 slice; a few entities get a wide fan-out so
        # the large-cluster case is never empty.
        is_large_cluster = not is_singleton and rng.random() < 0.06

        if not is_singleton:
            for vendor, bucket in (("S2", s2), ("S3", s3)):
                if is_large_cluster:
                    n_frags = rng.randint(3, cfg.max_fragments_per_vendor + 3)
                else:
                    n_frags = rng.randint(0, cfg.max_fragments_per_vendor)
                for _ in range(n_frags):
                    frag_name = name
                    frag_addr = address

                    for _ in range(rng.randint(1, 3)):
                        op = rng.choice(_CORRUPTIONS)
                        frag_name = op(frag_name, country, rng)
                    for _ in range(rng.randint(1, 2)):
                        op = rng.choice(_CORRUPTIONS)
                        frag_addr = op(frag_addr, country, rng)

                    # vendors append branch words (research.md 3.1)
                    if rng.random() < 0.15:
                        frag_name = f"{frag_name} {rng.choice(['Branch', 'Store', 'Outlet'])}"

                    # one-distinguishing-token case: "Store #12" vs "#14"
                    if rng.random() < 0.08:
                        frag_name = f"{frag_name} #{rng.randrange(2, 40)}"

                    # io_rules.md section 4 placeholders
                    if rng.random() < cfg.placeholder_rate:
                        frag_addr = rng.choice(["", "-", "0", "NA", "None", "null", "n/a"])
                    if rng.random() < cfg.placeholder_rate / 2:
                        frag_name = rng.choice(["NA", "None", "-"])

                    frag_id = f"{vendor}-{len(bucket) + 1:05d}"
                    bucket.append(
                        {
                            "entity_id": frag_id,
                            "business_name": frag_name,
                            "business_address": frag_addr,
                            "country": country,
                        }
                    )
                    matched.add(frag_id)

        truth[s1_id] = matched

    # A5.3 orphans: fragments matching no S1 at all. These set lambda_null
    # (research.md 3.5) and must never be forced into a match.
    n_orphans = int(len(s2) * cfg.orphan_rate)
    for _ in range(n_orphans):
        country = _pick_country(rng)
        vendor, bucket = rng.choice([("S2", s2), ("S3", s3)])
        name = f"{rng.choice(_NAME_HEADS)} {rng.choice(_NAME_DESCRIPTORS)}"
        address = (
            f"{rng.randrange(10, 900)} {rng.choice(_STREETS)}, "
            f"{rng.choice(_CITIES[country])}"
        )
        bucket.append(
            {
                "entity_id": f"{vendor}-{len(bucket) + 1:05d}",
                "business_name": name,
                "business_address": address,
                "country": country,
            }
        )

    return s1, s2, s3, truth


def split_train_test(s1, s2, s3, truth, config=None):
    """Split so France is test-only (A5.2).

    Mirrors io_rules.md section 3: training covers US and India, and the test
    set adds France, which is absent from training. That is what makes the
    Sprint 2 leave-one-country-out split meaningful.
    """
    cfg = config or ProxyConfig()
    if not cfg.france_test_only:
        return {"train": (s1, s2, s3, truth), "test": (s1, s2, s3, truth)}

    by_id = {r["entity_id"]: r for r in s1}
    frag_by_id = {r["entity_id"]: r for r in (*s2, *s3)}

    train_s1 = [r for r in s1 if r["country"] != "France"]
    test_s1 = list(s1)

    train_ids = {r["entity_id"] for r in train_s1}
    train_truth = {k: v for k, v in truth.items() if k in train_ids}

    train_frag_ids = {f for ids in train_truth.values() for f in ids}
    train_s2 = [r for r in s2 if r["entity_id"] in train_frag_ids or r["country"] != "France"]
    train_s3 = [r for r in s3 if r["entity_id"] in train_frag_ids or r["country"] != "France"]

    return {
        "train": (train_s1, train_s2, train_s3, train_truth),
        "test": (test_s1, list(s2), list(s3), truth),
    }


def write_tsv(records, path, columns=None):
    """Write records as TSV matching the io_rules.md section 2 schema.

    Written with the csv module and QUOTE_NONE so the file round-trips through
    `read_source` and passes the section 1 field-count check.
    """
    import csv as _csv
    from pathlib import Path

    cols = columns or ["entity_id", "business_name", "business_address", "country"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle, delimiter="\t", quoting=_csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        writer.writerow(cols)
        for record in records:
            writer.writerow([_scrub(record[c]) for c in cols])
    return path


def _scrub(value):
    """Strip characters that would break the TSV contract.

    A literal tab or newline inside a field would fail the io_rules.md section 1
    four-field check, so the generator never emits one.
    """
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_truth(truth, path):
    """Write ground truth in the io_rules.md section 5.1 output shape."""
    import csv as _csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = _csv.writer(handle, delimiter="\t", quoting=_csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in sorted(truth):
            writer.writerow([s1_id, ",".join(sorted(truth[s1_id]))])
    return path
