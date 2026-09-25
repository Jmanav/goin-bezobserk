"""Owner A Sprint 0 tests.

The io_rules.md section 8 "silent killers" are the spec: each one gets a test
that proves the guard fires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import audit, normalise, placeholders, proxy_data, safe_io  # noqa: E402


# --- A1 safe read ------------------------------------------------------------


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_na_like_names_survive_the_read(tmp_path):
    """io_rules.md section 1: a business literally named "NA" must not vanish."""
    path = _write(
        tmp_path,
        "s1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tNA\t12 Maple Street\tUS\n"
        "S1-00002\tNone\t\tIndia\n"
        "S1-00003\tnull\t9 Cedar Road\tFrance\n",
    )
    df = safe_io.read_source(path)
    assert list(df["business_name"]) == ["NA", "None", "null"]
    assert df["business_address"].iloc[1] == ""
    assert df.isna().sum().sum() == 0


def test_ids_stay_strings_with_prefixes(tmp_path):
    path = _write(
        tmp_path,
        "s2.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00047\tBrightleaf Pharmacy\t1 Oakline Avenue\tUS\n",
    )
    df = safe_io.read_source(path)
    assert df["entity_id"].iloc[0] == "S2-00047"
    # dtype=str yields object on pandas 2 and StringDtype on pandas 3; what
    # matters is that the id stayed textual and kept its prefix and any zeros.
    assert not str(df["entity_id"].dtype).startswith(("int", "float"))
    assert isinstance(df["entity_id"].iloc[0], str)


def test_sanity_check_passes_on_clean_file(tmp_path):
    path = _write(
        tmp_path,
        "s1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tA\t1 Maple Street\tUS\n"
        "S1-00002\tB\t2 Cedar Road\tIndia\n",
    )
    df, report = safe_io.load_source(path)
    assert report.ok
    assert report.n_rows == 2
    assert report.n_raw_lines == 3


def test_sanity_check_catches_wrong_field_count(tmp_path):
    """Check 2: a row with a stray tab must fail loudly, not silently parse."""
    path = _write(
        tmp_path,
        "bad.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tA\t1 Maple Street\tUS\n"
        "S1-00002\tB\textra\ttab\tIndia\n",
    )
    df = safe_io.read_source(path)
    with pytest.raises(safe_io.SanityCheckError) as excinfo:
        safe_io.check_source(df, path)
    assert "field count" in str(excinfo.value)


def test_sanity_check_catches_row_count_mismatch(tmp_path):
    path = _write(
        tmp_path,
        "short.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tA\t1 Maple Street\tUS\n",
    )
    df = safe_io.read_source(path)
    df = df.iloc[0:0]
    with pytest.raises(safe_io.SanityCheckError) as excinfo:
        safe_io.check_source(df, path)
    assert "row count" in str(excinfo.value)


def test_namespace_check_catches_cross_source_collision():
    """io_rules.md section 2: bare numbers collide, so prefixes are the key."""
    import pandas as pd

    frames = {
        "S1": pd.DataFrame({"entity_id": ["S1-00001"]}),
        "S2": pd.DataFrame({"entity_id": ["S1-00001"]}),
    }
    with pytest.raises(safe_io.SanityCheckError) as excinfo:
        safe_io.check_namespace(frames)
    message = str(excinfo.value)
    assert "lack prefix" in message or "share" in message


def test_namespace_check_accepts_disjoint_sources():
    import pandas as pd

    frames = {
        "S1": pd.DataFrame({"entity_id": ["S1-00001", "S1-00002"]}),
        "S2": pd.DataFrame({"entity_id": ["S2-00001"]}),
        "S3": pd.DataFrame({"entity_id": ["S3-00001"]}),
    }
    assert safe_io.check_namespace(frames) == []


# --- A2 normalisation --------------------------------------------------------


def test_accent_fold_keeps_both_forms():
    """research.md 3.1: keep the accented and folded variants side by side."""
    plain = normalise.normalise_record("Societe Generale SA", "1 Rue des Peupliers", "France")
    accented = normalise.normalise_record("Société Générale SA", "1 Rue des Peupliers", "France")

    # the accented form is preserved, not blanked out
    assert "société" in accented.name_norm
    # and the folded variant reconciles the two spellings
    assert accented.name_core_ascii == plain.name_core_ascii == "societe generale"


def test_accented_letters_are_never_dropped_as_punctuation():
    """Regression: the alnum filter used to blank accents, giving "soci t"."""
    record = normalise.normalise_record("Café Lumière", "2 Rue Émile", "France")
    assert "café" in record.name_norm
    assert record.name_core_ascii == "cafe lumiere"
    assert "émile" in record.addr_norm


def test_ampersand_and_punctuation_variants_collapse():
    a = normalise.normalise_name("Smith & Sons")
    b = normalise.normalise_name("Smith and Sons")
    assert a == b
    assert normalise.normalise_name("O’Brien") == normalise.normalise_name("O'Brien")


def test_legal_suffix_is_detected_and_kept_as_feature():
    """A3.2: suffixes become a conflict feature, not a silent deletion."""
    record = normalise.normalise_record("Brightleaf Traders Pvt Ltd", "5 Gandhi Marg", "India")
    assert "pvt ltd" in record.legal_suffixes
    assert "pvt" not in record.name_core
    assert "brightleaf" in record.name_core


def test_suffix_conflict_is_visible_across_countries():
    a = normalise.normalise_record("Acme Pvt Ltd", "1 Nehru Nagar", "India")
    b = normalise.normalise_record("Acme LLP", "1 Nehru Nagar", "India")
    assert a.legal_suffixes != b.legal_suffixes
    assert a.name_core == b.name_core


def test_multiword_suffix_is_stripped_whole():
    """Regression: "pvt ltd" lost "ltd" first and left a stray "pvt"."""
    assert normalise.strip_legal_suffixes("brightleaf pvt ltd") == "brightleaf"
    assert normalise.strip_legal_suffixes("acme private limited") == "acme"
    assert normalise.strip_legal_suffixes("acme pvt. ltd.") == "acme"
    # no orphaned fragment of a multiword suffix may survive
    for text in ("brightleaf pvt ltd", "acme private limited"):
        core = normalise.strip_legal_suffixes(text)
        assert "pvt" not in core and "ltd" not in core and "limited" not in core


def test_descriptor_suffixes_from_research_are_stripped_too():
    """research.md 3.1 lists Enterprises/Traders under India legal suffixes.

    They are descriptors rather than legal forms, so stripping them is
    aggressive -- but it is what the spec says, and A2.4's IDF down-weighting is
    the documented place to soften it. Pinned here so the behaviour is a visible
    decision rather than an accident.
    """
    assert normalise.strip_legal_suffixes("brightleaf traders pvt ltd") == "brightleaf"


def test_societe_is_not_stripped_from_a_trade_name():
    """research.md 3.1 uses "Societe Generale" as a name to preserve."""
    record = normalise.normalise_record("Societe Generale SA", "1 Rue x", "France")
    assert record.name_core == "societe generale"


def test_suffix_only_name_keeps_its_core():
    """Stripping must never empty the name -- that leaves no signal at all."""
    assert normalise.strip_legal_suffixes("enterprises") == "enterprises"


def test_french_suffixes_are_present_in_sprint_0():
    """A3.4: FR rules ship in Sprint 0, not Sprint 2 -- France is test-only."""
    record = normalise.normalise_record("Maison Verte SARL", "2 Avenue Clairval", "France")
    assert "sarl" in record.legal_suffixes
    assert "maison verte" == record.name_core


def test_street_abbreviations_expand():
    assert "street" in normalise.normalise_address("12 Maple St")
    assert "road" in normalise.normalise_address("9 Cedar Rd")
    assert "boulevard" in normalise.normalise_address("4 Harbour Blvd")


def test_canonical_india_tokens_are_not_rewritten():
    out = normalise.normalise_address("5 Gandhi Marg, Nehru Nagar")
    assert "marg" in out
    assert "nagar" in out


def test_postcode_shapes_per_country():
    assert normalise.extract_postcode("12 maple street fairhaven 94105", "US") == "94105"
    assert normalise.extract_postcode("5 gandhi marg nashik 422001", "India") == "422001"
    assert normalise.extract_postcode("2 avenue clairval valmont 75008", "France") == "75008"


def test_unseen_country_still_extracts_a_postcode():
    """io_rules.md section 3: an unseen label must still flow through."""
    assert normalise.extract_postcode("2 rue x 75008", "Atlantis") == "75008"
    assert normalise.extract_postcode("2 rue x 75008", "") == "75008"


def test_house_number_and_unit_are_parsed_separately():
    """research.md 5: Suite 200 vs 210 needs parsed fields, not fuzzy ratios."""
    a = normalise.extract_numbers(normalise.normalise_address("120 Maple St, Ste 200"))
    b = normalise.extract_numbers(normalise.normalise_address("120 Maple St, Ste 210"))
    assert a["house_number"] == b["house_number"] == "120"
    assert a["units"] != b["units"]


def test_landmark_flag_detects_all_locales():
    assert normalise.landmark_flag(normalise.normalise_address("near the bus stand"))
    assert normalise.landmark_flag(normalise.normalise_address("opp main market"))
    assert normalise.landmark_flag(normalise.normalise_address("2 rue x en face de la mairie"))


def test_transliteration_variants_collapse():
    a = normalise.normalise_record("Shree Ganesh Traders", "1 Nehru Nagar", "India")
    b = normalise.normalise_record("Shri Ganesh Traders", "1 Nehru Nagar", "India")
    assert a.name_tokens == b.name_tokens


def test_phonetic_key_is_stable_for_variants():
    a = normalise.double_metaphone("agarwal")
    b = normalise.double_metaphone("aggarwal")
    assert a and a == b


def test_phonetic_fallback_works_without_the_metaphone_package():
    """A2.7: the built-in reduction must stand in when `metaphone` is absent."""
    fallback = normalise._fallback_phonetic
    assert fallback("agarwal") == fallback("aggarwal")
    assert fallback("shri") == fallback("shree")
    # and it must still separate genuinely different names
    assert fallback("brightleaf") != fallback("ironwood")


def test_phonetic_key_is_empty_for_missing_name():
    record = normalise.normalise_record("NA", "1 Maple St", "US")
    assert record.phonetic_key == ""


def test_country_label_is_kept_not_mapped():
    """io_rules.md section 3: normalise casing but keep the label."""
    assert normalise.normalise_country("FRANCE") == "france"
    assert normalise.normalise_country("  United States ") == "united states"
    assert normalise.normalise_country("Atlantis") == "atlantis"


def test_normalised_contract_is_complete():
    """A2.1: B, C and D code against this field list."""
    record = normalise.normalise_record("Acme Inc", "1 Maple St, Fairhaven 94105", "US")
    assert set(record.as_dict()) == set(normalise.NORMALISED_FIELDS)


# --- A4 placeholders ---------------------------------------------------------


@pytest.mark.parametrize("value", ["", "-", "0", "NA", "None", "null", "n/a", "  ", "--", "000"])
def test_placeholders_are_detected(value):
    assert placeholders.is_placeholder(value)


@pytest.mark.parametrize("value", ["NA Foods", "Zero Degrees Cafe", "A1 Traders"])
def test_real_names_are_not_placeholders(value):
    assert not placeholders.is_placeholder(value)


def test_placeholder_address_yields_missing_flag_and_empty_norm():
    """io_rules.md section 4: an empty address is not a match signal."""
    record = normalise.normalise_record("Brightleaf Pharmacy", "NA", "US")
    assert record.addr_missing
    assert record.addr_norm == ""
    assert record.postcode is None
    assert not record.landmark_flag


def test_two_placeholder_addresses_share_no_signal():
    a = normalise.normalise_record("Alpha Traders", "NA", "India")
    b = normalise.normalise_record("Beta Traders", "-", "India")
    assert a.addr_missing and b.addr_missing
    assert a.addr_norm == b.addr_norm == ""
    assert a.postcode is None and b.postcode is None


def test_discover_placeholders_finds_frequent_short_junk():
    values = ["NA"] * 20 + [f"Business {i}" for i in range(80)]
    found = placeholders.discover_placeholders(values)
    assert "na" in found


# --- A5 proxy data ----------------------------------------------------------


def test_proxy_generates_and_round_trips(tmp_path):
    cfg = proxy_data.ProxyConfig(n_entities=60, seed=7)
    s1, s2, s3, truth = proxy_data.generate(cfg)

    assert len(s1) == 60
    assert len(truth) == 60
    assert s2 and s3

    paths = {}
    for name, records in (("s1", s1), ("s2", s2), ("s3", s3)):
        paths[name] = proxy_data.write_tsv(records, tmp_path / f"{name}.tsv")

    frames = {}
    for source, name in (("S1", "s1"), ("S2", "s2"), ("S3", "s3")):
        df, report = safe_io.load_source(paths[name])
        assert report.ok, report
        frames[source] = df

    assert safe_io.check_namespace(frames) == []


def test_proxy_is_deterministic():
    a = proxy_data.generate(proxy_data.ProxyConfig(n_entities=30, seed=11))
    b = proxy_data.generate(proxy_data.ProxyConfig(n_entities=30, seed=11))
    assert a[0] == b[0]
    assert a[3] == b[3]


def test_singleton_rate_is_a_knob():
    """A0.4: the true rate is unknown, so it must stay parameterised."""
    low = proxy_data.generate(proxy_data.ProxyConfig(n_entities=200, singleton_rate=0.1, seed=3))
    high = proxy_data.generate(proxy_data.ProxyConfig(n_entities=200, singleton_rate=0.9, seed=3))
    low_rate = audit.cardinality_report(low[3])["singleton_rate"]
    high_rate = audit.cardinality_report(high[3])["singleton_rate"]
    assert low_rate < high_rate


def test_france_is_test_only():
    """A5.2 mirrors io_rules.md section 3: test adds France, train has none."""
    cfg = proxy_data.ProxyConfig(n_entities=300, seed=5)
    s1, s2, s3, truth = proxy_data.generate(cfg)
    split = proxy_data.split_train_test(s1, s2, s3, truth, cfg)

    train_countries = {r["country"] for r in split["train"][0]}
    test_countries = {r["country"] for r in split["test"][0]}
    assert "France" not in train_countries
    assert "France" in test_countries


# --- audit probes -----------------------------------------------------------


def test_fragment_belongs_to_at_most_one_s1():
    """A0.2 / research.md Q3: expected 0 fragments with multiple owners."""
    _, _, _, truth = proxy_data.generate(proxy_data.ProxyConfig(n_entities=150, seed=13))
    assert audit.fragments_in_multiple_s1(truth) == {}


def test_per_vendor_counts_report_maxima():
    """A0.3 / Q4: reports the max so D knows whether to enable bipartite N2."""
    _, _, _, truth = proxy_data.generate(proxy_data.ProxyConfig(n_entities=150, seed=17))
    _, maxima = audit.per_vendor_match_counts(truth)
    assert set(maxima) <= {"S2", "S3"}
    assert all(v >= 1 for v in maxima.values())


def test_token_idf_ranks_rare_tokens_higher():
    token_lists = [["brightleaf", "pharmacy"]] + [["city", "pharmacy"]] * 20
    idf, df = audit.token_idf(token_lists)
    assert idf["brightleaf"] > idf["pharmacy"]
    assert df["pharmacy"] == 21


def test_country_report_exposes_exact_labels():
    """A0.5: the audit is how "France" vs "FR" gets settled."""
    import pandas as pd

    frames = {
        "S1": pd.DataFrame(
            {
                "entity_id": ["S1-00001", "S1-00002"],
                "business_name": ["A", "B"],
                "business_address": ["x", "y"],
                "country": ["France", "FR"],
            }
        )
    }
    report = audit.country_report(frames)
    assert report["S1"]["raw_values"] == {"France": 1, "FR": 1}
    assert report["S1"]["n_distinct_raw"] == 2


def test_cardinality_buckets_match_research_slices():
    truth = {
        "S1-1": set(),
        "S1-2": {"S2-1"},
        "S1-3": {"S2-2", "S3-1"},
        "S1-4": {"S2-3", "S2-4", "S3-2", "S3-3", "S3-4"},
    }
    report = audit.cardinality_report(truth)
    assert report["buckets"] == {"t=0": 1, "t=1": 1, "t=2-4": 1, "t>=5": 1}
    assert report["singleton_rate"] == 0.25


# --- Indic script handling (A0.6 resolved: 41% S2 / 32% S3 India non-Latin) --


def test_indic_combining_marks_are_never_stripped_as_punctuation():
    """Regression: Indic vowel signs and virama are categories Mn/Mc, for which
    str.isalnum() is False. Testing isalnum alone blanked every one of them and
    shattered a Malayalam name down to its first character."""
    raw = "\u0d21\u0d4d\u0d30\u0d40\u0d02 \u0d2e\u0d40\u0d21\u0d3f\u0d2f"
    out = normalise.normalise_name(raw)
    assert out == raw.casefold(), out
    assert len(out) > 5, f"Indic text collapsed to {out!r}"


def test_indic_scripts_survive_normalisation():
    for raw in (
        "\u092e\u0949\u0921\u0930\u094d\u0928 \u091f\u0947\u0915\u094d\u0928\u094b\u0932\u0949\u091c\u0940\u091c",
        "\u0b9a\u0bbf\u0b9f\u0bcd\u0b9f\u0bbf \u0b85\u0b95\u0bcd\u0bb0\u0bcb",
        "\u0987\u09a8\u09cd\u09a6\u09cb",
        "\u0ab8\u0abf\u0a9f\u0ac0",
    ):
        out = normalise.normalise_name(raw)
        assert len(out) >= len(raw) - 2, f"{raw!r} collapsed to {out!r}"


def test_transliterated_legal_suffix_is_stripped():
    """The suffix is transliterated along with the name, so a Latin-only suffix
    list never reaches name_core on ~40% of India fragments."""
    cases = [
        ("\u0d21\u0d4d\u0d30\u0d40\u0d02 \u0d2e\u0d40\u0d21\u0d3f\u0d2f \u0d2a\u0d4d\u0d30\u0d48\u0d35\u0d31\u0d4d\u0d31\u0d4d \u0d32\u0d3f\u0d2e\u0d3f\u0d31\u0d4d\u0d31\u0d21\u0d4d",
         "\u0d21\u0d4d\u0d30\u0d40\u0d02 \u0d2e\u0d40\u0d21\u0d3f\u0d2f"),
        ("\u0938\u093f\u091f\u0940 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921",
         "\u0938\u093f\u091f\u0940"),
    ]
    for raw, expected_core in cases:
        r = normalise.normalise_record(raw, "Mumbai", "India")
        assert r.name_core == expected_core, (raw, r.name_core)
        assert r.legal_suffixes, f"no suffix detected for {raw!r}"


def test_latin_names_are_unaffected_by_the_indic_fix():
    assert normalise.normalise_name("Acme Hardware LLC") == "acme hardware llc"
    assert normalise.normalise_name("Soci\u00e9t\u00e9 G\u00e9n\u00e9rale SA") == "soci\u00e9t\u00e9 g\u00e9n\u00e9rale sa"
    r = normalise.normalise_record("Gangapur Foods Ventures Pvt Ltd", "Nashik", "India")
    assert r.name_core == "gangapur foods ventures"
