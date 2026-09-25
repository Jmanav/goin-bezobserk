"""Owner B tests: blocking channels, RRF fusion, PC/RR reporting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import blocking, blocking_report, dense  # noqa: E402
from ber.normalise import normalise_record  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name",
                                       "business_address", "country"])


S1 = _frame([
    ("S1-1", "Acme Hardware LLC", "120 Maple St, Fairhaven 94105", "US"),
    ("S1-2", "Brightleaf Traders Pvt Ltd", "5 Gandhi Marg, Nashik 422001", "India"),
    ("S1-3", "Cafe Lumiere SARL", "2 Rue Emile Zola, Valmont 75008", "France"),
    ("S1-4", "Riverside Clinic", "88 Oak Ave, Fairhaven 94105", "US"),
])


# --- B1 char TF-IDF ---------------------------------------------------------


def test_char_tfidf_retrieves_a_typo_variant():
    ch = blocking.CharTfidfChannel(top_n=5).fit(
        list(S1["entity_id"]), list(S1["business_name"]))
    res = ch.query(["S2-1"], ["Acme Hardwear LLC"])
    hits = [s1 for s1, _, _ in res.hits["S2-1"]]
    assert "S1-1" in hits


def test_channels_emit_rank_alongside_score():
    """B1.4/B2.3: Owner C consumes per-channel ranks as features and RRF needs
    them, so a channel that returns scores only is unusable downstream."""
    ch = blocking.CharTfidfChannel(top_n=5).fit(
        list(S1["entity_id"]), list(S1["business_name"]))
    rows = ch.query(["S2-1"], ["Acme Hardware LLC"]).hits["S2-1"]
    assert rows
    assert [r[2] for r in rows] == list(range(len(rows)))
    assert all(isinstance(r[1], float) for r in rows)


def test_char_tfidf_respects_min_cosine():
    ch = blocking.CharTfidfChannel(top_n=50, min_cos=0.99).fit(
        list(S1["entity_id"]), list(S1["business_name"]))
    assert ch.query(["S2-1"], ["totally unrelated string"]).hits["S2-1"] == []


# --- B2 BM25 ----------------------------------------------------------------


def test_bm25_ranks_a_rare_token_above_a_common_one():
    """The role of this channel (research.md 3.2) is rare tokens."""
    ids = ["S1-a", "S1-b", "S1-c"]
    docs = ["clinic fairhaven", "clinic riverside", "clinic oakwood"]
    ch = blocking.BM25Channel(top_n=3).fit(ids, docs)
    hits = ch.query(["f1"], ["riverside clinic"]).hits["f1"]
    assert hits[0][0] == "S1-b"


def test_bm25_scores_are_positive_and_ordered():
    ch = blocking.BM25Channel(top_n=4).fit(
        list(S1["entity_id"]),
        [f"{n} {a}" for n, a in zip(S1["business_name"], S1["business_address"])])
    hits = ch.query(["f1"], ["acme hardware maple fairhaven"]).hits["f1"]
    scores = [s for _, s, _ in hits]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0


# --- B4 exact keys ----------------------------------------------------------


def _records(frame):
    return {r.entity_id: normalise_record(r.business_name, r.business_address,
                                          r.country)
            for r in frame.itertuples(index=False)}


def test_key_channel_matches_on_postcode_and_house_number():
    ch = blocking.KeyChannel().fit(_records(S1))
    frag = {"S2-1": normalise_record("Acme Hdwr", "120 Maple Street, 94105", "US")}
    hits = [s1 for s1, _, _ in ch.query(frag).hits["S2-1"]]
    assert "S1-1" in hits


def test_placeholders_never_generate_keys():
    """io_rules.md 4 / B4.3: two records must never become candidates *because*
    they share a placeholder."""
    recs = {
        "S1-x": normalise_record("Alpha Traders", "NA", "India"),
        "S1-y": normalise_record("Beta Traders", "-", "India"),
    }
    assert blocking.build_key_index(recs) == {}


def test_key_collisions_are_capped():
    """research.md 5 'Runtime blow-ups': a key shared by half the corpus is a
    quadratic time bomb, not a signal."""
    recs = {
        f"S1-{i}": normalise_record(f"Shop {i}", "1 Main Street, 94105", "US")
        for i in range(blocking.KEY_COLLISION_CAP + 50)
    }
    index = blocking.build_key_index(recs)
    assert all(len(v) <= blocking.KEY_COLLISION_CAP for v in index.values())


# --- B5 RRF -----------------------------------------------------------------


def test_rrf_formula_is_exact():
    """research.md 3.2 fusion row: score = sum 1 / (60 + rank)."""
    a = blocking.ChannelResult("a", {"f": [("S1-1", 0.9, 0), ("S1-2", 0.8, 1)]})
    b = blocking.ChannelResult("b", {"f": [("S1-2", 0.7, 0)]})
    fused = blocking.reciprocal_rank_fusion([a, b], k=2)
    # S1-2 = 1/61 + 1/60 beats S1-1 = 1/60
    assert fused["f"][0] == "S1-2"


def test_rrf_keeps_top_k():
    hits = [(f"S1-{i}", 1.0 - i / 100, i) for i in range(50)]
    ch = blocking.ChannelResult("a", {"f": hits})
    assert len(blocking.reciprocal_rank_fusion([ch], k=25)["f"]) == 25


def test_confident_key_hits_survive_the_top_k_cut():
    """research.md 3.2: keep top-K 'plus all key hits with <= 5 collisions'."""
    hits = [(f"S1-{i}", 1.0 - i / 100, i) for i in range(50)]
    ch = blocking.ChannelResult("a", {"f": hits})
    fused = blocking.reciprocal_rank_fusion(
        [ch], k=5, always_keep={"f": {"S1-49"}})
    assert "S1-49" in fused["f"]
    assert len(fused["f"]) == 6


def test_invert_to_s1_dedupes_and_sorts():
    inverted = blocking.invert_to_s1({"S2-1": ["S1-a", "S1-b"], "S3-1": ["S1-a"]})
    assert inverted["S1-a"] == ["S2-1", "S3-1"]
    assert inverted["S1-b"] == ["S2-1"]


def test_blocking_never_merges_s1_entities():
    """io_rules.md 8 / research.md 3.4: blocking emits candidates, never merged
    components. A fragment matching two S1 rows must leave both intact."""
    fused = {"S2-1": ["S1-a", "S1-b"]}
    inverted = blocking.invert_to_s1(fused)
    assert set(inverted) == {"S1-a", "S1-b"}
    assert inverted["S1-a"] == ["S2-1"] and inverted["S1-b"] == ["S2-1"]


# --- B6 reporting -----------------------------------------------------------


def test_pair_completeness_counts_retrieved_true_pairs():
    truth = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}
    cands = {"S1-1": ["S2-1"], "S1-2": ["S3-1"]}
    assert blocking_report.pair_completeness(cands, truth)["pair_completeness"] == \
        pytest.approx(2 / 3)


def test_reduction_ratio_matches_the_definition():
    rr = blocking_report.reduction_ratio({"S1-1": ["a", "b"]}, n_s1=10, n_fragments=10)
    assert rr["reduction_ratio"] == pytest.approx(1 - 2 / 100)


def test_entity_ceiling_diverges_from_pair_pc():
    """B6.5: pair PC can look excellent while the per-entity ceiling is poor,
    because one missed fragment caps that whole row's F0.5. With 26% of real
    rows at t >= 5 this gap is the number that actually bounds the score."""
    truth = {f"S1-{i}": {f"S2-{i}a", f"S2-{i}b", f"S2-{i}c", f"S2-{i}d"}
             for i in range(10)}
    # retrieve 3 of every 4 -> pair PC 0.75, but ZERO rows fully covered
    cands = {k: sorted(v)[:3] for k, v in truth.items()}
    pc = blocking_report.pair_completeness(cands, truth)["pair_completeness"]
    ceiling = blocking_report.entity_recall_ceiling(cands, truth)
    assert pc == pytest.approx(0.75)
    assert ceiling["entity_recall_ceiling"] == 0.0
    assert ceiling["macro_f05_ceiling"] < 1.0


def test_singletons_count_as_covered():
    """A true singleton needs nothing retrieved, so it cannot cap the ceiling."""
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    ceiling = blocking_report.entity_recall_ceiling({"S1-2": ["S2-1"]}, truth)
    assert ceiling["entity_recall_ceiling"] == 1.0
    assert ceiling["n_singletons"] == 1


def test_pc_vs_k_is_monotonic_in_k():
    hits = {f"f{i}": [(f"S1-{j}", 1.0 - j / 100, j) for j in range(40)]
            for i in range(5)}
    ch = blocking.ChannelResult("a", hits)
    truth = {f"S1-{j}": {f"f{i}" for i in range(5)} for j in range(40)}
    curve = blocking_report.pc_vs_k([ch], truth, ks=(5, 10, 20, 40))
    pcs = [row["pair_completeness"] for row in curve]
    assert pcs == sorted(pcs)


def test_slice_report_splits_by_country_and_vendor():
    truth = {"S1-1": {"S2-1"}, "S1-2": {"S3-1"}}
    cands = {"S1-1": ["S2-1"], "S1-2": []}
    rep = blocking_report.slice_report(cands, truth,
                                       {"S1-1": "us", "S1-2": "france"})
    assert rep["by_country"]["us"]["pair_completeness"] == 1.0
    assert rep["by_country"]["france"]["pair_completeness"] == 0.0
    assert set(rep["by_vendor"]) == {"S2", "S3"}


# --- B3 dense channel -------------------------------------------------------


def test_dense_models_are_licence_cleared_and_under_the_cap():
    """io_rules.md 9: final models MIT/Apache-2.0 and <= 8B params."""
    for key, spec in dense.MODELS.items():
        assert spec["licence"] in {"MIT", "Apache-2.0"}, (key, spec["licence"])
        assert "B" not in spec["params"] or float(
            spec["params"].rstrip("B")) <= 8.0


def test_dense_describe_reports_an_unpinned_revision():
    """research.md 9 requires pinned commits; an unpinned run must say so."""
    record = dense.describe()
    assert "UNPINNED" in record["revision"]
    assert record["licence"] in {"MIT", "Apache-2.0"}


def test_dense_channel_fails_loudly_when_unavailable():
    """B3.5: degrading silently would hide missing recall on ~40% of India
    rows, so absence is an exception, not a shrug."""
    if dense.available():
        pytest.skip("dense stack installed here")
    with pytest.raises(dense.DenseUnavailable):
        dense.DenseChannel().fit(["S1-1"], ["acme"])


# --- B7 harness -------------------------------------------------------------


def test_harness_runs_all_available_channels_and_reports_gaps():
    harness = blocking_report.BlockingHarness(k=10, use_dense=True).fit(S1)
    frags = _frame([
        ("S2-1", "Acme Hardwear", "120 Maple St, Fairhaven 94105", "US"),
        ("S2-2", "Brightleaf Traders", "5 Gandhi Marg, Nashik 422001", "India"),
    ])
    fused, channels, always = harness.query(frags)
    assert set(fused) == {"S2-1", "S2-2"}
    assert "S1-1" in fused["S2-1"]
    assert "S1-2" in fused["S2-2"]
    if not dense.available():
        assert any("DENSE CHANNEL UNAVAILABLE" in n for n in harness.notes)


def test_country_sharding_is_off_by_default():
    """B0.1 is unresolved, so sharding must stay off (research.md 3.2)."""
    assert blocking_report.BlockingHarness().shard_by_country is False


def test_unseen_country_fragments_still_get_candidates():
    """io_rules.md 3: an unseen or missing country label must still flow
    through. France is the case this protects."""
    harness = blocking_report.BlockingHarness(k=10).fit(S1)
    frags = _frame([("S2-9", "Cafe Lumiere", "2 Rue Emile Zola, Valmont 75008",
                     "Atlantis")])
    fused, _, _ = harness.query(frags)
    assert fused["S2-9"], "unseen country label produced no candidates"


def test_bm25_is_case_and_punctuation_insensitive():
    """Regression: BM25 tokenised raw text, so "Acme" and "acme" were distinct
    terms and "St," never matched "St". The harness hid this by passing
    pre-normalised text, but Owner C calls the channel directly."""
    ids = ["S1-1", "S1-2"]
    docs = ["Acme Hardware LLC, 120 Maple St., Fairhaven", "Riverside Clinic"]
    ch = blocking.BM25Channel(top_n=4).fit(ids, docs)
    lower = ch.query(["f"], ["acme hardware maple fairhaven"]).hits["f"]
    upper = ch.query(["f"], ["ACME HARDWARE MAPLE FAIRHAVEN"]).hits["f"]
    assert lower and upper
    assert lower[0][0] == upper[0][0] == "S1-1"
    assert lower[0][1] == pytest.approx(upper[0][1])
