"""Owner C tests: pair feature library and Stage-A scorer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import features as F, matcher  # noqa: E402
from ber.normalise import normalise_record  # noqa: E402


def rec(name, addr, country="India"):
    return normalise_record(name, addr, country)


A = rec("Shree Ganesh Traders Pvt Ltd", "5 Gandhi Marg, Nashik 422001")
B = rec("Shri Ganesh Traders Private Limited", "5 Gandhi Marg, Nashik 422001")
C = rec("Acme Hardware LLP", "120 Maple St Ste 210, Fairhaven 94105", "US")


def _idf(*records):
    return F.token_idf([r.name_tokens for r in records])


def _feats(x, y, **kw):
    idf = _idf(A, B, C)
    return F.pair_features(x, y, idf=idf, default_idf=F.default_idf_value(idf), **kw)


def test_features_separate_a_match_from_a_non_match():
    match = _feats(A, B)
    other = _feats(A, C)
    assert match["name_core_ratio"] > other["name_core_ratio"]
    assert match["idf_jaccard"] > other["idf_jaccard"]
    assert match["postcode_state"] == 1 and other["postcode_state"] == 3


def test_row_length_matches_the_registry():
    assert len(F.features_to_row(_feats(A, B))) == len(F.FEATURE_NAMES)


def test_missing_is_distinct_from_different():
    """io_rules.md 4 / research.md 5: a blank address is not evidence of a
    mismatch, so the three-state encoding must not collapse them."""
    blank = rec("Alpha Traders", "NA")
    assert _feats(blank, A)["house_state"] == 0          # missing
    assert _feats(A, C)["house_state"] == 2              # different
    assert _feats(A, B)["house_state"] == 1              # equal


def test_suffix_conflict_is_a_signal_not_a_deletion():
    """research.md 3.1 / A3.2: "X Pvt Ltd" and "X LLP" share a name_core but are
    different companies."""
    llc = rec("Acme Hardware LLC", "120 Maple St, Fairhaven 94105", "US")
    llp = rec("Acme Hardware LLP", "120 Maple St, Fairhaven 94105", "US")
    assert llc.name_core == llp.name_core
    assert _feats(llc, llp)["suffix_state"] == 2


def test_acronym_feature_fires():
    ibm = rec("IBM", "1 New Orchard Road", "US")
    full = rec("International Business Machines", "1 New Orchard Road", "US")
    assert _feats(ibm, full)["acronym"] == 1


def test_postcode_prefix_is_its_own_state():
    near = rec("Alpha", "9 Main Street, Nashik 422009")
    assert _feats(A, near)["postcode_state"] == 2


def test_listwise_features_reflect_rank():
    top = _feats(A, B, rrf_score=0.03, rrf_rank=0, cand_count=10, best_score=0.03)
    low = _feats(A, B, rrf_score=0.01, rrf_rank=7, cand_count=10, best_score=0.03)
    assert top["is_best_candidate"] == 1 and low["is_best_candidate"] == 0
    assert low["score_gap_to_best"] > top["score_gap_to_best"]


def test_chain_and_shared_address_counts():
    records = {"a": A, "b": B, "c": C}
    ncc, adc = F.corpus_counts(records)
    assert adc[A.addr_norm] == 2      # A and B share an address
    assert all(v >= 1 for v in ncc.values())


def test_unseen_token_gets_the_rarest_idf():
    idf = _idf(A, B)
    assert F.default_idf_value(idf) >= max(idf.values())


# --- scorer -----------------------------------------------------------------


def _toy_training_set(n=400):
    fused, fs, fr, s1, truth = {}, {}, {}, {}, {}
    for i in range(n):
        s1_id, frag_id = f"S1-{i}", f"S2-{i}"
        s1[s1_id] = rec(f"Alpha{i} Foods Pvt Ltd", f"{i} Main Road, Pune 411001")
        fr[frag_id] = rec(f"Alpha{i} Foods Private Limited",
                          f"{i} Main Road, Pune 411001")
        decoy = f"S1-{(i + 1) % n}"
        fused[frag_id] = [s1_id, decoy]
        fs[frag_id] = {s1_id: 0.03, decoy: 0.02}
        truth[s1_id] = {frag_id}
    return fused, fs, fr, s1, truth


def test_training_pairs_label_from_ground_truth():
    fused, fs, fr, s1, truth = _toy_training_set(50)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, groups = matcher.build_training_pairs(fused, fs, fr, s1, truth,
                                                idf, ncc, adc)
    assert X.shape == (100, len(F.FEATURE_NAMES))
    assert y.sum() == 50
    assert len(set(groups)) == 50


def test_scorer_separates_positives_from_negatives():
    fused, fs, fr, s1, truth = _toy_training_set(300)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    model = matcher.PairScorer(n_estimators=60).fit(X, y)
    p = model.predict(X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3


def test_score_candidates_returns_decoder_shaped_marginals():
    """Output must be {fragment: {s1: p}} -- exactly what
    decode.pipeline_predictions takes as `marginals`."""
    fused, fs, fr, s1, truth = _toy_training_set(120)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    model = matcher.PairScorer(n_estimators=40).fit(X, y)
    marg = matcher.score_candidates(model, fused, fs, fr, s1, idf, ncc, adc)
    assert set(marg) == set(fused)
    for frag_id, row in marg.items():
        assert set(row) == set(fused[frag_id])
        assert all(0.0 <= v <= 1.0 for v in row.values())


def test_model_beats_the_rank_placeholder():
    """The whole point of Owner C: the decoder's rank heuristic is a floor."""
    from ber import decode, scorer as sc
    fused, fs, fr, s1, truth = _toy_training_set(200)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    model = matcher.PairScorer(n_estimators=60).fit(X, y)
    marg = matcher.score_candidates(model, fused, fs, fr, s1, idf, ncc, adc)

    s1_ids = list(s1)
    base = sc.macro_f_beta(truth, decode.pipeline_predictions(fused, s1_ids)).macro_f05
    tuned = sc.macro_f_beta(
        truth, decode.pipeline_predictions(fused, s1_ids, marginals=marg)).macro_f05
    assert tuned >= base


def test_scorer_rejects_single_class_input():
    X = np.zeros((10, len(F.FEATURE_NAMES)), dtype="float32")
    with pytest.raises(ValueError):
        matcher.PairScorer().fit(X, np.zeros(10, dtype="int8"))


# --- persistence: the test split has no ground truth to train on -------------


def test_scorer_round_trips(tmp_path):
    fused, fs, fr, s1, truth = _toy_training_set(150)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    model = matcher.PairScorer(n_estimators=40).fit(X, y)

    path = matcher.save_scorer(model, tmp_path / "m.pkl")
    back = matcher.load_scorer(path)
    assert back.backend == model.backend
    assert np.allclose(back.predict(X), model.predict(X))


def test_index_stats_round_trip(tmp_path):
    """idf and the chain counts are computed over S1, so recomputing them on the
    test split would shift the feature distribution the model was fitted on."""
    _, _, _, s1, _ = _toy_training_set(30)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    path = matcher.save_index_stats(tmp_path / "s.pkl", idf, ncc, adc)
    i2, n2, a2 = matcher.load_index_stats(path)
    assert i2 == idf and n2 == ncc and a2 == adc


def test_loading_fails_loudly_on_a_feature_registry_change(tmp_path, monkeypatch):
    """A silent column mismatch would score garbage, so it must raise."""
    fused, fs, fr, s1, truth = _toy_training_set(60)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    path = matcher.save_scorer(matcher.PairScorer(n_estimators=20).fit(X, y),
                               tmp_path / "m.pkl")
    monkeypatch.setattr(F, "FEATURE_NAMES", F.FEATURE_NAMES[:-1])
    with pytest.raises(ValueError, match="feature registry changed"):
        matcher.load_scorer(path)


def test_reusing_the_design_matrix_gives_identical_scores():
    """The training path already featurises every candidate pair, so
    score_candidates recomputing them doubled the scoring stage for no gain."""
    fused, fs, fr, s1, truth = _toy_training_set(150)
    idf = F.token_idf([r.name_tokens for r in s1.values()])
    ncc, adc = F.corpus_counts(s1)
    X, y, _ = matcher.build_training_pairs(fused, fs, fr, s1, truth, idf, ncc, adc)
    model = matcher.PairScorer(n_estimators=40).fit(X, y)

    recomputed = matcher.score_candidates(model, fused, fs, fr, s1, idf, ncc, adc)
    reused = matcher.score_candidates(model, fused, fs, fr, s1, idf, ncc, adc, X=X)
    assert set(recomputed) == set(reused)
    for frag in recomputed:
        for s1_id, p in recomputed[frag].items():
            assert reused[frag][s1_id] == pytest.approx(p, abs=1e-12)
