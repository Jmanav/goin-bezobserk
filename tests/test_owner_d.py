"""Owner D tests. The D1.3 table below IS Gate 0 criterion 3."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import decode, scorer, submission  # noqa: E402


# --- D1.3: research.md 1.2 consequence table, reproduced exactly -------------

CONSEQUENCE_TABLE = [
    # (true set, predicted set, expected F0.5, label)
    (set(),                  set(),                  1.000,        "t=0, {} -> free 1.0"),
    (set(),                  {"S2-1"},               0.000,        "t=0, any prediction -> 0.0"),
    ({"S2-1"},               set(),                  0.000,        "t=1, {}"),
    ({"S2-1"},               {"S2-1"},               1.000,        "t=1, correct"),
    ({"S2-1"},               {"S2-1", "S2-9"},       1.25 / 2.25,  "t=1, correct+wrong"),
    ({"S2-1"},               {"S2-9"},               0.000,        "t=1, wrong"),
    ({"S2-1", "S2-2"},       {"S2-1"},               1.25 / 1.5,   "t=2, 1 correct"),
    ({"S2-1", "S2-2"},       {"S2-1", "S2-2"},       1.000,        "t=2, both"),
    ({"S2-1", "S2-2"},       {"S2-1", "S2-2", "S2-9"}, 2.5 / 3.5,  "t=2, both+1 wrong"),
    ({"S2-1", "S2-2", "S2-3", "S2-4"}, {"S2-1"},     1.25 / 2,     "t=4, 1 correct"),
    ({"S2-1", "S2-2", "S2-3", "S2-4"},
     {"S2-1", "S2-2", "S2-3"},                        3.75 / 4,    "t=4, 3 correct"),
]


@pytest.mark.parametrize("true_ids,pred_ids,expected,label", CONSEQUENCE_TABLE)
def test_gate0_consequence_table(true_ids, pred_ids, expected, label):
    got = scorer.entity_f_beta(true_ids, pred_ids)
    assert got == pytest.approx(expected, abs=1e-12), f"{label}: {got} != {expected}"


def test_singleton_rule_is_all_or_nothing():
    """research.md 1.2: a true singleton is a free 1.0 when left empty and a
    total loss on any prediction. No partial credit either way."""
    assert scorer.entity_f_beta(set(), set()) == 1.0
    assert scorer.entity_f_beta(set(), {"S2-1"}) == 0.0
    assert scorer.entity_f_beta(set(), {"S2-1", "S2-2"}) == 0.0


def test_false_merge_costs_more_than_a_miss():
    """The precision tilt: at t=2, missing one costs 0.167 and adding one wrong
    costs 0.286. This asymmetry is the whole reason empty beats guessing."""
    miss = 1.0 - scorer.entity_f_beta({"a", "b"}, {"a"})
    over = 1.0 - scorer.entity_f_beta({"a", "b"}, {"a", "b", "c"})
    assert miss == pytest.approx(1 / 6, abs=1e-9)
    assert over == pytest.approx(2 / 7, abs=1e-9)
    assert over > miss


def test_macro_average_counts_every_truth_row():
    """D1.2: scored over truth's keys. An S1 row the model never emitted still
    scores zero rather than being skipped."""
    truth = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}, "S1-3": set()}
    preds = {"S1-1": {"S2-1"}}          # S1-2 missing entirely, S1-3 absent
    rep = scorer.macro_f_beta(truth, preds)
    assert rep.n_rows == 3
    # S1-1 = 1.0, S1-2 = 0.0 (missing), S1-3 = 1.0 (absent == empty)
    assert rep.macro_f05 == pytest.approx(2 / 3)


def test_order_and_whitespace_do_not_matter():
    """D1.5 / io_rules.md 5.3: order-insensitive, trimmed."""
    assert scorer.entity_f_beta(["S2-1", "S2-2"], ["S2-2", "S2-1"]) == 1.0
    assert scorer.entity_f_beta("S2-1,S2-2", " S2-2 , S2-1 ") == 1.0


def test_slices_partition_without_changing_the_macro():
    truth = {f"S1-{i}": ({"S2-1"} if i % 2 else set()) for i in range(10)}
    preds = {f"S1-{i}": ({"S2-1"} if i % 2 else set()) for i in range(10)}
    rep = scorer.macro_f_beta(truth, preds, slices={"t": scorer.cardinality_slice(truth)})
    assert rep.macro_f05 == pytest.approx(1.0)
    assert sum(b["n_rows"] for b in rep.per_slice["t"].values()) == 10


# --- D3 submission writer ---------------------------------------------------


def test_writer_emits_one_row_per_s1_including_singletons(tmp_path):
    """io_rules.md 5.1: a missing S1 row is a rejection; singletons keep their
    row with a blank field."""
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    preds = {"S1-1": ["S2-9", "S3-4"], "S1-3": []}      # S1-2 never predicted
    out = tmp_path / "matching_results.tsv"
    submission.write_matching_results(out, s1_ids, preds)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 4
    assert lines[1] == "S1-1\tS2-9,S3-4"
    assert lines[2] == "S1-2\t"       # blank, but present
    assert lines[3] == "S1-3\t"


def test_writer_dedupes_sorts_and_never_quotes(tmp_path):
    out = tmp_path / "m.tsv"
    submission.write_matching_results(out, ["S1-1"], {"S1-1": ["S3-2", "S2-1", "S3-2"]})
    body = out.read_text(encoding="utf-8").splitlines()[1]
    assert body == "S1-1\tS2-1,S3-2"
    assert '"' not in body
    assert ", " not in body


def test_writer_rejects_self_matches_and_unknown_prefixes(tmp_path):
    out = tmp_path / "m.tsv"
    with pytest.raises(submission.SubmissionError, match="S1-"):
        submission.write_matching_results(out, ["S1-1"], {"S1-1": ["S1-2"]})


def test_reader_round_trips(tmp_path):
    out = tmp_path / "m.tsv"
    preds = {"S1-1": ["S2-1", "S3-2"], "S1-2": []}
    submission.write_matching_results(out, ["S1-1", "S1-2"], preds)
    back = submission.read_submission(out)
    assert back == {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()}


# --- D4 validator -----------------------------------------------------------


def _write(path, rows, header="source1_entity_id\tmatched_entity_ids"):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        for k, v in rows:
            fh.write(f"{k}\t{v}\n")


def test_validator_passes_a_well_formed_submission(tmp_path):
    m = tmp_path / "matching_results.tsv"
    c = tmp_path / "candidate_pairs.tsv"
    _write(m, [("S1-1", "S2-1"), ("S1-2", "")])
    _write(c, [("S1-1", "S2-1,S2-5"), ("S1-2", "S3-9")],
           "source1_entity_id\tcandidate_entity_ids")
    rep = submission.validate(m, c, s1_ids=["S1-1", "S1-2"],
                              fragment_ids={"S2-1", "S2-5", "S3-9"})
    assert rep.ok, rep.problems


def test_validator_catches_a_missing_s1_row(tmp_path):
    m = tmp_path / "m.tsv"
    _write(m, [("S1-1", "S2-1")])
    rep = submission.validate(m, None, s1_ids=["S1-1", "S1-2"],
                              fragment_ids={"S2-1"})
    assert not rep.ok
    assert any("missing" in p.lower() for p in rep.problems)


def test_validator_catches_matches_not_in_candidates(tmp_path):
    """io_rules.md 5.2 / D4.3: a match outside the candidate set is a pipeline
    bug and must be surfaced loudly, not warned about."""
    m = tmp_path / "m.tsv"
    c = tmp_path / "c.tsv"
    _write(m, [("S1-1", "S2-7")])
    _write(c, [("S1-1", "S2-1")], "source1_entity_id\tcandidate_entity_ids")
    rep = submission.validate(m, c, s1_ids=["S1-1"], fragment_ids={"S2-1", "S2-7"})
    assert not rep.ok
    assert any("candidate" in p.lower() for p in rep.problems)


def test_validator_catches_ids_absent_from_the_test_set(tmp_path):
    m = tmp_path / "m.tsv"
    _write(m, [("S1-1", "S2-999")])
    rep = submission.validate(m, None, s1_ids=["S1-1"], fragment_ids={"S2-1"})
    assert not rep.ok
    assert any("not in the test set" in p.lower() for p in rep.problems)


def test_validator_catches_duplicate_s1_rows(tmp_path):
    m = tmp_path / "m.tsv"
    _write(m, [("S1-1", "S2-1"), ("S1-1", "S2-2")])
    rep = submission.validate(m, None, s1_ids=["S1-1"], fragment_ids={"S2-1", "S2-2"})
    assert not rep.ok
    assert any("duplicate" in p.lower() for p in rep.problems)


def test_unseen_country_rows_still_appear(tmp_path):
    """io_rules.md 3: an unseen or missing country label must still reach the
    submission. France is the case this protects."""
    out = tmp_path / "m.tsv"
    s1_ids = ["S1-fr", "S1-us"]
    submission.write_matching_results(out, s1_ids, {"S1-us": ["S2-1"]})
    back = submission.read_submission(out)
    assert set(back) == set(s1_ids)


# --- D6 decoder: research.md 1.3 worked examples -----------------------------


def test_worked_example_a_single_candidate_breakeven_is_half():
    """research.md 1.3 A: pi=0.6 -> predict; pi=0.4 -> abstain; break-even 0.5."""
    assert decode.expected_f_beta(1, [0.6], lam=0.0) == pytest.approx(0.6)
    assert decode.expected_f_beta(0, [0.6], lam=0.0) == pytest.approx(0.4)
    assert decode.best_set([0.6], lam=0.0) == [0]
    assert decode.best_set([0.4], lam=0.0) == []
    assert decode.expected_f_beta(1, [0.5], lam=0.0) == pytest.approx(
        decode.expected_f_beta(0, [0.5], lam=0.0)
    )


def test_worked_example_b_two_candidates_rejects_the_second():
    """research.md 1.3 B: pi=(0.9, 0.5) -> {} 0.05, {1} 0.825, {1,2} 0.728.
    The second candidate at 0.5 is rejected: once one match is predicted, the
    effective threshold for adding another is above 0.5."""
    q = [0.9, 0.5]
    assert decode.expected_f_beta(0, q, lam=0.0) == pytest.approx(0.05, abs=1e-9)
    assert decode.expected_f_beta(1, q, lam=0.0) == pytest.approx(0.825, abs=1e-9)
    assert decode.expected_f_beta(2, q, lam=0.0) == pytest.approx(0.728, abs=1e-3)
    assert decode.best_set(q, lam=0.0) == [0]


def test_decoder_defaults_to_empty_when_all_candidates_are_weak():
    assert decode.best_set([0.1, 0.05], lam=0.0) == []


def test_decoder_grows_the_set_when_candidates_are_strong():
    """With the measured mean t = 3.46, the common case is k >= 2, not k = 0."""
    assert len(decode.best_set([0.95, 0.93, 0.9, 0.88], lam=0.0)) >= 3


def test_poisson_binomial_pmf_sums_to_one():
    for q in ([], [0.5], [0.2, 0.8], [0.1, 0.5, 0.9, 0.3]):
        pmf = decode.poisson_binomial_pmf(q)
        assert sum(pmf) == pytest.approx(1.0)
        assert len(pmf) == len(q) + 1


def test_poisson_binomial_matches_brute_force():
    from itertools import product
    q = [0.3, 0.7, 0.45]
    pmf = decode.poisson_binomial_pmf(q)
    brute = [0.0] * (len(q) + 1)
    for combo in product([0, 1], repeat=len(q)):
        p = 1.0
        for qi, take in zip(q, combo):
            p *= qi if take else (1 - qi)
        brute[sum(combo)] += p
    for a, b in zip(pmf, brute):
        assert a == pytest.approx(b)


def test_unseen_mass_lowers_the_value_of_abstaining():
    """research.md 1.3 C: blocked-out mass reduces P(T=0), so {} is worth less."""
    assert decode.expected_f_beta(0, [0.3], lam=0.5) < decode.expected_f_beta(0, [0.3], lam=0.0)


# --- D5 N2 exclusivity ------------------------------------------------------


def test_exclusivity_makes_two_s1_compete_for_one_fragment():
    """research.md 3.4: connected components would merge S1-a and S1-b through a
    shared fragment. Exclusivity makes them compete instead."""
    pi = decode.fragment_marginals({"S2-1": {"S1-a": 0.8, "S1-b": 0.6}}, lam_null=1.0)
    row = pi["S2-1"]
    assert sum(row.values()) == pytest.approx(1.0)
    assert row["S1-a"] > row["S1-b"]
    assert decode.NULL in row


def test_null_option_absorbs_orphan_fragments():
    weak = decode.fragment_marginals({"S2-1": {"S1-a": 0.05}}, lam_null=1.0)["S2-1"]
    assert weak[decode.NULL] > weak["S1-a"]


def test_lambda_null_scales_the_abstain_weight():
    low = decode.fragment_marginals({"S2-1": {"S1-a": 0.5}}, lam_null=0.1)["S2-1"]
    high = decode.fragment_marginals({"S2-1": {"S1-a": 0.5}}, lam_null=10.0)["S2-1"]
    assert low[decode.NULL] < high[decode.NULL]


def test_final_guard_keeps_argmax_when_two_s1_claim_a_fragment():
    """D5.4: after decoding, a fragment selected by more than one S1 is resolved
    to the highest marginal."""
    pi = {"S2-1": {"S1-a": 0.7, "S1-b": 0.2, decode.NULL: 0.1}}
    cleaned = decode.enforce_exclusivity({"S1-a": ["S2-1"], "S1-b": ["S2-1"]}, pi)
    assert cleaned["S1-a"] == ["S2-1"]
    assert cleaned["S1-b"] == []


def test_incremental_best_k_matches_exact_recomputation():
    """Regression: best_k sweeps k incrementally for speed. An earlier version
    deconvolved candidates out of the full pmf by forward substitution, which is
    numerically unstable -- a pmf that should sum to 1 summed to -12531 on a
    12-candidate row, and the decoder silently returned the wrong k."""
    import random
    rng = random.Random(7)
    for _ in range(600):
        m = rng.randint(1, 14)
        q = [rng.random() for _ in range(m)]
        lam = rng.choice([0.0, 0.1, 0.5, 1.0, 2.0])
        got_k, got_score = decode.best_k(q, lam)
        ranked = sorted(q, reverse=True)
        exact = [decode.expected_f_beta(k, ranked, lam) for k in range(m + 1)]
        want_k = max(range(m + 1), key=lambda k: exact[k])
        assert got_k == want_k, (q, lam, got_k, want_k)
        assert got_score == pytest.approx(exact[want_k], abs=1e-12)


def test_suffix_pmfs_stay_normalised():
    """Every intermediate pmf must remain a distribution."""
    import random
    rng = random.Random(11)
    for _ in range(200):
        q = [rng.random() for _ in range(rng.randint(1, 14))]
        pmf = decode.poisson_binomial_pmf(q)
        assert sum(pmf) == pytest.approx(1.0, abs=1e-9)
        assert all(v >= -1e-12 for v in pmf)


def test_decoder_handles_degenerate_probabilities():
    assert decode.best_k([0.0, 0.0], lam=1.0)[0] == 0
    assert decode.best_k([1e-9], lam=0.0)[0] == 0
    k, _ = decode.best_k([1.0, 0.999, 0.5], lam=0.5)
    assert k >= 1
