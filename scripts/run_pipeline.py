"""End-to-end pipeline: read -> normalise -> block -> decode -> write (Gate 0 #1).

Runs Owner A's reader and normaliser, Owner B's blocking harness and Owner D's
decoder and writer, emits both output files, runs the team validator and (when
ground truth is present) scores the result locally.

Scoring is a PLACEHOLDER. Owner C does not exist yet, so blocking ranks are
mapped to pseudo-probabilities by decode.rrf_to_marginals. The macro-F0.5 this
prints is therefore a floor, not a result -- the point of this script is to prove
the format path works, since io_rules.md calls format mismatches the most common
failure mode and the writer and validator have never run against real output.

    BER_SPLIT=train BER_N_S1=5000 python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.normalise import normalise_record  # noqa: E402
from ber import (blocking, blocking_report, decode, dense,  # noqa: E402
                 features as featlib, matcher, run_audit, scorer, submission)

DATA_ROOT = os.environ.get(
    "BER_DATA_ROOT",
    "/content/drive/MyDrive/6ab10eb3b23ba_student_resource/student_resource/dataset",
)
SPLIT = os.environ.get("BER_SPLIT", "train")
N_S1 = int(os.environ.get("BER_N_S1", "5000"))
N_DISTRACTORS = int(os.environ.get("BER_N_DISTRACTORS", "5000"))
K = int(os.environ.get("BER_K", "15"))
USE_DENSE = os.environ.get("BER_USE_DENSE", "1") == "1"
MODEL_KEY = os.environ.get("BER_MODEL", "multilingual-e5-small")
OUT_DIR = Path(os.environ.get("BER_OUT", "output"))
LAM_NULL = float(os.environ.get("BER_LAM_NULL", "1.0"))
TRAIN_MODEL = os.environ.get("BER_TRAIN_MODEL", "1") == "1"
MODEL_DIR = Path(os.environ.get("BER_MODEL_DIR", "models"))
SEED = 42


def show(title, obj):
    bar = "=" * 72
    print("\n" + bar + "\n" + title + "\n" + bar)
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main():
    root = run_audit.find_dataset_root(DATA_ROOT)
    if root is None:
        raise SystemExit(f"no dataset/train+test under {DATA_ROOT}")
    data_dir = root / SPLIT
    print(f"split: {data_dir}")

    frames, reports = run_audit.load_dataset(data_dir)
    truth_all = run_audit.load_ground_truth(data_dir)
    for source, rep in reports.items():
        print(f"  {source}: rows={rep.n_rows:,} nan={rep.n_nan} ok={rep.ok}")

    # --- sample, S1-first so truth only references indexed S1 rows ----------
    rng = random.Random(SEED)
    if truth_all:
        s1_ids = rng.sample(list(truth_all), min(N_S1, len(truth_all)))
        truth = {k: truth_all[k] for k in s1_ids}
        wanted = {f for ids in truth.values() for f in ids}
    else:
        s1_ids = rng.sample(list(frames["S1"]["entity_id"]),
                            min(N_S1, len(frames["S1"])))
        truth, wanted = None, set()
    s1_set = set(s1_ids)

    s1 = frames["S1"][frames["S1"]["entity_id"].isin(s1_set)].reset_index(drop=True)
    parts = []
    for key in ("S2", "S3"):
        df = frames[key]
        if wanted:
            parts.append(df[df["entity_id"].isin(wanted)])
            pool = df[~df["entity_id"].isin(wanted)]
        else:
            pool = df
        take = min(N_DISTRACTORS // 2, len(pool))
        if take:
            parts.append(pool.sample(n=take, random_state=SEED))
        del df, pool
    frames.clear()
    frags = pd.concat(parts, ignore_index=True)
    del parts
    print(f"  sampled S1={len(s1):,}  fragments={len(frags):,}")

    # --- block --------------------------------------------------------------
    cfg = dense.DenseConfig(model_key=MODEL_KEY, top_n=50)
    harness = blocking_report.BlockingHarness(k=K, use_dense=USE_DENSE,
                                              dense_config=cfg)
    t0 = time.time()
    harness.fit(s1)
    for note in harness.notes:
        print("NOTE:", note)
    fused, channels, always = harness.query(frags)
    block_s = time.time() - t0
    print(f"  blocking: {block_s:.1f}s")

    # --- score (Owner C) ----------------------------------------------------
    # Without a trained model the decoder falls back to decode.rrf_to_marginals,
    # which is a rank heuristic, not a scorer. Training needs ground truth, so
    # this path is available on train and skipped on test.
    marginals = None
    model_path = MODEL_DIR / "pair_scorer.pkl"
    stats_path = MODEL_DIR / "index_stats.pkl"

    if not truth and model_path.exists():
        # Test split: no ground truth to train on, so reuse the scorer fitted on
        # train. The S1-derived statistics travel with it, because recomputing
        # idf and the chain counts over test S1 would shift the feature
        # distribution the model was fitted on.
        t0 = time.time()
        model = matcher.load_scorer(model_path)
        idf, ncc, adc = matcher.load_index_stats(stats_path)
        fs = {}
        blocking.reciprocal_rank_fusion(channels, k=K, always_keep=always,
                                        fused_scores=fs)
        s1_recs = {r.entity_id: normalise_record(r.business_name,
                                                 r.business_address, r.country)
                   for r in s1.itertuples(index=False)}
        fr_recs = {r.entity_id: normalise_record(r.business_name,
                                                 r.business_address, r.country)
                   for r in frags.itertuples(index=False)}
        marginals = matcher.score_candidates(
            model, fused, fs, fr_recs, s1_recs, idf, ncc, adc)
        print(f"  scoring: {time.time()-t0:.1f}s  loaded {model_path} "
              f"backend={model.backend}")
    elif not truth:
        print(f"  scoring: NO MODEL at {model_path} -- falling back to the rank")
        print("    heuristic, which scores far below a trained model. Run the")
        print("    train split first to fit and persist one.")

    if truth and TRAIN_MODEL:
        t0 = time.time()
        fs = {}
        blocking.reciprocal_rank_fusion(channels, k=K, always_keep=always,
                                        fused_scores=fs)
        s1_recs = {r.entity_id: normalise_record(r.business_name,
                                                 r.business_address, r.country)
                   for r in s1.itertuples(index=False)}
        fr_recs = {r.entity_id: normalise_record(r.business_name,
                                                 r.business_address, r.country)
                   for r in frags.itertuples(index=False)}
        idf = featlib.token_idf([r.name_tokens for r in s1_recs.values()])
        ncc, adc = featlib.corpus_counts(s1_recs)

        X, y, groups = matcher.build_training_pairs(
            fused, fs, fr_recs, s1_recs, truth, idf, ncc, adc)
        if len(X) and len(set(y)) > 1:
            # Hold out by fragment so a fragment's candidates never straddle the
            # split (research.md 8.3a: group by entity, not by row).
            uniq = sorted(set(groups))
            train_frags = set(uniq[:int(len(uniq) * 0.7)])
            tr = [i for i, g in enumerate(groups) if g in train_frags]
            model = matcher.PairScorer().fit(X[tr], y[tr])
            marginals = matcher.score_candidates(
                model, fused, fs, fr_recs, s1_recs, idf, ncc, adc)
            matcher.save_scorer(model, model_path)
            matcher.save_index_stats(stats_path, idf, ncc, adc)
            print(f"  scoring: {time.time()-t0:.1f}s  backend={model.backend} "
                  f"pairs={len(X):,} positives={int(y.sum()):,}")
            print(f"  saved {model_path} and {stats_path} for the test run")
            imp = model.feature_importance(8)
            if imp:
                print("  top features:", ", ".join(imp))
        else:
            print("  scoring: skipped (insufficient labelled pairs)")

    # --- decode -------------------------------------------------------------
    t0 = time.time()
    predictions = decode.pipeline_predictions(fused, s1_ids, lam_null=LAM_NULL,
                                              marginals=marginals)
    decode_s = time.time() - t0
    n_pred = sum(len(v) for v in predictions.values())
    print(f"  decode: {decode_s:.1f}s  predicted ids={n_pred:,}")

    # --- write both files ---------------------------------------------------
    candidates_by_s1 = blocking.invert_to_s1(fused)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_path = submission.write_matching_results(
        OUT_DIR / "matching_results.tsv", s1_ids, predictions)
    c_path = submission.write_candidate_pairs(
        OUT_DIR / "candidate_pairs.tsv", s1_ids, candidates_by_s1)
    print(f"\nwrote {m_path}\nwrote {c_path}")

    # --- validate -----------------------------------------------------------
    fragment_ids = set(frags["entity_id"])
    test_s1 = root / "test" / "test_source1.tsv"
    submitting = SPLIT == "test" and N_S1 >= 1_000_000
    report = submission.validate(
        m_path, c_path, s1_ids=s1_ids, fragment_ids=fragment_ids,
        test_s1_path=test_s1 if (submitting and test_s1.exists()) else None,
        is_submission=submitting and test_s1.exists(),
    )
    show("team validator (D4.2, D4.3)", {
        "ok": report.ok, "n_rows": report.n_rows,
        "n_predicted_ids": report.n_predicted_ids,
        "problems": report.problems[:10],
        "n_problems": len(report.problems),
    })

    official = root.parent / "utils" / "validate_submission.py"
    if official.exists():
        ok, output = submission.run_official_validator(
            official, m_path, c_path, root / "test")
        print(f"\norganiser validator ({official}): {'PASS' if ok else 'FAIL'}")
        print(output[:2000])
    else:
        print(f"\norganiser validator not found at {official} -- team checks only")

    # --- score --------------------------------------------------------------
    if truth:
        country = {r.entity_id: str(r.country).strip().casefold()
                   for r in s1.itertuples(index=False)}
        rep = scorer.macro_f_beta(truth, predictions, slices={
            "t": scorer.cardinality_slice(truth), "country": country})
        label = ("Owner C model" if marginals else "PLACEHOLDER rank heuristic")
        show(f"local macro-F0.5 ({label})",
             rep.to_dict())
        ceiling = blocking_report.entity_recall_ceiling(candidates_by_s1, truth)
        show("blocking ceiling for comparison", ceiling)
        print("\nThe gap between this score and the ceiling is what Owner C's")
        print("model is worth. The score itself is a floor, not a result.")

    # A sampled run cannot pass Gate 0 criterion 2. The team validator checks
    # only the ids it was handed, so on a 20k train sample it reported PASS while
    # the organiser's validator failed the same files for missing all 1,732,544
    # test entities. Only a full test-split run produces a submittable file.
    is_full_test = SPLIT == "test" and N_S1 >= 1_000_000
    print("\n" + "=" * 72)
    print(f"Gate 0 criterion 1 (end-to-end run): {'PASS' if report.ok else 'FAIL'}")
    if is_full_test:
        print(f"Gate 0 criterion 2 (validator PASS): {'PASS' if report.ok else 'FAIL'}")
    else:
        print(f"Gate 0 criterion 2 (validator PASS): NOT TESTED -- sampled "
              f"{SPLIT} run, not a submittable file.")
        print("  For a real submission every test S1 row must be present:")
        print("    BER_SPLIT=test BER_N_S1=2000000 BER_TRAIN_MODEL=0")
        print("  (the test split ships no ground truth, so the model must be")
        print("   trained on train and reused -- see the model-reuse note)")
    print("=" * 72)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
