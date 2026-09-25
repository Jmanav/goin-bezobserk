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

from ber import (blocking, blocking_report, decode, dense,  # noqa: E402
                 run_audit, scorer, submission)

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

    # --- decode -------------------------------------------------------------
    t0 = time.time()
    predictions = decode.pipeline_predictions(fused, s1_ids, lam_null=LAM_NULL)
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
    report = submission.validate(m_path, c_path, s1_ids=s1_ids,
                                 fragment_ids=fragment_ids)
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
        show("local macro-F0.5 (PLACEHOLDER SCORING -- no Owner C model)",
             rep.to_dict())
        ceiling = blocking_report.entity_recall_ceiling(candidates_by_s1, truth)
        show("blocking ceiling for comparison", ceiling)
        print("\nThe gap between this score and the ceiling is what Owner C's")
        print("model is worth. The score itself is a floor, not a result.")

    print("\n" + "=" * 72)
    print(f"Gate 0 criterion 1 (end-to-end run): {'PASS' if report.ok else 'FAIL'}")
    print(f"Gate 0 criterion 2 (validator PASS): {'PASS' if report.ok else 'FAIL'}")
    print("=" * 72)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
