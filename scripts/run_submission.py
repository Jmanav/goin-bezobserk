"""Full test-split submission run, chunked and resumable.

The full run is ~11-15 h end to end, which is longer than a Colab session and
longer than most people want to babysit. This processes the fragment stream in
chunks and checkpoints after each one, so a disconnect costs one chunk rather
than the whole run: re-running picks up where it stopped.

The S1 index is NOT sampled. io_rules.md 5.1 requires a row for every test S1
entity, and the index is what the model was fitted against, so only the fragment
stream is chunked.

    BER_SPLIT=test BER_FRAG_CHUNK=500000 python scripts/run_submission.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber import (blocking, blocking_report, decode, dense,  # noqa: E402
                 features as featlib, matcher, run_audit, submission)

DATA_ROOT = os.environ.get(
    "BER_DATA_ROOT",
    "/content/drive/MyDrive/6ab10eb3b23ba_student_resource/student_resource/dataset",
)
SPLIT = os.environ.get("BER_SPLIT", "test")
SPLIT_DIR = os.environ.get("BER_SPLIT_DIR", "")
K = int(os.environ.get("BER_K", "15"))
FRAG_CHUNK = int(os.environ.get("BER_FRAG_CHUNK", "500000"))
MODEL_DIR = Path(os.environ.get("BER_MODEL_DIR", "models"))
OUT_DIR = Path(os.environ.get("BER_OUT", "output"))
CKPT_DIR = Path(os.environ.get("BER_CKPT", "checkpoints"))
USE_DENSE = os.environ.get("BER_USE_DENSE", "1") == "1"
MODEL_KEY = os.environ.get("BER_MODEL", "multilingual-e5-small")
LAM_NULL = float(os.environ.get("BER_LAM_NULL", "1.0"))


def _find_test_s1(data_dir):
    """The test S1 file, whatever prefix it carries."""
    for pattern in ("test_source1.tsv", "*source1*.tsv"):
        hits = sorted(Path(data_dir).glob(pattern))
        if hits:
            return hits[0]
    raise SystemExit(f"no source1 TSV in {data_dir}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # SPLIT_DIR points straight at a split directory, for the case where the
    # splits live in separate dataset mounts rather than under one root.
    if SPLIT_DIR:
        data_dir = Path(SPLIT_DIR)
        root = data_dir.parent
        if not data_dir.is_dir():
            raise SystemExit(f"BER_SPLIT_DIR does not exist: {data_dir}")
    else:
        root = run_audit.find_dataset_root(DATA_ROOT, require=(SPLIT,))
        if root is None:
            raise SystemExit(
                f"no dataset/{SPLIT} under {DATA_ROOT}. A submission needs the "
                "test split: every one of the test S1 rows must appear in the "
                "output (io_rules.md 5.1). If the splits are in separate "
                "dataset mounts, set BER_SPLIT_DIR to the test directory."
            )
        data_dir = root / SPLIT
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    model = matcher.load_scorer(MODEL_DIR / "pair_scorer.pkl")
    idf, ncc, adc = matcher.load_index_stats(MODEL_DIR / "index_stats.pkl")
    log(f"loaded scorer ({model.backend}) and index stats")

    frames, _ = run_audit.load_dataset(data_dir)
    s1 = frames["S1"]
    s1_ids = [str(x).strip() for x in s1["entity_id"]]
    frags_all = pd.concat([frames["S2"], frames["S3"]], ignore_index=True)
    frames.clear()
    log(f"S1={len(s1):,}  fragments={len(frags_all):,}")

    # The index is built once and reused across every chunk; it is the expensive
    # part and does not depend on which fragments are being queried.
    t0 = time.time()
    cfg = dense.DenseConfig(model_key=MODEL_KEY, top_n=50)
    harness = blocking_report.BlockingHarness(k=K, use_dense=USE_DENSE,
                                              dense_config=cfg).fit(s1)
    for note in harness.notes:
        log(f"NOTE: {note}")
    log(f"index built in {time.time()-t0:.0f}s")

    n_chunks = (len(frags_all) + FRAG_CHUNK - 1) // FRAG_CHUNK
    predictions, candidates = {}, {}

    for c in range(n_chunks):
        ckpt = CKPT_DIR / f"chunk_{c:04d}.pkl"
        if ckpt.exists():
            with open(ckpt, "rb") as fh:
                part = pickle.load(fh)
            log(f"chunk {c+1}/{n_chunks}: loaded from checkpoint")
        else:
            t0 = time.time()
            chunk = frags_all.iloc[c * FRAG_CHUNK:(c + 1) * FRAG_CHUNK]
            fused, channels, always = harness.query(chunk)
            fs = {}
            blocking.reciprocal_rank_fusion(channels, k=K, always_keep=always,
                                            fused_scores=fs)
            marg = matcher.score_candidates(
                model, fused, fs, harness.fragment_records, harness.s1_records,
                idf, ncc, adc)
            part = {"fused": fused, "marginals": marg}
            with open(ckpt, "wb") as fh:
                pickle.dump(part, fh, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"chunk {c+1}/{n_chunks}: {len(chunk):,} fragments in "
                f"{time.time()-t0:.0f}s -> {ckpt}")
            del chunk, channels, fs

        for frag_id, cands in part["fused"].items():
            for s1_id in cands:
                candidates.setdefault(s1_id, set()).add(frag_id)
        predictions.setdefault("_marginals", {}).update(part["marginals"])
        del part

    # Decode once over the whole stream: a fragment's exclusivity guard has to
    # see every S1 that claimed it, so this cannot be done per chunk.
    log("decoding")
    t0 = time.time()
    marginals = predictions.pop("_marginals")
    fused_all = {f: list(row) for f, row in marginals.items()}
    preds = decode.pipeline_predictions(fused_all, s1_ids, lam_null=LAM_NULL,
                                        marginals=marginals)
    log(f"decoded in {time.time()-t0:.0f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_path = submission.write_matching_results(
        OUT_DIR / "matching_results.tsv", s1_ids, preds)
    c_path = submission.write_candidate_pairs(
        OUT_DIR / "candidate_pairs.tsv", s1_ids,
        {k: sorted(v) for k, v in candidates.items()})
    log(f"wrote {m_path} and {c_path}")

    report = submission.validate(
        m_path, c_path, s1_ids=s1_ids,
        fragment_ids=set(frags_all["entity_id"]),
        test_s1_path=_find_test_s1(data_dir), is_submission=True)
    print(json.dumps({"ok": report.ok, "n_rows": report.n_rows,
                      "n_predicted_ids": report.n_predicted_ids,
                      "problems": report.problems[:10]}, indent=2))

    official = Path(os.environ.get(
        "BER_VALIDATOR", root.parent / "utils" / "validate_submission.py"))
    if official.exists():
        ok, output = submission.run_official_validator(
            official, m_path, c_path, data_dir)
        print(f"\norganiser validator: {'PASS' if ok else 'FAIL'}")
        print(output[:3000])
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
