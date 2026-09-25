"""First real PC@K measurement for the blocking harness (Owner B, B7.2/B7.3).

Runs the four channels over a sample of the real training data and reports pair
completeness, the per-entity recall ceiling, the PC-vs-K curve and per-country /
per-vendor slices.

Sampling is S1-first: a sample of S1 rows is drawn, then exactly their true
fragments plus a pool of distractors. Sampling fragments independently would
leave truth referring to S1 rows that were never indexed, so PC would be
measured against a denominator the harness was never given a chance to retrieve.

Scale up N_S1 gradually. The char-TF-IDF channel builds a fragment x S1 sparse
product in row chunks, and research.md section 6's budgets assume 8-9M rows
against a real 24M, so time the small runs before trusting the big ones.
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

from ber import blocking, blocking_report, dense, run_audit  # noqa: E402

DATA_ROOT = os.environ.get(
    "BER_DATA_ROOT",
    "/content/drive/MyDrive/6ab10eb3b23ba_student_resource/student_resource/dataset",
)
N_S1 = int(os.environ.get("BER_N_S1", "20000"))
N_DISTRACTOR_FRAGS = int(os.environ.get("BER_N_DISTRACTORS", "20000"))
K = int(os.environ.get("BER_K", "25"))
USE_DENSE = os.environ.get("BER_USE_DENSE", "1") == "1"
MODEL_KEY = os.environ.get("BER_MODEL", "multilingual-e5-small")
SEED = 42


def show(title, obj):
    bar = "=" * 72
    print("\n" + bar + "\n" + title + "\n" + bar)
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main():
    root = run_audit.find_dataset_root(DATA_ROOT)
    if root is None:
        raise SystemExit(f"no dataset/train+test under {DATA_ROOT}")
    train = root / "train"
    print("train dir:", train)
    print("dense stack available:", dense.available())
    show("model record (io_rules.md section 9)",
         dense.describe(dense.DenseConfig(model_key=MODEL_KEY)))

    truth_all = run_audit.load_ground_truth(train)
    frames, _ = run_audit.load_dataset(train)
    s1_all = frames["S1"]
    print(f"\nfull train: S1={len(s1_all):,} S2={len(frames['S2']):,} "
          f"S3={len(frames['S3']):,} truth={len(truth_all):,}")

    # --- S1-first sample ---------------------------------------------------
    rng = random.Random(SEED)
    s1_ids = rng.sample(list(truth_all), min(N_S1, len(truth_all)))
    s1_set = set(s1_ids)
    truth = {k: truth_all[k] for k in s1_ids}
    wanted_frags = {f for ids in truth.values() for f in ids}
    print(f"sampled S1={len(truth):,}  their true fragments={len(wanted_frags):,}")

    s1 = s1_all[s1_all["entity_id"].isin(s1_set)].reset_index(drop=True)
    del s1_all

    # True fragments, plus distractors so precision pressure is realistic: with
    # only true fragments present every channel looks far better than it is.
    parts, n_distract = [], 0
    for key in ("S2", "S3"):
        df = frames[key]
        keep = df[df["entity_id"].isin(wanted_frags)]
        parts.append(keep)
        pool = df[~df["entity_id"].isin(wanted_frags)]
        take = min(N_DISTRACTOR_FRAGS // 2, len(pool))
        if take:
            parts.append(pool.sample(n=take, random_state=SEED))
            n_distract += take
        del df, keep, pool
    frames.clear()

    frags = pd.concat(parts, ignore_index=True)
    del parts
    print(f"fragments queried={len(frags):,} (of which distractors={n_distract:,})")

    # --- run ----------------------------------------------------------------
    cfg = dense.DenseConfig(model_key=MODEL_KEY, top_n=50)
    harness = blocking_report.BlockingHarness(k=K, use_dense=USE_DENSE,
                                              dense_config=cfg)
    t0 = time.time()
    harness.fit(s1)
    fit_s = time.time() - t0
    for note in harness.notes:
        print("NOTE:", note)

    t0 = time.time()
    fused, channels, always = harness.query(frags)
    query_s = time.time() - t0
    by_s1 = blocking.invert_to_s1(fused)

    show("timing and per-channel reach", {
        "fit_seconds": round(fit_s, 1),
        "query_seconds": round(query_s, 1),
        "seconds_per_1k_fragments": round(query_s / max(len(frags), 1) * 1000, 2),
        # Cost scales with S1 x fragments, not fragments alone: every fragment is
        # scored against the whole S1 index. A linear-in-fragments projection
        # understates the full run by ~6x at these sample sizes.
        "seconds_per_fragment_x_s1_unit":
            query_s / max(len(frags) * len(s1), 1),
        "projected_hours_full_test_1_73M_s1_x_9_97M_frags":
            round(query_s / max(len(frags) * len(s1), 1)
                  * 9_969_589 * 1_732_544 / 3600, 1),
        # Per-stage wall clock. Two earlier diagnoses were wrong because they
        # reasoned from profiles of isolated components instead of measuring the
        # real run, so the harness now reports where its time actually goes.
        "stage_seconds": {k: round(v, 1)
                          for k, v in sorted(getattr(harness, "timings", {}).items(),
                                             key=lambda kv: -kv[1])},
        "channels": {
            ch.name: {
                "fragments_with_a_hit": sum(1 for v in ch.hits.values() if v),
                "total_hits": sum(len(v) for v in ch.hits.values()),
            } for ch in channels
        },
    })

    show("B6.1 pair completeness", blocking_report.pair_completeness(by_s1, truth))
    show("B6.2 reduction ratio",
         blocking_report.reduction_ratio(by_s1, len(s1), len(frags)))
    show("B6.5 per-entity recall ceiling (bounds the leaderboard score)",
         blocking_report.entity_recall_ceiling(by_s1, truth))
    show("B6.3 candidates per fragment",
         blocking_report.candidates_per_fragment(fused))

    print("\n" + "=" * 72 + "\nB6.3 PC-vs-K curve\n" + "=" * 72)
    curve = blocking_report.pc_vs_k(channels, truth, ks=(5, 10, 15, 20, 25, 30, 50),
                                    always_keep=always)
    for row in curve:
        print(f"  K={row['k']:<3} PC={row['pair_completeness']:.4f}  "
              f"entity-ceiling={row['entity_recall_ceiling']:.4f}  "
              f"F0.5-ceiling={row['macro_f05_ceiling']:.4f}  "
              f"pairs={row['n_candidate_pairs']:,}")

    country = {r.entity_id: str(r.country).strip().casefold()
               for r in s1.itertuples(index=False)}
    show("B6.4 slices", blocking_report.slice_report(by_s1, truth, country))

    gate = next((r for r in curve if r["k"] <= 30 and
                 r["pair_completeness"] >= 0.99), None)
    print("\n" + "=" * 72)
    print("Gate 1 target (research.md 3.2): PC >= 99.0% at K <= 30")
    print(f"  {'MET at K=' + str(gate['k']) if gate else 'NOT MET at any K <= 30'}")
    if not USE_DENSE or not dense.available():
        print("  NOTE: dense channel did not run, so this understates reachable PC")
    print("=" * 72)


if __name__ == "__main__":
    main()
