"""Owner A audit runner.

Prints the research.md section 10 audit checklist over a dataset directory, or
over freshly generated proxy data. Report only: nothing here mutates or filters
input.

    python -m ber.run_audit --proxy
    python -m ber.run_audit --data-dir dataset/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import audit, proxy_data, safe_io
from .normalise import normalise_frame, tokenise
from .placeholders import discover_placeholders


def _rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _show(label, value):
    print(f"\n-- {label}")
    print(json.dumps(value, indent=2, default=str, ensure_ascii=False))


def load_dataset(data_dir):
    """Load S1/S2/S3 from a directory, running every section 1 sanity check."""
    data_dir = Path(data_dir)
    frames = {}
    reports = {}
    for source, n in (("S1", 1), ("S2", 2), ("S3", 3)):
        candidates = [
            data_dir / f"source{n}.tsv",
            data_dir / f"{source.lower()}.tsv",
            *sorted(data_dir.glob(f"*source{n}*.tsv")),
            *sorted(data_dir.glob(f"*{source.lower()}*.tsv")),
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"no {source} TSV in {data_dir} (looked for source{n}.tsv, "
                f"{source.lower()}.tsv and *source{n}*.tsv)"
            )
        df, report = safe_io.load_source(path)
        frames[source] = df
        reports[source] = report
    return frames, reports


def find_dataset_root(*hints, require=("train", "test")):
    """Locate the organiser's dataset dir (the one containing train/ and test/).

    Accepts either the student_resource root or the dataset dir itself. Returns
    None when nothing matches, so callers can fall back to proxy data.

    `require` names the splits that must be present. It defaults to both, but a
    partial upload -- train present, test still uploading -- is a normal state
    to work in: model training needs only train/, so requiring both would block
    hours of usable work on a file transfer.
    """
    candidates = []
    for hint in hints:
        if not hint:
            continue
        base = Path(hint)
        candidates += [base, base / "dataset", base / "student_resource" / "dataset"]
    for base in candidates:
        if all((base / split).is_dir() for split in require):
            return base
    return None


def load_ground_truth(data_dir):
    """Load *ground_truth*.tsv from a split dir, or None when absent."""
    matches = sorted(Path(data_dir).glob("*ground_truth*.tsv"))
    if not matches:
        return None
    return safe_io.read_ground_truth(matches[0])


def build_proxy(out_dir, n_entities=400, seed=20260925, singleton_rate=0.40):
    """Generate proxy data and write it as TSV (A5)."""
    cfg = proxy_data.ProxyConfig(
        n_entities=n_entities, seed=seed, singleton_rate=singleton_rate
    )
    s1, s2, s3, truth = proxy_data.generate(cfg)
    split = proxy_data.split_train_test(s1, s2, s3, truth, cfg)

    out_dir = Path(out_dir)
    paths = {}
    for phase in ("train", "test"):
        p_s1, p_s2, p_s3, p_truth = split[phase]
        phase_dir = out_dir / phase
        paths[phase] = {
            "S1": proxy_data.write_tsv(p_s1, phase_dir / "source1.tsv"),
            "S2": proxy_data.write_tsv(p_s2, phase_dir / "source2.tsv"),
            "S3": proxy_data.write_tsv(p_s3, phase_dir / "source3.tsv"),
            "truth": proxy_data.write_truth(p_truth, phase_dir / "ground_truth.tsv"),
        }
    return split, paths


def run(frames, truth=None, train_frames=None):
    """Print the research.md section 10 audit checklist."""

    _rule("audit #10  scale (rows per source)")
    _show("rows", audit.scale_report(frames))

    _rule("audit #1  parsing / literal NA-like values")
    print("Literal 'NA'/'None'/'null'/'' survive only because read_source turns")
    print("NA handling off (io_rules.md section 1). Seeing them here is proof.")
    _show("literal na-like counts", audit.literal_na_counts(frames))

    _rule("audit #2  namespace  [A1.4]")
    problems = safe_io.check_namespace(frames, raise_on_fail=False)
    _show("namespace problems (expect [])", problems)

    _rule("audit #7  country mix  [A1.5 / A0.5 VERIFY]")
    print("Exact string values are what settles 'France' vs 'FR' vs 'france'.")
    _show("country report", audit.country_report(frames))
    if train_frames:
        _show("labels in test but not train", sorted(audit.unseen_test_labels(train_frames, frames)))

    _rule("audit #9 / A0.6  script census")
    _show("script census", audit.script_census(frames))

    _rule("io_rules section 4 / A4.2  placeholder discovery")
    _show("discovered placeholders (DF > 1%, len <= 3)", audit.placeholder_report(frames))

    _rule("A2.3  S1 token IDF")
    s1_norm = normalise_frame(frames["S1"])
    idf, df_counts = audit.token_idf(s1_norm["name_tokens"])
    top_df = audit.high_df_tokens(df_counts, len(s1_norm))
    _show("most frequent name tokens (down-weight by IDF, do not delete)", top_df[:15])
    rare = sorted(idf.items(), key=lambda kv: -kv[1])[:10]
    _show("rarest tokens (highest IDF -> best blocking keys)", rare)

    if truth is not None:
        _rule("audit #3  structure  [A0.2 / A0.3  VERIFY  Q3 / Q4]")
        multi = audit.fragments_in_multiple_s1(truth)
        _show("fragments owned by >1 S1 (expect {})", multi)
        print(
            "\nQ3 asymmetric framing: "
            + ("HOLDS" if not multi else "VIOLATED -- research.md 1.1 framing is wrong")
        )

        _, maxima = audit.per_vendor_match_counts(truth)
        _show("max matched fragments per S1, per vendor", maxima)
        one_to_one = all(v <= 1 for v in maxima.values())
        print(
            "\nQ4 within-vendor 1:1: "
            + (
                "HOLDS -> Owner D may enable the N2 bipartite branch (D5.3)"
                if one_to_one
                else "DOES NOT HOLD -> N2 bipartite branch stays off (D5.3)"
            )
        )

        _rule("audit #4 / #5  singleton rate and cardinality  [A0.4  VERIFY  Q5]")
        card = audit.cardinality_report(truth)
        _show("cardinality", card)
        print(
            f"\nSingleton rate {card['singleton_rate']:.1%}: research.md 1.2 notes "
            "singleton decisions alone can swing this much of the score."
        )

        _rule("audit #6  orphans  [sets lambda_null for N2]")
        frag_frames = {k: v for k, v in frames.items() if k != "S1"}
        _show("orphans per vendor", audit.orphan_report(truth, frag_frames))

    _rule("done")
    print("Reports only -- nothing above filtered or mutated the data.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Owner A data audit (research.md section 10)")
    parser.add_argument("--data-dir", help="directory holding source1/2/3 TSVs")
    parser.add_argument("--proxy", action="store_true", help="generate and audit proxy data")
    parser.add_argument("--proxy-dir", default="data/interim/proxy")
    parser.add_argument("--n-entities", type=int, default=400)
    parser.add_argument("--singleton-rate", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)

    if not args.proxy and not args.data_dir:
        parser.error("pass --proxy or --data-dir")

    if args.proxy:
        split, paths = build_proxy(
            args.proxy_dir,
            n_entities=args.n_entities,
            seed=args.seed,
            singleton_rate=args.singleton_rate,
        )
        print(f"proxy data written under {args.proxy_dir}")

        train_frames, _ = load_dataset(Path(args.proxy_dir) / "train")
        test_frames, _ = load_dataset(Path(args.proxy_dir) / "test")
        truth = split["test"][3]

        _rule("PROXY TEST SPLIT (France present)")
        run(test_frames, truth=truth, train_frames=train_frames)
    else:
        root = find_dataset_root(args.data_dir)
        if root is None:
            # a bare split dir (no train/ + test/ children) is still auditable
            frames, _ = load_dataset(args.data_dir)
            run(frames, truth=load_ground_truth(args.data_dir))
        else:
            train_frames, _ = load_dataset(root / "train")
            test_frames, _ = load_dataset(root / "test")
            _rule(f"TRAIN SPLIT  ({root / 'train'})")
            run(train_frames, truth=load_ground_truth(root / "train"))
            _rule(f"TEST SPLIT  ({root / 'test'})")
            run(test_frames, truth=load_ground_truth(root / "test"), train_frames=train_frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def audit_json(frames, truth=None, s1_df=None, train_frames=None, gt_path=None):
    """Return the section 10 audit as a plain JSON-able dict.

    Same probes as run(), but returned rather than printed, so the numbers can
    be copied out of a notebook. Report only: nothing here mutates or filters.
    """
    out = {
        "scale": audit.scale_report(frames),
        "literal_na_counts": audit.literal_na_counts(frames),
        "namespace_problems": safe_io.check_namespace(frames, raise_on_fail=False),
        "country_report": audit.country_report(frames),
        "script_census": audit.script_census(frames),
        "placeholders": audit.placeholder_report(frames),
    }
    if train_frames is not None:
        out["labels_in_test_not_train"] = sorted(
            audit.unseen_test_labels(train_frames, frames)
        )
    if gt_path is not None:
        out["ground_truth_shape"] = audit.ground_truth_shape(gt_path)

    if truth is not None:
        multi = audit.fragments_in_multiple_s1(truth)
        _, maxima = audit.per_vendor_match_counts(truth)
        out["structure"] = {
            "n_fragments_in_multiple_s1": len(multi),
            "example_violations": dict(list(multi.items())[:5]),
            "q3_asymmetric_framing_holds": not multi,
            "max_matched_per_s1_per_vendor": maxima,
            "q4_within_vendor_1to1_holds": all(v <= 1 for v in maxima.values()),
        }
        out["cardinality"] = audit.cardinality_report(truth)
        out["orphans"] = audit.orphan_report(
            truth, {k: v for k, v in frames.items() if k != "S1"}
        )
        if s1_df is not None:
            out["cardinality_by_country"] = audit.cardinality_by_country(truth, s1_df)
    return out
