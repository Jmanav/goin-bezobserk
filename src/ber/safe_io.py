"""Safe TSV reading and input sanity checks.

Flags and checks are fixed by docs/io_rules.md section 1 and must not be
re-derived. Owner A, Sprint 0 tasks A1.1-A1.5.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
N_FIELDS = len(EXPECTED_COLUMNS)

SOURCE_PREFIXES = ("S1-", "S2-", "S3-")


class SanityCheckError(AssertionError):
    """A read-time sanity check failed.

    io_rules.md section 8 lists these as silent killers, so they raise rather
    than warn (A1.3).
    """


def read_source(path):
    """Read a source TSV exactly per io_rules.md section 1.

    Every flag below is load-bearing; see io_rules.md section 1 for why each
    one matters. Do not add or drop keyword arguments here.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="warn",
    )


def _raw_lines(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read().splitlines()


@dataclass
class SanityReport:
    path: str
    n_rows: int
    n_raw_lines: int
    bad_field_counts: dict = field(default_factory=dict)
    n_nan: int = 0

    @property
    def ok(self):
        return (
            self.n_rows == self.n_raw_lines - 1
            and not self.bad_field_counts
            and self.n_nan == 0
        )


def check_source(df, path, raise_on_fail=True):
    """Run the three io_rules.md section 1 sanity checks (A1.2).

    1. row count equals raw line count minus the header
    2. every raw line has exactly 4 tab-separated fields, counted on the raw
       line before trusting the parse
    3. no unexpected NaN anywhere
    """
    lines = _raw_lines(path)

    bad = {}
    for lineno, line in enumerate(lines, start=1):
        n = line.count("\t") + 1
        if n != N_FIELDS:
            bad[lineno] = n

    n_nan = int(df.isna().sum().sum())

    report = SanityReport(
        path=str(path),
        n_rows=len(df),
        n_raw_lines=len(lines),
        bad_field_counts=bad,
        n_nan=n_nan,
    )

    if raise_on_fail and not report.ok:
        raise SanityCheckError(_format_failure(report))
    return report


def _format_failure(report):
    problems = []
    if report.n_rows != report.n_raw_lines - 1:
        problems.append(
            f"check 1 (row count): parsed {report.n_rows} rows but file has "
            f"{report.n_raw_lines} lines (expected {report.n_raw_lines - 1} rows)"
        )
    if report.bad_field_counts:
        sample = list(report.bad_field_counts.items())[:20]
        detail = ", ".join(f"line {ln}: {n} fields" for ln, n in sample)
        more = "" if len(report.bad_field_counts) <= 20 else f" (+{len(report.bad_field_counts) - 20} more)"
        problems.append(
            f"check 2 (field count): {len(report.bad_field_counts)} line(s) not "
            f"{N_FIELDS} fields -> {detail}{more}"
        )
    if report.n_nan:
        problems.append(f"check 3 (NaN): found {report.n_nan} NaN cell(s)")
    return f"{report.path}\n  " + "\n  ".join(problems)


def load_source(path, raise_on_fail=True):
    """Read a source file and run the sanity checks in one call."""
    df = read_source(path)
    report = check_source(df, path, raise_on_fail=raise_on_fail)
    return df, report


def expected_prefix(source):
    return f"{source}-"


def check_namespace(frames, raise_on_fail=True):
    """Assert ID prefixes and cross-source disjointness (A1.4).

    `frames` maps source label ("S1"/"S2"/"S3") to its DataFrame. Prefixes are
    never stripped: io_rules.md section 2 makes the prefix the namespace, since
    bare numbers collide across sources.
    """
    problems = []
    id_sets = {}

    for source, df in frames.items():
        prefix = expected_prefix(source)
        ids = df["entity_id"]
        wrong = ids[~ids.str.startswith(prefix)]
        if len(wrong):
            problems.append(
                f"{source}: {len(wrong)} id(s) lack prefix {prefix!r}, "
                f"e.g. {wrong.head(5).tolist()}"
            )
        dupes = ids[ids.duplicated()]
        if len(dupes):
            problems.append(
                f"{source}: {len(dupes)} duplicate entity_id(s), "
                f"e.g. {dupes.head(5).tolist()}"
            )
        id_sets[source] = set(ids)

    labels = sorted(id_sets)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            overlap = id_sets[a] & id_sets[b]
            if overlap:
                problems.append(
                    f"{a} and {b} share {len(overlap)} id(s), "
                    f"e.g. {sorted(overlap)[:5]}"
                )

    if problems and raise_on_fail:
        raise SanityCheckError("namespace check failed\n  " + "\n  ".join(problems))
    return problems


def read_ground_truth(path):
    """Read a ground-truth TSV into {s1_id: set(fragment_ids)}.

    The organiser's on-disk shape is not fixed by io_rules.md, so it is detected
    from the header rather than assumed: either the section 5.1 shape (one row
    per S1 with a comma-separated list) or one pair per row. An unrecognised
    header raises instead of silently producing empty truth.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    cols = list(df.columns)
    if len(cols) < 2:
        raise SanityCheckError(f"{path}: expected >=2 columns, got {cols}")

    key_col = cols[0]
    val_col = cols[1]
    listish = "matched_entity_ids" in cols or "candidate_entity_ids" in cols
    if listish:
        val_col = "matched_entity_ids" if "matched_entity_ids" in cols else "candidate_entity_ids"

    truth = {}
    for key, value in zip(df[key_col], df[val_col]):
        key = key.strip()
        if not key:
            continue
        bucket = truth.setdefault(key, set())
        if listish:
            bucket.update(v.strip() for v in value.split(",") if v.strip())
        elif value.strip():
            bucket.add(value.strip())
    return truth


def check_ground_truth_ids(truth_ids, known_ids, raise_on_fail=True):
    """Assert ground-truth IDs are a subset of the loaded IDs (A1.4)."""
    missing = set(truth_ids) - set(known_ids)
    if missing and raise_on_fail:
        raise SanityCheckError(
            f"ground truth references {len(missing)} unknown id(s), "
            f"e.g. {sorted(missing)[:5]}"
        )
    return missing
