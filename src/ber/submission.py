"""Submission writer and validator (Owner D, tasks D3.1-D3.4, D4.1-D4.4).

io_rules.md section 5 is the binding format spec and section 8 lists the
format mistakes that cause silent rejection. Everything here raises rather than
warns: a format error costs a whole submission.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

FRAGMENT_PREFIXES = ("S2-", "S3-")


class SubmissionError(AssertionError):
    """A submission-format rule was violated (io_rules.md section 5)."""


def _clean_ids(ids):
    if ids is None:
        return []
    if isinstance(ids, str):
        ids = ids.split(",")
    return sorted({s.strip() for s in ids if s and s.strip()})


def _write_rows(path, header, s1_ids, predictions, allow_empty=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle, delimiter="\t", quoting=csv.QUOTE_NONE,
            escapechar="\\", lineterminator="\n",
        )
        writer.writerow(header)
        for s1_id in s1_ids:
            s1_id = str(s1_id).strip()
            if s1_id in seen:
                raise SubmissionError(f"duplicate source1_entity_id: {s1_id}")
            seen.add(s1_id)

            ids = _clean_ids(predictions.get(s1_id))
            for i in ids:
                if not i.startswith(FRAGMENT_PREFIXES):
                    raise SubmissionError(
                        f"{s1_id}: {i!r} is not an S2-/S3- id. Self-matches to S1 "
                        "and bare numeric ids are both rejected (io_rules.md 5.1)."
                    )
            if not ids and not allow_empty:
                raise SubmissionError(f"{s1_id}: empty list not allowed here")
            writer.writerow([s1_id, ",".join(ids)])
    return path


def write_matching_results(path, s1_ids, predictions):
    """Write matching_results.tsv (D3.1, D3.2).

    Exactly one row per S1 id in `s1_ids`, in the order given. A singleton keeps
    its row with a blank field -- omitting the row is a rejection, not a
    shortcut (io_rules.md 5.1).
    """
    return _write_rows(path, MATCHING_HEADER, s1_ids, predictions)


def write_candidate_pairs(path, s1_ids, candidates):
    """Write candidate_pairs.tsv (D3.3).

    Same schema as matching_results.tsv per io_rules.md 5.2, confirmed against
    the organiser's own ground-truth file, which uses the one-row-per-S1 list
    shape. This must be the FINAL candidate set the model scored, not an early
    blocking pass.
    """
    return _write_rows(path, CANDIDATE_HEADER, s1_ids, candidates)


def read_submission(path):
    """Read a submission back as {s1_id: set(ids)} for round-trip checks."""
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE,
                            escapechar="\\")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            value = row[1] if len(row) > 1 else ""
            out[key] = set(_clean_ids(value))
    return out


@dataclass
class ValidationReport:
    problems: list = field(default_factory=list)
    n_rows: int = 0
    n_predicted_ids: int = 0

    @property
    def ok(self):
        return not self.problems

    def to_dict(self):
        return {"ok": self.ok, "n_rows": self.n_rows,
                "n_predicted_ids": self.n_predicted_ids, "problems": self.problems}


def validate(matching_path, candidate_path, s1_ids, fragment_ids,
             test_s1_path=None, is_submission=False):
    """Team pre-flight assertions (D4.2, D4.3).

    Complements the organiser's validator rather than replacing it: run both.
    Returns a report instead of raising so every problem surfaces at once.

    `s1_ids` is whatever the caller is scoring, which for a sampled development
    run is a subset. That alone cannot tell a real submission from a sample: this
    validator once reported PASS on 20,000 train rows while the organiser's
    validator failed the same files for missing all 1,732,544 test entities.
    Pass `test_s1_path` (test_source1.tsv) with `is_submission=True` to check
    against the authoritative universe instead.
    """
    problems = []
    if is_submission:
        if test_s1_path is None:
            problems.append(
                "is_submission=True requires test_s1_path: without the test "
                "S1 file this cannot verify the submission covers every "
                "required entity (io_rules.md 5.1)"
            )
        else:
            authoritative = read_source_ids(test_s1_path)
            s1_ids = sorted(authoritative)
            fragment_ids = set(fragment_ids)
    expected = [str(s).strip() for s in s1_ids]
    expected_set = set(expected)

    rows = _raw_rows(matching_path, problems, "matching_results")
    submitted = {}
    for key, value in rows:
        if key in submitted:
            problems.append(f"duplicate source1_entity_id row: {key}")
        submitted[key] = value

    missing = expected_set - set(submitted)
    if missing:
        problems.append(
            f"{len(missing)} S1 row(s) missing from the submission "
            f"(io_rules.md 5.1: a missing row is a rejection), e.g. "
            f"{sorted(missing)[:5]}"
        )
    extra = set(submitted) - expected_set
    if extra:
        problems.append(f"{len(extra)} row(s) not in the test S1 set, e.g. "
                        f"{sorted(extra)[:5]}")

    n_ids = 0
    frag_set = set(fragment_ids)
    for key, raw in submitted.items():
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        n_ids += len(ids)
        if len(set(ids)) != len(ids):
            problems.append(f"{key}: duplicate ids within the list")
        for i in ids:
            if not i.startswith(FRAGMENT_PREFIXES):
                problems.append(f"{key}: {i!r} is not an S2-/S3- id")
            elif i not in frag_set:
                problems.append(f"{key}: {i!r} is not in the test set")
        if raw != raw.strip() or ", " in raw:
            problems.append(f"{key}: stray whitespace in the id list")

    if candidate_path is not None:
        cand_rows = dict(_raw_rows(candidate_path, problems, "candidate_pairs"))
        for key, raw in submitted.items():
            matched = {s.strip() for s in raw.split(",") if s.strip()}
            if not matched:
                continue
            cands = {s.strip() for s in cand_rows.get(key, "").split(",") if s.strip()}
            outside = matched - cands
            if outside:
                # io_rules.md 5.2 / D4.3: this is a pipeline bug, not a warning.
                problems.append(
                    f"{key}: {len(outside)} matched id(s) absent from "
                    f"candidate_pairs.tsv, e.g. {sorted(outside)[:3]} "
                    "(matches must be a subset of candidates)"
                )

    return ValidationReport(problems=problems, n_rows=len(submitted),
                            n_predicted_ids=n_ids)


def read_source_ids(path):
    """entity_id column of a source TSV, read under the io_rules.md 1 contract."""
    import csv as _csv

    out = set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = _csv.reader(handle, delimiter="	", quoting=_csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            return out
        for row in reader:
            if row:
                out.add(row[0].strip())
    return out


def _raw_rows(path, problems, label):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE,
                                escapechar="\\")
            header = next(reader, None)
            if header is None:
                problems.append(f"{label}: file is empty")
                return rows
            if len(header) != 2:
                problems.append(f"{label}: header has {len(header)} columns, expected 2")
            for row in reader:
                if not row:
                    continue
                rows.append((row[0].strip(), row[1] if len(row) > 1 else ""))
    except FileNotFoundError:
        problems.append(f"{label}: file not found at {path}")
    return rows


def run_official_validator(validator_path, matching, candidate, test_dir):
    """Wrap the organiser's validate_submission.py (D4.1, io_rules.md section 7).

    It checks format only and does not compute the score. Returns
    (ok, combined_output); the caller decides how loudly to fail.
    """
    cmd = [
        sys.executable, str(validator_path),
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", str(test_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
