"""Exact local macro-F0.5 scorer (Owner D, tasks D1.1-D1.5).

Gate 0 criterion 3: this must reproduce a hand-computed F0.5 on a toy set
exactly. The formula and the singleton rule come from research.md section 1.2
and io_rules.md section 6 and are not to be re-derived.

    F0.5 = 1.25*tp / (0.25*t + p)   for t + p > 0
    F0.5 = 1.0                      when t = p = 0   (the singleton rule)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

BETA_SQ = 0.25          # beta = 0.5
NUMERATOR_SCALE = 1.25  # 1 + beta^2


def entity_f_beta(true_ids, pred_ids):
    """F0.5 for one S1 row (D1.1).

    Comparison is order-insensitive and whitespace-trimmed (D1.5, io_rules.md
    section 5.3), so the caller may pass any iterable.
    """
    t_set = _clean(true_ids)
    p_set = _clean(pred_ids)
    t, p = len(t_set), len(p_set)

    # The singleton rule. A true singleton predicted empty earns a free 1.0;
    # any prediction on it scores 0.0. research.md section 1.2.
    if t == 0 and p == 0:
        return 1.0

    tp = len(t_set & p_set)
    denominator = BETA_SQ * t + p
    if denominator == 0:
        return 0.0
    return NUMERATOR_SCALE * tp / denominator


def _clean(ids):
    if ids is None:
        return frozenset()
    if isinstance(ids, str):
        ids = ids.split(",")
    return frozenset(s.strip() for s in ids if s and s.strip())


@dataclass
class ScoreReport:
    macro_f05: float
    n_rows: int
    per_slice: dict = field(default_factory=dict)
    singleton_accuracy: float | None = None
    n_singletons: int = 0

    def to_dict(self):
        return {
            "macro_f05": self.macro_f05,
            "n_rows": self.n_rows,
            "n_singletons": self.n_singletons,
            "singleton_accuracy": self.singleton_accuracy,
            "per_slice": self.per_slice,
        }


def macro_f_beta(truth, predictions, slices=None):
    """Macro-average F0.5 over every S1 row in `truth` (D1.2).

    Scored over the keys of `truth`, not of `predictions`: an S1 row the model
    never emitted still scores, and omitting it would silently inflate the mean.
    io_rules.md section 5.1 makes a missing S1 row a rejection anyway.

    `slices` maps a slice name to {s1_id: bucket}; each bucket reports its own
    mean (D1.4). Slice membership never changes the macro figure.
    """
    per_row = {}
    for s1_id, true_ids in truth.items():
        per_row[s1_id] = entity_f_beta(true_ids, predictions.get(s1_id))

    n = len(per_row)
    macro = sum(per_row.values()) / n if n else 0.0

    # Singleton accuracy: of the true singletons, how many were left empty.
    # Reported separately because it is all-or-nothing per row.
    singletons = [s for s, ids in truth.items() if not _clean(ids)]
    singleton_acc = (
        sum(per_row[s] for s in singletons) / len(singletons) if singletons else None
    )

    per_slice = {}
    for slice_name, assignment in (slices or {}).items():
        buckets = {}
        for s1_id, score in per_row.items():
            bucket = assignment.get(s1_id, "<unassigned>")
            acc = buckets.setdefault(bucket, [0.0, 0])
            acc[0] += score
            acc[1] += 1
        per_slice[slice_name] = {
            bucket: {"macro_f05": total / count, "n_rows": count}
            for bucket, (total, count) in sorted(buckets.items())
        }

    return ScoreReport(
        macro_f05=macro,
        n_rows=n,
        per_slice=per_slice,
        singleton_accuracy=singleton_acc,
        n_singletons=len(singletons),
    )


def cardinality_slice(truth):
    """The t-bucket slice of research.md section 8.2.

    The measured distribution is t=0 5.6%, t=1 5.4%, t=2-4 63.0%, t>=5 26.0%,
    so t=2-4 and t>=5 are where the score is actually won or lost.
    """
    out = {}
    for s1_id, ids in truth.items():
        t = len(_clean(ids))
        if t == 0:
            out[s1_id] = "t=0"
        elif t == 1:
            out[s1_id] = "t=1"
        elif t <= 4:
            out[s1_id] = "t=2-4"
        else:
            out[s1_id] = "t>=5"
    return out


def country_slice(s1_df):
    """Country slice keyed by S1 id, casefolded but never mapped to a closed set."""
    return {
        eid: (str(c).strip().casefold() or "<missing>")
        for eid, c in zip(s1_df["entity_id"], s1_df["country"])
    }


def vendor_breakdown(truth, predictions):
    """True/predicted counts per vendor prefix, as a sanity report."""
    out = {}
    for label, mapping in (("true", truth), ("pred", predictions)):
        counter = Counter()
        for ids in mapping.values():
            for i in _clean(ids):
                counter[i.split("-", 1)[0]] += 1
        out[label] = dict(counter)
    return out
