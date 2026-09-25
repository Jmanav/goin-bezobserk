"""PC / RR reporting and the blocking harness entry point (Owner B, B6-B7).

Metric definitions are exact, from research.md section 3.2 "Audit metrics":

    PC = |C intersect M| / |M|
    RR = 1 - |C| / (|S1| x |S2 union S3|)

B6.5 is the number that actually bounds the leaderboard score: the share of S1
rows whose FULL true set lies inside the candidate set. Pair-level PC can look
excellent while that figure is poor, because one missed fragment on a t=5 entity
caps its F0.5 no matter how good the model is. With the measured cardinality
(mean t = 3.46, 26% of rows at t >= 5) the two diverge sharply.
"""

from __future__ import annotations

from collections import defaultdict

from . import blocking, dense
from .normalise import normalise_record


def pair_completeness(candidates_by_s1, truth):
    """PC over fragment-level pairs (B6.1)."""
    total = found = 0
    for s1_id, true_frags in truth.items():
        cands = set(candidates_by_s1.get(s1_id, ()))
        for frag in true_frags:
            total += 1
            if frag in cands:
                found += 1
    return {"pair_completeness": found / total if total else 1.0,
            "n_true_pairs": total, "n_found": found}


def reduction_ratio(candidates_by_s1, n_s1, n_fragments):
    """RR against the full cross product (B6.2)."""
    n_candidates = sum(len(v) for v in candidates_by_s1.values())
    denom = n_s1 * n_fragments
    return {"reduction_ratio": 1.0 - (n_candidates / denom) if denom else 1.0,
            "n_candidate_pairs": n_candidates, "n_possible_pairs": denom}


def entity_recall_ceiling(candidates_by_s1, truth):
    """Share of S1 rows whose complete true set is inside C (B6.5).

    This bounds the achievable macro-F0.5: a true match outside C can never be
    predicted, so that row's score is capped before the model runs.
    """
    complete = partial = empty_ok = 0
    capped_loss = 0.0
    for s1_id, true_frags in truth.items():
        true_set = set(true_frags)
        cands = set(candidates_by_s1.get(s1_id, ()))
        if not true_set:
            empty_ok += 1
            continue
        inside = true_set & cands
        if inside == true_set:
            complete += 1
        else:
            partial += 1
            # best achievable F0.5 for this row: predict exactly what is
            # retrievable, i.e. tp = |inside|, p = |inside|, t = |true_set|
            tp = len(inside)
            if tp:
                capped_loss += 1.0 - (1.25 * tp / (0.25 * len(true_set) + tp))
            else:
                capped_loss += 1.0

    n = len(truth) or 1
    non_singleton = complete + partial
    return {
        "n_s1": len(truth),
        "n_singletons": empty_ok,
        "n_fully_covered": complete,
        "n_partially_covered": partial,
        "entity_recall_ceiling": (complete + empty_ok) / n,
        "share_of_non_singletons_fully_covered":
            complete / non_singleton if non_singleton else 1.0,
        "macro_f05_ceiling": 1.0 - capped_loss / n,
    }


def candidates_per_fragment(fragment_candidates):
    """Distribution of candidate counts per fragment (B6.3)."""
    sizes = [len(v) for v in fragment_candidates.values()]
    if not sizes:
        return {"n_fragments": 0}
    sizes_sorted = sorted(sizes)
    return {
        "n_fragments": len(sizes),
        "mean": sum(sizes) / len(sizes),
        "min": sizes_sorted[0],
        "p50": sizes_sorted[len(sizes_sorted) // 2],
        "p95": sizes_sorted[int(len(sizes_sorted) * 0.95)],
        "max": sizes_sorted[-1],
        "n_with_zero": sum(1 for s in sizes if s == 0),
    }


def pc_vs_k(channels, truth, ks=(5, 10, 15, 20, 25, 30, 50), always_keep=None):
    # NOTE: this re-fuses per K, which walks every channel hit again for each
    # point -- ~476s of the 50k run. Fusing once at max(K) and truncating is NOT
    # equivalent, because always_keep ids are appended after the top-K cut, so a
    # truncation clips them and understates PC at small K (verified: 0.0250 vs
    # 0.0298 at K=5). Pass fewer ks when the curve is not the point.
    """PC-vs-K curve (B6.3), the roadmap Sprint 0 deliverable.

    Reports both pair-level PC and the per-entity ceiling at each K, because
    they answer different questions and can move in opposite directions.
    """
    curve = []
    for k in ks:
        fused = blocking.reciprocal_rank_fusion(channels, k=k, always_keep=always_keep)
        by_s1 = blocking.invert_to_s1(fused)
        pc = pair_completeness(by_s1, truth)
        ceiling = entity_recall_ceiling(by_s1, truth)
        curve.append({
            "k": k,
            "pair_completeness": pc["pair_completeness"],
            "entity_recall_ceiling": ceiling["entity_recall_ceiling"],
            "macro_f05_ceiling": ceiling["macro_f05_ceiling"],
            "n_candidate_pairs": sum(len(v) for v in by_s1.values()),
            "mean_candidates_per_fragment":
                candidates_per_fragment(fused)["mean"],
        })
    return curve


def slice_report(candidates_by_s1, truth, s1_country, vendor_of=None):
    """PC per country and per vendor (B6.4)."""
    by_country = defaultdict(lambda: {"n_true": 0, "n_found": 0})
    by_vendor = defaultdict(lambda: {"n_true": 0, "n_found": 0})

    for s1_id, true_frags in truth.items():
        country = s1_country.get(s1_id, "<missing>")
        cands = set(candidates_by_s1.get(s1_id, ()))
        for frag in true_frags:
            hit = frag in cands
            by_country[country]["n_true"] += 1
            by_country[country]["n_found"] += int(hit)
            vendor = (vendor_of or {}).get(frag) or frag.split("-", 1)[0]
            by_vendor[vendor]["n_true"] += 1
            by_vendor[vendor]["n_found"] += int(hit)

    def finish(d):
        return {k: {**v, "pair_completeness": v["n_found"] / v["n_true"]
                    if v["n_true"] else 1.0}
                for k, v in sorted(d.items())}

    return {"by_country": finish(by_country), "by_vendor": finish(by_vendor)}


# --- B7: harness entry point ------------------------------------------------


class BlockingHarness:
    """One interface over all four channels plus RRF (B7.1).

    Sprint 1 tunes the channels behind this; Sprint 0 only has to make them run
    and report honestly.
    """

    def __init__(self, k=blocking.DEFAULT_K, top_n=blocking.DEFAULT_TOP_N,
                 use_dense=True, dense_config=None, shard_by_country=False):
        self.k = k
        self.top_n = top_n
        # B0.1 is unresolved, so country sharding stays OFF by default and must
        # never exclude an unseen label when enabled (io_rules.md section 3).
        self.shard_by_country = shard_by_country
        self.use_dense = use_dense
        self.dense_config = dense_config
        self.char = blocking.CharTfidfChannel(top_n=top_n)
        self.bm25 = blocking.BM25Channel(top_n=top_n)
        self.keys = blocking.KeyChannel()
        self.dense = None
        self.idf = None
        self._notes = []

    def fit(self, s1_frame):
        ids, records = _records(s1_frame)
        names = [records[i].name_norm for i in ids]
        blobs = [f"{records[i].name_norm} {records[i].addr_norm}".strip() for i in ids]

        self.idf = blocking.token_idf(names)
        self.char.fit(ids, names)
        self.bm25.fit(ids, blobs)
        self.keys.fit(records, idf=self.idf)

        if self.use_dense:
            if dense.available():
                self.dense = dense.DenseChannel(self.dense_config).fit(ids, blobs)
                self._notes.append("dense channel active")
            else:
                self._notes.append(
                    "DENSE CHANNEL UNAVAILABLE -- faiss/sentence-transformers not "
                    "installed. Sparse channels retrieve nothing for fully "
                    "transliterated names (41% of S2 India rows), so treat this "
                    "run as missing recall, not a clean baseline."
                )
        self._s1_ids = ids
        return self

    def query(self, frag_frame):
        ids, records = _records(frag_frame)
        names = [records[i].name_norm for i in ids]
        blobs = [f"{records[i].name_norm} {records[i].addr_norm}".strip() for i in ids]

        channels = [self.char.query(ids, names), self.bm25.query(ids, blobs),
                    self.keys.query(records)]
        if self.dense is not None:
            channels.append(self.dense.query(ids, blobs))

        always = self.keys.confident_hits(records)
        fused = blocking.reciprocal_rank_fusion(channels, k=self.k, always_keep=always)
        return fused, channels, always

    @property
    def notes(self):
        return list(self._notes)


def _records(frame):
    ids, records = [], {}
    for row in frame.itertuples(index=False):
        ids.append(row.entity_id)
        records[row.entity_id] = normalise_record(
            row.business_name, row.business_address, row.country
        )
    return ids, records
