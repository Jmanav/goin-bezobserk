"""N2 fragment exclusivity and the N1 expected-F0.5 decoder (Owner D, D5-D6).

Pseudocode is given verbatim in research.md sections 3.4 and 3.5; this
implements it rather than redesigning it. The maths is established (Chai 2005,
Jansche 2007, Dembczynski 2011, Ye 2012); what is specific here is macro
averaging over S1 rows, the singleton rule, and blocked-out mass.

Measured cardinality (train, 2026-09-25): singleton rate 5.58%, t=2-4 63.0%,
t>=5 26.0%, mean t=3.46. The empty set remains the DEFAULT when candidates are
weak, but it is the optimal answer for a small minority of rows -- so the
decoder is mostly choosing a set SIZE, not choosing whether to abstain.
"""

from __future__ import annotations

import math
from functools import lru_cache

NULL = "<NULL>"

BETA_SQ = 0.25
NUMERATOR_SCALE = 1.25

PROB_FLOOR = 0.02   # research.md 3.5: candidates below this are dropped
MAX_LAMBDA_TERMS = 12


def poisson_binomial_pmf(probs):
    """P(k successes) for independent Bernoulli trials, O(n^2) DP (D6.1).

    Returns a list of length len(probs) + 1.
    """
    pmf = [1.0]
    for p in probs:
        p = min(max(float(p), 0.0), 1.0)
        nxt = [0.0] * (len(pmf) + 1)
        for k, acc in enumerate(pmf):
            if acc == 0.0:
                continue
            nxt[k] += acc * (1.0 - p)
            nxt[k + 1] += acc * p
        pmf = nxt
    return pmf


@lru_cache(maxsize=4096)
def _poisson_pmf(lam, max_k):
    """Cached: lambda comes from a small set of per-stratum estimates, so the
    same handful of values recur across millions of S1 rows."""
    if lam <= 0:
        return (1.0,) + (0.0,) * max_k
    out = []
    for k in range(max_k + 1):
        out.append(math.exp(-lam) * lam**k / math.factorial(k))
    total = sum(out)
    return tuple(v / total for v in out) if total else tuple(out)


def _convolve(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0.0:
            continue
        for j, bj in enumerate(b):
            if bj:
                out[i + j] += ai * bj
    return out


def expected_f_beta(k, q, lam=0.0):
    """E[F0.5] of predicting the top-k candidates (research.md 3.5, D6.1-D6.3).

    `q` is the candidate marginals sorted descending; `lam` is the expected
    number of true matches that blocking never retrieved.
    """
    q = list(q)
    inside = poisson_binomial_pmf(q[:k])
    outside = poisson_binomial_pmf(q[k:])
    if lam > 0:
        outside = _convolve(outside, _poisson_pmf(lam, MAX_LAMBDA_TERMS))

    # k == 0 is the singleton rule: F = 1 exactly when the true set is empty.
    if k == 0:
        return outside[0]

    total = 0.0
    for a, pa in enumerate(inside):
        if a == 0 or pa == 0.0:
            continue          # tp = 0 with p = k > 0 scores 0
        for b, pb in enumerate(outside):
            if pb:
                total += pa * pb * NUMERATOR_SCALE * a / (BETA_SQ * (a + b) + k)
    return total


def _score_from_pmfs(k, inside, outside):
    if k == 0:
        return outside[0]
    total = 0.0
    for a, pa in enumerate(inside):
        if a == 0 or pa == 0.0:
            continue
        for b, pb in enumerate(outside):
            if pb:
                total += pa * pb * NUMERATOR_SCALE * a / (BETA_SQ * (a + b) + k)
    return total


def best_k(q, lam=0.0):
    """argmax over k of expected_f_beta (D6.4). Top-k is optimal under independence.

    Sweeps k upward while carrying both PMFs incrementally: adding candidate k
    to the inside set is one O(k) DP step, and removing it from the outside set
    is the same step run backwards. Recomputing both from scratch for every k
    made this the decode bottleneck (~1.6 ms/row, i.e. an hour over 2.2M rows).
    """
    q = sorted((float(x) for x in q), reverse=True)
    m = len(q)

    lam_pmf = list(_poisson_pmf(lam, MAX_LAMBDA_TERMS)) if lam > 0 else None

    # Suffix PMFs, built once by scanning from the end: suffix[k] is the pmf of
    # the candidates NOT selected when the top k are taken. Deconvolving them
    # out of the full pmf instead (forward substitution, dividing by 1 - p at
    # each step) is numerically unstable and silently picked the wrong k -- a
    # pmf that should sum to 1 summed to -12531 on a 12-candidate row.
    suffix = [None] * (m + 1)
    suffix[m] = [1.0]
    for i in range(m - 1, -1, -1):
        suffix[i] = _add_trial(suffix[i + 1], min(max(q[i], 0.0), 1.0))

    inside = [1.0]
    best, best_score = 0, _score_from_pmfs(0, inside, _with_lam(suffix[0], lam_pmf))
    for k in range(1, m + 1):
        inside = _add_trial(inside, min(max(q[k - 1], 0.0), 1.0))
        score = _score_from_pmfs(k, inside, _with_lam(suffix[k], lam_pmf))
        if score > best_score:
            best, best_score = k, score
    return best, best_score


def _with_lam(pmf, lam_pmf):
    return pmf if lam_pmf is None else _convolve(pmf, lam_pmf)


def _add_trial(pmf, p):
    out = [0.0] * (len(pmf) + 1)
    for k, acc in enumerate(pmf):
        if acc:
            out[k] += acc * (1.0 - p)
            out[k + 1] += acc * p
    return out


def best_set(q, lam=0.0, floor=PROB_FLOOR):
    """Indices of the chosen candidates, empty by default (D6.4, D6.5).

    Indices refer to the ORIGINAL order of `q`, so the caller can map them back
    to candidate ids without re-sorting.
    """
    ranked = sorted(range(len(q)), key=lambda i: -float(q[i]))
    kept = [i for i in ranked if float(q[i]) >= floor]
    if not kept:
        return []
    k, _ = best_k([q[i] for i in kept], lam)
    return kept[:k]


def decode_entity(candidate_ids, marginals, lam=0.0, floor=PROB_FLOOR):
    """Choose one S1 row's output set. Empty is the default."""
    chosen = best_set(marginals, lam=lam, floor=floor)
    return [candidate_ids[i] for i in chosen]


# --- N2: fragment exclusivity + NULL (research.md 3.4, D5.1-D5.5) -----------


def fragment_marginals(scores, lam_null=1.0):
    """Per-fragment softmax over {S1 candidates} plus NULL (D5.1).

    `scores` maps fragment id -> {s1_id: calibrated p}. Returns the same shape
    with an added NULL key, each row summing to 1.

    No global solver: with a deduplicated master the only hard constraint is
    "each fragment matches at most one S1", which decomposes per fragment
    (D5.2). Connected components would instead merge two S1 entities whenever a
    single noisy fragment links to both.
    """
    out = {}
    for frag_id, cands in scores.items():
        weights = {}
        for s1_id, p in cands.items():
            p = min(max(float(p), 0.0), 1.0 - 1e-12)
            weights[s1_id] = p / (1.0 - p)      # odds
        z = lam_null + sum(weights.values())
        if z <= 0:
            out[frag_id] = {NULL: 1.0}
            continue
        row = {s1_id: w / z for s1_id, w in weights.items()}
        row[NULL] = lam_null / z
        out[frag_id] = row
    return out


def s1_candidate_marginals(pi):
    """Invert fragment marginals into {s1_id: {fragment_id: pi}} for N1 (D5.5)."""
    out = {}
    for frag_id, row in pi.items():
        for s1_id, value in row.items():
            if s1_id != NULL:
                out.setdefault(s1_id, {})[frag_id] = value
    return out


def enforce_exclusivity(predictions, pi):
    """Final guard (D5.4): a fragment claimed by >1 S1 goes to the argmax.

    Runs after decoding, since the per-S1 decoders choose independently and can
    still collide on a shared fragment.
    """
    owner = {}
    for s1_id, frags in predictions.items():
        for frag_id in frags:
            score = pi.get(frag_id, {}).get(s1_id, 0.0)
            current = owner.get(frag_id)
            if current is None or score > current[1]:
                owner[frag_id] = (s1_id, score)

    return {
        s1_id: [f for f in frags if owner.get(f, (None,))[0] == s1_id]
        for s1_id, frags in predictions.items()
    }


def decode_all(s1_marginals, lam_by_s1=None, lam_default=0.0, floor=PROB_FLOOR):
    """Decode every S1 row, then apply the exclusivity guard."""
    predictions = {}
    for s1_id, cands in s1_marginals.items():
        ids = list(cands)
        q = [cands[i] for i in ids]
        lam = (lam_by_s1 or {}).get(s1_id, lam_default)
        predictions[s1_id] = decode_entity(ids, q, lam=lam, floor=floor)
    return predictions
