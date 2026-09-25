"""Blocking / candidate generation (Owner B, tasks B1-B5).

Direction is fixed by research.md section 3.2: query = each S2/S3 fragment,
index = S1. This matches the many-to-one structure and bounds candidates per
fragment. There is no symmetric all-pairs blocker here and there must not be.

Four channels, fused by RRF:
  B1  char 3-5-gram TF-IDF on the name        -- typos, abbreviations
  B2  word BM25 on name / addr / name+addr    -- rare tokens
  B3  dense ANN over a multilingual encoder   -- semantic, transliteration
  B4  exact keys                              -- anchors

The dense channel is NOT optional in practice: the audit found 41% of S2 and 32%
of S3 India names carry non-Latin script, where a transliterated name shares
ZERO character n-grams with its Latin S1 counterpart and Double Metaphone is
Latin-only. Sparse channels cannot bridge those rows at all. It degrades
gracefully when the libraries or a GPU are absent (B3.5), but a run without it
should be treated as missing recall, not as a clean baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

RRF_K = 60          # research.md 3.2 fusion row: score = sum 1 / (60 + rank)
DEFAULT_TOP_N = 50  # per-channel depth before fusion
DEFAULT_K = 25      # research.md 3.2: keep top-K = 20-30 after fusion
MIN_COS = 0.2       # research.md 3.2 sparse-name row
KEY_COLLISION_CAP = 200
KEY_HIT_MAX_COLLISIONS = 5

_TOKEN = re.compile(r"[^\s]+")


def _tokenise(text):
    """Casefolded whitespace tokens with edge punctuation trimmed.

    The harness feeds pre-normalised text, but BM25 is also called directly by
    Owner C's feature code and by the reporter. Tokenising raw text would make
    "Acme" and "acme" distinct terms and keep "St," apart from "St", so the
    channel would silently retrieve nothing for a differently-cased query.
    """
    out = []
    for raw in _TOKEN.findall(text or ""):
        token = raw.strip(".,;:!?()[]{}\"'").casefold()
        if token:
            out.append(token)
    return out


@dataclass
class ChannelResult:
    """Per-channel hits: {fragment_id: [(s1_id, score, rank), ...]}."""

    name: str
    hits: dict = field(default_factory=dict)

    def ranks(self):
        return {f: {s1: rank for s1, _, rank in rows} for f, rows in self.hits.items()}


# --- B1: char n-gram TF-IDF -------------------------------------------------


class CharTfidfChannel:
    """char_wb 3-5-gram TF-IDF over S1 names (B1.1-B1.4).

    Cosine similarity via normalised sparse dot products, computed in row
    chunks so the fragment x S1 product is never materialised whole -- that
    matrix is 1.7M x 2.2M at real scale and cannot exist in memory.
    """

    def __init__(self, ngram_range=(3, 5), min_cos=MIN_COS, top_n=DEFAULT_TOP_N,
                 chunk_size=2000, max_features=None):
        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, lowercase=False,
            max_features=max_features,
        )
        self.min_cos = min_cos
        self.top_n = top_n
        self.chunk_size = chunk_size
        self.s1_ids = None
        self.matrix = None

    def fit(self, s1_ids, s1_texts):
        self.s1_ids = list(s1_ids)
        self.matrix = self.vectoriser.fit_transform(s1_texts)
        self.matrix = _l2_normalise(self.matrix)
        return self

    def query(self, frag_ids, frag_texts):
        out = {}
        frag_ids = list(frag_ids)
        frag_texts = list(frag_texts)
        index_t = self.matrix.T.tocsr()

        for start in range(0, len(frag_ids), self.chunk_size):
            stop = start + self.chunk_size
            q = _l2_normalise(self.vectoriser.transform(frag_texts[start:stop]))
            sims = (q @ index_t).tocsr()
            for local, frag_id in enumerate(frag_ids[start:stop]):
                row = sims.getrow(local)
                out[frag_id] = _top_from_sparse_row(
                    row, self.s1_ids, self.top_n, self.min_cos
                )
            del sims, q
        return ChannelResult("char_tfidf", out)


def _l2_normalise(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ matrix


def _top_from_sparse_row(row, ids, top_n, min_score):
    if row.nnz == 0:
        return []
    data, cols = row.data, row.indices
    keep = data >= min_score
    data, cols = data[keep], cols[keep]
    if data.size == 0:
        return []
    order = np.argsort(-data)[:top_n]
    return [(ids[cols[i]], float(data[i]), rank)
            for rank, i in enumerate(order)]


# --- B2: word BM25 ----------------------------------------------------------


class BM25Channel:
    """Word-level BM25 (B2.1-B2.3), k1 = 1.2, b = 0.75 per research.md 3.2.

    Implemented on scipy.sparse rather than pulling in bm25s or Pyserini: the
    scoring is a handful of lines, and one fewer dependency is one fewer thing
    to pin for the reproducibility run (io_rules.md section 9).
    """

    def __init__(self, k1=1.2, b=0.75, top_n=DEFAULT_TOP_N, chunk_size=2000):
        self.k1 = k1
        self.b = b
        self.top_n = top_n
        self.chunk_size = chunk_size
        self.vocab = {}
        self.idf = None
        self.weights = None       # term-weight matrix over S1 docs
        self.s1_ids = None

    def fit(self, s1_ids, s1_texts):
        self.s1_ids = list(s1_ids)
        docs = [_tokenise(t) for t in s1_texts]
        self.vocab = {}
        for doc in docs:
            for token in doc:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        n_docs, n_terms = len(docs), len(self.vocab)
        rows, cols, vals = [], [], []
        lengths = np.zeros(n_docs)
        for i, doc in enumerate(docs):
            counts = Counter(self.vocab[t] for t in doc)
            lengths[i] = len(doc)
            for term, tf in counts.items():
                rows.append(i)
                cols.append(term)
                vals.append(tf)

        tf_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, n_terms))
        df = np.asarray((tf_matrix > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        avg_len = lengths.mean() if n_docs else 0.0
        denom_norm = self.k1 * (1 - self.b + self.b * lengths / (avg_len or 1.0))

        # BM25 term weight per (doc, term), precomputed so a query is one dot
        # product: w = idf * tf * (k1 + 1) / (tf + k1(1 - b + b*len/avg)).
        coo = tf_matrix.tocoo()
        weights = coo.data * (self.k1 + 1) / (coo.data + denom_norm[coo.row])
        weights = weights * self.idf[coo.col]
        self.weights = sp.csr_matrix((weights, (coo.row, coo.col)),
                                     shape=(n_docs, n_terms))
        return self

    def query(self, frag_ids, frag_texts):
        out = {}
        frag_ids = list(frag_ids)
        frag_texts = list(frag_texts)
        index_t = self.weights.T.tocsr()

        for start in range(0, len(frag_ids), self.chunk_size):
            stop = start + self.chunk_size
            batch = frag_texts[start:stop]
            q = self._query_matrix(batch)
            scores = (q @ index_t).tocsr()
            for local, frag_id in enumerate(frag_ids[start:stop]):
                out[frag_id] = _top_from_sparse_row(
                    scores.getrow(local), self.s1_ids, self.top_n, 0.0
                )
            del scores, q
        return ChannelResult("bm25", out)

    def _query_matrix(self, texts):
        rows, cols, vals = [], [], []
        for i, text in enumerate(texts):
            for token in set(_tokenise(text)):
                j = self.vocab.get(token)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0)
        return sp.csr_matrix((vals, (rows, cols)),
                             shape=(len(texts), len(self.vocab)))


# --- B4: exact keys ---------------------------------------------------------


def build_key_index(records, idf=None, rare_idf_percentile=0.5):
    """Index S1 by the three key types in research.md 3.2 (B4.1-B4.3).

    `records` maps id -> the Owner A normalised record. Placeholder-derived
    values never become keys (B4.3, io_rules.md section 4): two records must
    never become candidates *because* they share "NA".
    """
    index = defaultdict(list)
    for rec_id, rec in records.items():
        for key in _keys_for(rec, idf, rare_idf_percentile):
            index[key].append(rec_id)

    # research.md 5 "Runtime blow-ups": a key shared by half the corpus is a
    # quadratic time bomb, not a signal.
    return {k: v for k, v in index.items() if len(v) <= KEY_COLLISION_CAP}


def _keys_for(rec, idf=None, rare_idf_percentile=0.5):
    keys = []
    postcode = getattr(rec, "postcode", None)
    numbers = getattr(rec, "addr_numbers", None) or {}
    house = numbers.get("house_number")
    core = getattr(rec, "name_core", "") or ""
    phonetic = getattr(rec, "phonetic_key", None)
    addr_missing = getattr(rec, "addr_missing", False)
    name_missing = getattr(rec, "name_missing", False)

    if postcode and not addr_missing:
        if house:
            keys.append(("pc_house", postcode, house))
        if not name_missing and core:
            for token in _rare_tokens(core, idf, rare_idf_percentile):
                keys.append(("pc_token", postcode, token))
        if phonetic and not name_missing:
            keys.append(("metaphone_pc", phonetic, postcode[:3]))
    return keys


def _rare_tokens(name_core, idf, percentile):
    tokens = [t for t in name_core.split() if t]
    if not tokens:
        return []
    if not idf:
        return tokens[:3]
    scored = sorted(tokens, key=lambda t: -idf.get(t, max(idf.values(), default=1.0)))
    cutoff = max(1, int(len(scored) * percentile))
    return scored[:cutoff]


class KeyChannel:
    """Exact-key lookups (B4). Hits are unranked by nature, so all share rank 0."""

    def __init__(self, max_collisions=KEY_HIT_MAX_COLLISIONS):
        self.index = {}
        self.max_collisions = max_collisions
        self.idf = None

    def fit(self, s1_records, idf=None):
        self.idf = idf
        self.index = build_key_index(s1_records, idf=idf)
        return self

    def query(self, frag_records):
        out = {}
        for frag_id, rec in frag_records.items():
            hits = {}
            for key in _keys_for(rec, self.idf):
                bucket = self.index.get(key)
                if bucket is None:
                    continue
                for s1_id in bucket:
                    hits[s1_id] = max(hits.get(s1_id, 0.0), 1.0 / len(bucket))
            out[frag_id] = [(s1, score, 0) for s1, score in
                            sorted(hits.items(), key=lambda kv: -kv[1])]
        return ChannelResult("keys", out)

    def confident_hits(self, frag_records):
        """Key hits with <= 5 collisions, which B5.2 keeps regardless of K."""
        out = {}
        for frag_id, rec in frag_records.items():
            keep = set()
            for key in _keys_for(rec, self.idf):
                bucket = self.index.get(key)
                if bucket and len(bucket) <= self.max_collisions:
                    keep.update(bucket)
            out[frag_id] = keep
        return out


# --- B5: RRF fusion ---------------------------------------------------------


def reciprocal_rank_fusion(channels, k=DEFAULT_K, rrf_k=RRF_K, always_keep=None):
    """Fuse channel outputs by RRF (B5.1, B5.2).

        score = sum over channels of 1 / (rrf_k + rank)

    `always_keep` maps fragment -> ids that survive the top-K cut regardless
    (research.md 3.2: "plus all key hits with <= 5 collisions").
    """
    fragments = set()
    for channel in channels:
        fragments.update(channel.hits)

    fused = {}
    for frag_id in fragments:
        scores = defaultdict(float)
        for channel in channels:
            for s1_id, _, rank in channel.hits.get(frag_id, []):
                scores[s1_id] += 1.0 / (rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [s1 for s1, _ in ranked[:k]]

        forced = (always_keep or {}).get(frag_id) or set()
        for s1_id in sorted(forced):
            if s1_id not in kept:
                kept.append(s1_id)
        fused[frag_id] = kept
    return fused


def invert_to_s1(fragment_candidates):
    """Turn {fragment: [s1...]} into {s1: [fragment...]} for candidate_pairs.tsv.

    The submission is keyed by S1 (io_rules.md 5.2) while blocking runs
    fragment-first, so the final set has to be inverted before it is written.
    """
    out = defaultdict(list)
    for frag_id, s1_ids in fragment_candidates.items():
        for s1_id in s1_ids:
            out[s1_id].append(frag_id)
    return {s1: sorted(set(frags)) for s1, frags in out.items()}


def token_idf(texts):
    """Token IDF over S1 names, for the rare-token key and Owner C's features."""
    df = Counter()
    n = 0
    for text in texts:
        n += 1
        df.update(set(_tokenise(text)))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
