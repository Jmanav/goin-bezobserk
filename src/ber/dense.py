"""Dense ANN channel (Owner B, tasks B3.1-B3.6).

Why this channel is load-bearing rather than a nice-to-have: the audit found
41.3% of S2 and 32.4% of S3 India rows carry non-Latin script, and a fully
transliterated name shares ZERO character n-grams with its Latin S1
counterpart while Double Metaphone is Latin-only. For those rows the sparse and
key channels retrieve nothing at all. research.md section 3.2's go/no-go ("drop
dense if it adds < 0.3 PC points") must therefore not be evaluated on a
US-heavy slice.

Sprint 0 ships the off-the-shelf encoder only. Contrastive fine-tuning with
S1-sibling hard negatives is Sprint 1 (B3.6) -- `fit` leaves the hook.

Licences (io_rules.md section 9, research.md section 9 table): all candidates
below are MIT or Apache-2.0 and far under the 8B cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocking import ChannelResult

# research.md section 9 licence table. Pin the commit at competition time
# (B3.1); the revision field is where that pin goes.
MODELS = {
    "multilingual-e5-small": {
        "hf_id": "intfloat/multilingual-e5-small",
        "licence": "MIT", "params": "118M", "dim": 384,
        "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
    "multilingual-e5-large": {
        "hf_id": "intfloat/multilingual-e5-large",
        "licence": "MIT", "params": "560M", "dim": 1024,
        "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "licence": "MIT", "params": "568M", "dim": 1024,
        "query_prefix": "", "passage_prefix": "",
    },
    "qwen3-embedding-0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "licence": "Apache-2.0", "params": "0.6B", "dim": 1024,
        "query_prefix": "", "passage_prefix": "",
    },
}

DEFAULT_MODEL = "multilingual-e5-large"


class DenseUnavailable(RuntimeError):
    """sentence-transformers / faiss / torch are missing (B3.5)."""


@dataclass
class DenseConfig:
    model_key: str = DEFAULT_MODEL
    revision: str | None = None      # pin at competition time (B3.1)
    batch_size: int = 256
    top_n: int = 50
    use_gpu: bool = True
    normalise: bool = True
    cache_path: str | None = None    # Parquet embedding cache (B3.3)

    @property
    def spec(self):
        return MODELS[self.model_key]


def available():
    """True when the dense stack can run here."""
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def describe(config=None):
    """Licence + param record for the compliance log (io_rules.md section 9)."""
    config = config or DenseConfig()
    spec = config.spec
    return {
        "model": spec["hf_id"],
        "revision": config.revision or "UNPINNED -- pin before submission (B3.1)",
        "licence": spec["licence"],
        "params": spec["params"],
        "dim": spec["dim"],
        "available_here": available(),
    }


class DenseChannel:
    """Encoder + FAISS index over S1 (B3.2-B3.4).

    Raises DenseUnavailable rather than silently degrading: a missing dense
    channel is missing recall on ~40% of India rows, and that must be a visible
    decision, not an accident.
    """

    def __init__(self, config=None):
        self.config = config or DenseConfig()
        self.model = None
        self.index = None
        self.s1_ids = None

    def _load_model(self):
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseUnavailable(
                "sentence-transformers is not installed. Install it, or run the "
                "sparse-only path knowingly -- see B3.5."
            ) from exc
        spec = self.config.spec
        kwargs = {}
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        self.model = SentenceTransformer(spec["hf_id"], **kwargs)
        return self.model

    def encode(self, texts, is_query=False):
        model = self._load_model()
        spec = self.config.spec
        prefix = spec["query_prefix"] if is_query else spec["passage_prefix"]
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vectors = model.encode(
            payload,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalise,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def fit(self, s1_ids, s1_texts, embeddings=None):
        try:
            import faiss
        except ImportError as exc:
            raise DenseUnavailable("faiss is not installed") from exc

        self.s1_ids = list(s1_ids)
        vectors = embeddings if embeddings is not None else self.encode(s1_texts)
        vectors = np.ascontiguousarray(vectors, dtype="float32")

        # IndexFlatIP is exact and fine while S1 fits in memory (research.md 3.2
        # budgets S1 <= 3M at ~2 KB/row). Swap to IVF-PQ only when memory-bound;
        # doing it earlier trades recall for nothing.
        index = faiss.IndexFlatIP(vectors.shape[1])
        if self.config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass          # CPU index is correct, just slower
        index.add(vectors)
        self.index = index
        return self

    def query(self, frag_ids, frag_texts, embeddings=None):
        if self.index is None:
            raise DenseUnavailable("call fit() before query()")
        vectors = (embeddings if embeddings is not None
                   else self.encode(frag_texts, is_query=True))
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        scores, ids = self.index.search(vectors, self.config.top_n)

        out = {}
        for row, frag_id in enumerate(frag_ids):
            hits = []
            for rank, (col, score) in enumerate(zip(ids[row], scores[row])):
                if col >= 0:
                    hits.append((self.s1_ids[col], float(score), rank))
            out[frag_id] = hits
        return ChannelResult("dense", out)


def save_embeddings(path, ids, vectors):
    """Cache embeddings to Parquet (B3.3, research.md section 6)."""
    import pandas as pd

    frame = pd.DataFrame({"entity_id": list(ids)})
    frame["vector"] = [v.astype("float32").tolist() for v in vectors]
    frame.to_parquet(path, index=False)
    return path


def load_embeddings(path):
    import pandas as pd

    frame = pd.read_parquet(path)
    ids = frame["entity_id"].tolist()
    vectors = np.asarray([np.asarray(v, dtype="float32")
                          for v in frame["vector"]], dtype="float32")
    return ids, vectors
