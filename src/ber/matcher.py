"""Pair scoring model (Owner C, tasks C6-C7).

Stage A only: LightGBM over the features.py library, with a scikit-learn
logistic-regression fallback so the pipeline runs without the optional wheel.
The cross-encoder (Stage B) and LLM tier (Stage C) are cut for the 48-hour
build; research.md section 7's priority order puts them 7th and 8th.

Negatives come from the SAME blocking pipeline used at inference, per
research.md section 3.6: "the training distribution equals the inference
distribution. This is the most common silent bug." Sampling negatives any other
way -- random S1 rows, or all non-matches -- trains the model on a distribution
it will never see.
"""

from __future__ import annotations

import numpy as np

from . import features as featlib

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:          # pragma: no cover - environment dependent
    lgb = None
    HAVE_LIGHTGBM = False

# research.md 3.3 Stage A start params. Foursquare 7th place found much larger
# trees helped at 1.5M-row scale, so tune upward once CV exists.
LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 255,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "verbose": -1,
    "seed": 42,
}


def build_training_pairs(fused, fused_scores, frag_records, s1_records, truth,
                         idf, name_core_counts, address_counts):
    """Featurise every blocked candidate, labelled from ground truth.

    Returns (X, y, groups) where groups holds the fragment id per row, so
    GroupKFold can keep a fragment's candidates together.
    """
    default_idf = featlib.default_idf_value(idf)
    owner = {}
    for s1_id, frag_ids in (truth or {}).items():
        for frag_id in frag_ids:
            owner[frag_id] = s1_id

    rows, labels, groups = [], [], []
    for frag_id, cand_ids in fused.items():
        frag = frag_records.get(frag_id)
        if frag is None or not cand_ids:
            continue
        scores = (fused_scores or {}).get(frag_id, {})
        best = max(scores.values(), default=0.0)
        true_s1 = owner.get(frag_id)

        for rank, s1_id in enumerate(cand_ids):
            s1 = s1_records.get(s1_id)
            if s1 is None:
                continue
            f = featlib.pair_features(
                frag, s1, idf=idf, default_idf=default_idf,
                rrf_score=scores.get(s1_id, 0.0), rrf_rank=rank,
                cand_count=len(cand_ids), best_score=best,
                name_core_counts=name_core_counts, address_counts=address_counts,
            )
            f["vendor_is_s3"] = 1.0 if frag_id.startswith("S3-") else 0.0
            rows.append(featlib.features_to_row(f))
            labels.append(1 if s1_id == true_s1 else 0)
            groups.append(frag_id)

    X = np.asarray(rows, dtype="float32") if rows else np.zeros((0, len(featlib.FEATURE_NAMES)), "float32")
    return X, np.asarray(labels, dtype="int8"), groups


class PairScorer:
    """Stage-A scorer. Falls back to logistic regression without LightGBM."""

    def __init__(self, params=None, n_estimators=300):
        self.params = dict(LGB_PARAMS, **(params or {}))
        self.n_estimators = n_estimators
        self.model = None
        self.backend = None

    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            raise ValueError("training data has only one class")
        if HAVE_LIGHTGBM:
            self.model = lgb.train(self.params, lgb.Dataset(X, label=y),
                                   num_boost_round=self.n_estimators)
            self.backend = "lightgbm"
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ).fit(X, y)
            self.backend = "logistic-regression"
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("call fit() first")
        if len(X) == 0:
            return np.zeros(0, dtype="float64")
        if self.backend == "lightgbm":
            return np.asarray(self.model.predict(X), dtype="float64")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, top=15):
        if self.backend != "lightgbm":
            return {}
        gains = self.model.feature_importance("gain")
        pairs = sorted(zip(featlib.FEATURE_NAMES, gains),
                       key=lambda kv: -kv[1])[:top]
        return {name: float(gain) for name, gain in pairs}


def score_candidates(scorer, fused, fused_scores, frag_records, s1_records,
                     idf, name_core_counts, address_counts):
    """Score every blocked pair, returning {fragment: {s1: probability}}.

    That is exactly the shape decode.pipeline_predictions takes as `marginals`,
    so the placeholder rank heuristic is replaced without touching the decoder.
    """
    X, _, groups = build_training_pairs(
        fused, fused_scores, frag_records, s1_records, truth=None,
        idf=idf, name_core_counts=name_core_counts, address_counts=address_counts)
    if len(X) == 0:
        return {}
    probs = scorer.predict(X)

    out, i = {}, 0
    for frag_id, cand_ids in fused.items():
        if frag_id not in frag_records or not cand_ids:
            continue
        row = {}
        for s1_id in cand_ids:
            if s1_id in s1_records:
                row[s1_id] = float(probs[i])
                i += 1
        out[frag_id] = row
    return out
