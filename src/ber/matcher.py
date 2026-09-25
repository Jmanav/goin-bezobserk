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
                     idf, name_core_counts, address_counts, X=None):
    """Score every blocked pair, returning {fragment: {s1: probability}}.

    That is exactly the shape decode.pipeline_predictions takes as `marginals`,
    so the placeholder rank heuristic is replaced without touching the decoder.

    Pass `X` to reuse a design matrix already built by build_training_pairs over
    the same `fused` set: the training path featurises every candidate pair, and
    recomputing them here doubled the cost of the whole scoring stage for no
    gain. The row order is identical because both iterate `fused` in insertion
    order.
    """
    if X is None:
        X, _, _ = build_training_pairs(
            fused, fused_scores, frag_records, s1_records, truth=None,
            idf=idf, name_core_counts=name_core_counts,
            address_counts=address_counts)
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


# --- persistence ------------------------------------------------------------
#
# The test split ships no ground truth, so the scorer must be trained on train
# and reused on test. Without this the test run falls back to the decoder's rank
# heuristic and discards the model's contribution entirely.


def save_scorer(scorer, path):
    """Persist a fitted scorer, with the feature registry it was trained on.

    The registry is stored so a later load fails loudly on a feature-order
    change rather than silently scoring garbage.
    """
    import pickle
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": scorer.backend,
        "params": scorer.params,
        "feature_names": list(featlib.FEATURE_NAMES),
        "model": (scorer.model.model_to_string()
                  if scorer.backend == "lightgbm" else scorer.model),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_scorer(path):
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if payload["feature_names"] != list(featlib.FEATURE_NAMES):
        raise ValueError(
            "feature registry changed since this scorer was trained: the model "
            "expects %d features in a different order. Retrain rather than "
            "scoring with mismatched columns."
            % len(payload["feature_names"])
        )

    scorer = PairScorer(params=payload["params"])
    scorer.backend = payload["backend"]
    if payload["backend"] == "lightgbm":
        if not HAVE_LIGHTGBM:
            raise RuntimeError("scorer was trained with LightGBM, which is absent")
        scorer.model = lgb.Booster(model_str=payload["model"])
    else:
        scorer.model = payload["model"]
    return scorer


def save_index_stats(path, idf, name_core_counts, address_counts):
    """Persist the S1-derived statistics features depend on.

    idf, chain counts and shared-address counts are computed over S1. Recomputing
    them from the test S1 would silently change the feature distribution the
    model was fitted on, so they travel with the model.
    """
    import pickle
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"idf": idf, "name_core_counts": name_core_counts,
                     "address_counts": address_counts}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_index_stats(path):
    import pickle

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["idf"], payload["name_core_counts"], payload["address_counts"]
