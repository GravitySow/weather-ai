"""A fitted rain classifier blended with the current rain-state baseline.

The scalar weight is selected on validation observations only. It makes no
claim to predict new storm onset; persistence is principally useful while rain
is ongoing. The wrapper is serialized with the estimator in a model bundle.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def select_persistence_weight(y_true, model_probability, persistence):
    """Least-squares convex model weight, with a raw-model fallback on ties.

    Minimize mean((q + alpha * (p - q) - y)**2), 0 <= alpha <= 1.
    Callers must pass validation data, never the final test observations.
    """
    y, p, q = (np.asarray(v, dtype=float) for v in (y_true, model_probability, persistence))
    if y.ndim != 1 or p.shape != y.shape or q.shape != y.shape or y.size == 0:
        raise ValueError("Expected nonempty, aligned one-dimensional arrays")
    if not all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in (y, p, q)):
        raise ValueError("Labels and probabilities must be finite and between zero and one")
    direction = p - q
    denominator = float(np.dot(direction, direction))
    if denominator <= np.finfo(float).eps:
        return 1.0
    weight = float(np.clip(np.dot(y - q, direction) / denominator, 0.0, 1.0))
    blended = q + weight * direction
    if np.mean((blended - y) ** 2) >= np.mean((p - y) ** 2) - 1e-12:
        return 1.0
    return weight


class PersistenceBlendClassifier(ClassifierMixin, BaseEstimator):
    """Prediction-only wrapper around an already fitted binary classifier."""

    def __init__(self, estimator, model_weight=1.0, rain_feature="rain_now", rain_threshold=.10):
        self.estimator = estimator
        self.model_weight = model_weight
        self.rain_feature = rain_feature
        self.rain_threshold = rain_threshold

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def feature_names_in_(self):
        return self.estimator.feature_names_in_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    def __sklearn_is_fitted__(self):
        return hasattr(self.estimator, "classes_")

    def predict_proba(self, X):
        weight = float(self.model_weight)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("model_weight must be finite and between zero and one")
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Rain blend requires classifier classes [0, 1]")
        if hasattr(X, "columns"):
            rain = np.asarray(X[self.rain_feature], dtype=float)
        else:
            names = list(self.feature_names_in_)
            rain = np.asarray(X, dtype=float)[:, names.index(self.rain_feature)]
        if not np.isfinite(rain).all():
            raise ValueError("Current rain observations must be finite")
        p = self.estimator.predict_proba(X)[:, 1]
        persistence = (rain > self.rain_threshold).astype(float)
        blended = weight * p + (1 - weight) * persistence
        return np.column_stack((1 - blended, blended))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)
