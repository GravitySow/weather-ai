"""Lightweight distribution-drift checks for weather model inputs.

This is a monitoring aid, not a retraining trigger.  The baseline is stored
with each model bundle and can be compared with a recent feature window
without touching production data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_baseline(frame, features, bins=10):
    baseline = {"schema_version": 1, "features": {}, "bins": int(bins)}
    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        quantiles = np.quantile(values, np.linspace(0, 1, bins + 1)).astype(float)
        edges = np.unique(quantiles)
        if len(edges) < 2:
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        # Keep out-of-range values in the edge buckets; otherwise a sudden
        # sensor excursion would produce an all-zero histogram and undefined
        # PSI instead of the desired high-drift signal.
        clipped = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(clipped, bins=edges)
        proportions = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        baseline["features"][name] = {
            "n": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
            "edges": edges.tolist(), "proportions": proportions.tolist(),
        }
    return baseline


def save_baseline(baseline, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_baseline(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _psi(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = np.clip(expected / expected.sum(), 1e-6, None)
    if actual.sum() <= 0:
        return 1.0
    actual = np.clip(actual / actual.sum(), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compare_distribution(frame, baseline):
    """Return per-feature PSI and standardized mean shift for a recent frame."""
    rows = {}
    for name, spec in baseline.get("features", {}).items():
        if name not in frame:
            values = np.array([])
        else:
            values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            rows[name] = {"n": 0, "psi": None, "mean_shift": None, "status": "missing"}
            continue
        edges = np.asarray(spec["edges"], dtype=float)
        clipped = np.clip(values, edges[0], edges[-1])
        counts, _ = np.histogram(clipped, bins=edges)
        psi = _psi(spec["proportions"], counts)
        std = max(float(spec.get("std", 0.0)), 1e-6)
        shift = (float(np.mean(values)) - float(spec["mean"])) / std
        rows[name] = {"n": int(len(values)), "psi": psi, "mean_shift": float(shift),
                      "status": "high" if psi >= 0.25 else "watch" if psi >= 0.10 else "ok"}
    valid = [v["psi"] for v in rows.values() if v["psi"] is not None]
    return {"features": rows, "mean_psi": float(np.mean(valid)) if valid else None,
            "high_drift_features": [k for k, v in rows.items() if v["status"] == "high"],
            "n_features": len(rows)}
