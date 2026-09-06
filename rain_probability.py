"""Make probabilities for nested future-rain windows mutually consistent.

Rain anywhere in the next 5 minutes implies rain somewhere in the next 10
and 30 minutes, so their probabilities must be nondecreasing. Equal-weight
isotonic regression is the Euclidean projection onto that constraint set.

For any nested binary outcome vector y, y belongs to this closed convex set.
The projection p* of raw probabilities p therefore satisfies
    ||p* - y||^2 <= ||p - y||^2 - ||p - p*||^2.
Consequently total Brier loss across the horizons cannot increase. This does
not guarantee improvement for each individual horizon, or for point-in-time
events, whose outcomes are not necessarily nested.
"""

import math
import re
from collections.abc import Mapping
from numbers import Real


def enforce_horizon_coherence(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Project future-window rain probabilities onto increasing horizon order.

    Keys are positive whole-minute labels such as ``5m``, ``10m`` and ``30m``.
    Missing horizons are allowed; an empty mapping returns an empty mapping.
    The input is never mutated and the result preserves its original key order.
    Values must be finite real numbers in [0, 1]. Invalid input raises ValueError.

    Adjacent conflicting blocks are pooled with equal weight per horizon
    (PAVA). Unlike a cumulative maximum, this minimizes total squared change
    without systematically raising all longer-horizon probabilities.
    """
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a mapping of minute labels to probabilities")

    values = {}
    minutes = {}
    for label, value in probabilities.items():
        if not isinstance(label, str) or re.fullmatch(r"[1-9][0-9]*m", label) is None:
            raise ValueError(f"Invalid horizon {label!r}; expected a positive minute label such as '5m'")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Probability for {label!r} must be a finite real number in [0, 1]")
        try:
            probability = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Probability for {label!r} must be a finite real number in [0, 1]") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability for {label!r} must be a finite real number in [0, 1]")
        values[label] = probability
        minutes[label] = int(label[:-1])

    ordered_labels = sorted(values, key=minutes.__getitem__)
    # Each block stores [sum of original probabilities, number of horizons].
    blocks = []
    for label in ordered_labels:
        blocks.append([values[label], 1])
        while len(blocks) > 1:
            left_sum, left_count = blocks[-2]
            right_sum, right_count = blocks[-1]
            if left_sum / left_count <= right_sum / right_count:
                break
            blocks[-2:] = [[left_sum + right_sum, left_count + right_count]]

    offset = 0
    for total, count in blocks:
        mean = total / count
        for label in ordered_labels[offset:offset + count]:
            values[label] = mean
        offset += count
    return values
