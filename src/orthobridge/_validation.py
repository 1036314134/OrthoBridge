"""Input validation helpers for 1D queries and 2D historical reference banks."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scalar(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _query(values: np.ndarray, length: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size != length or result.dtype.kind not in "iuf":
        raise ValueError(f"query must be a real one-dimensional array of length {length}")
    result = result.astype(np.float64, copy=True)
    if np.isinf(result).any():
        raise ValueError("query must not contain infinity; use NaN for missing values")
    if np.isnan(result).all():
        raise ValueError("query must contain at least one observed value")
    return result


def _references(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or min(result.shape) == 0 or result.dtype.kind not in "iuf":
        raise ValueError("references must be a nonempty real matrix")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("references must be complete and finite")
    return result
