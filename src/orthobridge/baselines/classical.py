"""Classical and structured optimization imputation baselines.

Covers:
  - linear: Piecewise linear interpolation with constant boundary extrapolation
  - seasonal: Within-window phase average modulo P with linear fallback
  - hankel_softimpute: Delay-embedded nuclear-norm matrix completion (Soft-Impute)
  - graph_laplacian: Temporal + lag-P seasonal harmonic smoothing
"""

from __future__ import annotations

import warnings
import numpy as np


def linear_impute(values: np.ndarray, fallback_mean: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    obs = np.flatnonzero(np.isfinite(arr))
    if obs.size == 0:
        return np.full(arr.shape, float(fallback_mean), dtype=np.float64)
    filled = np.interp(np.arange(arr.size), obs, arr[obs])
    filled[obs] = arr[obs]
    return filled


def seasonal_impute(values: np.ndarray, period: int = 1, fallback_mean: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    obs_mask = np.isfinite(arr)
    if obs_mask.all():
        return arr
    if not obs_mask.any():
        return np.full(arr.shape, float(fallback_mean), dtype=np.float64)
    result = linear_impute(arr, fallback_mean=fallback_mean)
    eff_period = max(1, int(period))
    positions = np.arange(arr.size)
    phases = positions % eff_period
    missing = ~obs_mask
    for phase in np.unique(phases[missing]):
        donors = obs_mask & (phases == phase)
        if donors.any():
            result[missing & (phases == phase)] = float(np.mean(arr[donors]))
    result[obs_mask] = arr[obs_mask]
    return result


class HankelSoftImpute:
    """Delay-embedded nuclear-norm matrix completion fitted on a training prefix."""

    def __init__(self, embedding_dim: int = 32, shrinkage: float = 0.1, max_iter: int = 2000, tol: float = 1e-4):
        self.embedding_dim = int(embedding_dim)
        self.shrinkage = float(shrinkage)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.center = 0.0
        self.scale = 1.0

    def fit(self, prefix: np.ndarray) -> "HankelSoftImpute":
        arr = np.asarray(prefix, dtype=np.float64).reshape(-1)
        self.center = float(np.nanmean(arr))
        self.scale = max(float(np.nanstd(arr)), 1e-6)
        return self

    def impute(self, query: np.ndarray) -> np.ndarray:
        raw = np.asarray(query, dtype=np.float64).reshape(-1).copy()
        obs = np.isfinite(raw)
        if obs.all():
            return raw
        norm = (raw - self.center) / self.scale
        width = min(self.embedding_dim, len(norm) // 2)
        matrix = np.lib.stride_tricks.sliding_window_view(norm, width).copy()
        m_obs = np.isfinite(matrix)
        known = np.where(m_obs, matrix, 0.0)
        estimate = np.zeros_like(known)
        for _ in range(self.max_iter):
            filled = np.where(m_obs, known, estimate)
            left, singular, right = np.linalg.svd(filled, full_matrices=False)
            thresholded = np.maximum(singular - self.shrinkage, 0.0)
            updated = (left * thresholded) @ right
            change = float(np.linalg.norm(updated - estimate) / max(1.0, float(np.linalg.norm(estimate))))
            estimate = updated
            if change <= self.tol:
                break
        length = sum(estimate.shape) - 1
        total = np.zeros(length, dtype=np.float64)
        counts = np.zeros(length, dtype=np.float64)
        for col in range(estimate.shape[1]):
            total[col:col + estimate.shape[0]] += estimate[:, col]
            counts[col:col + estimate.shape[0]] += 1.0
        repaired = (total / counts) * self.scale + self.center
        repaired[obs] = raw[obs]
        return repaired


class GraphLaplacianImputer:
    """Temporal and lag-P seasonal harmonic extension over a 1D window."""

    def __init__(self, period: int = 1, temporal_weight: float = 1.0, seasonal_weight: float = 0.5):
        self.period = max(1, int(period))
        self.temporal_weight = float(temporal_weight)
        self.seasonal_weight = float(seasonal_weight)
        self.center = 0.0
        self.scale = 1.0

    def fit(self, prefix: np.ndarray) -> "GraphLaplacianImputer":
        arr = np.asarray(prefix, dtype=np.float64).reshape(-1)
        self.center = float(np.nanmean(arr))
        self.scale = max(float(np.nanstd(arr)), 1e-6)
        return self

    def impute(self, query: np.ndarray) -> np.ndarray:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components, laplacian
        from scipy.sparse.linalg import spsolve

        raw = np.asarray(query, dtype=np.float64).reshape(-1).copy()
        obs_mask = np.isfinite(raw)
        if obs_mask.all():
            return raw
        norm = (raw - self.center) / self.scale
        n = len(norm)
        rows, cols, weights = [], [], []

        def add_edge(u: int, v: int, w: float) -> None:
            rows.extend((u, v))
            cols.extend((v, u))
            weights.extend((w, w))

        lags = [(1, self.temporal_weight)]
        if self.period > 1 and self.seasonal_weight > 0.0:
            lags.append((self.period, self.seasonal_weight))
        for lag, weight in lags:
            for t in range(max(0, n - lag)):
                add_edge(t, t + lag, weight)
        adj = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        observed = np.flatnonzero(obs_mask)
        missing = np.flatnonzero(~obs_mask)
        _, labels = connected_components(adj, directed=False)
        anchored = set(labels[observed].tolist())
        if any(comp not in anchored for comp in labels[missing]):
            raise ValueError("Graph contains an unanchored missing component")
        op = laplacian(adj).tocsr()
        system = op[missing][:, missing].tocsc()
        rhs = -(op[missing][:, observed] @ norm[observed])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = np.asarray(spsolve(system, rhs), dtype=np.float64).reshape(-1)
        out = norm.copy()
        out[missing] = sol
        repaired = out * self.scale + self.center
        repaired[obs_mask] = raw[obs_mask]
        return repaired
