"""Historical reference retrieval baselines: Observed-kNN, KDTree-kNN, and Seasonal-kNN."""

from __future__ import annotations

import numpy as np

from .classical import linear_impute, seasonal_impute

_BLOCK_ELEMENTS = 1_048_576


def _masked_distances(query: np.ndarray, bank: np.ndarray, columns: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    count = len(bank) if rows is None else len(rows)
    out = np.empty(count, dtype=np.float64)
    step = max(1, _BLOCK_ELEMENTS // bank.shape[1])
    for start in range(0, count, step):
        sel = slice(start, start + step) if rows is None else rows[start:start + step]
        block = bank[sel][:, columns].astype(np.float64, copy=False)
        diff = block - query[columns]
        out[start:start + len(block)] = np.einsum("ij,ij->i", diff, diff) / columns.size
    return out


def _rank_candidates(indices: np.ndarray, distances: np.ndarray, k: int) -> np.ndarray:
    return np.lexsort((indices, distances))[:k]


def retrieval_impute(
    normalized_query: np.ndarray,
    normalized_bank: np.ndarray,
    method: str,
    *,
    period: int = 24,
    k: int = 8,
    candidate_k: int = 64,
    temperature: float = 1.0,
) -> np.ndarray:
    """Impute a normalized 1D query using Observed-kNN, KDTree-kNN, or Seasonal-kNN."""
    if method not in ("observed_knn", "kdtree_knn", "seasonal_knn"):
        raise ValueError(f"Unsupported retrieval method: {method!r}")
    query = np.asarray(normalized_query, dtype=np.float64).copy()
    obs_mask = np.isfinite(query)
    if obs_mask.all():
        return query
    if not obs_mask.any():
        return np.zeros_like(query)
    bank = np.asarray(normalized_bank, dtype=np.float64)
    if len(bank) == 0:
        return linear_impute(query, fallback_mean=0.0)

    observed = np.flatnonzero(obs_mask)
    count = min(int(k), len(bank))
    if method == "observed_knn":
        all_rows = np.arange(len(bank), dtype=np.int64)
        dists = _masked_distances(query, bank, observed)
        order = _rank_candidates(all_rows, dists, count)
        selected, selected_dists = all_rows[order], dists[order]
    elif method == "kdtree_knn":
        from scipy.spatial import cKDTree

        completed = linear_impute(query, fallback_mean=0.0)
        budget = min(len(bank), max(int(k), int(candidate_k)))
        tree = cKDTree(np.ascontiguousarray(bank, dtype=np.float64))
        _, cand = tree.query(completed, k=budget, workers=1)
        cand = np.unique(np.atleast_1d(cand).astype(np.int64))
        dists = _masked_distances(query, bank, observed, cand)
        order = _rank_candidates(cand, dists, count)
        selected, selected_dists = cand[order], dists[order]
    else:
        completed = seasonal_impute(query, period=period, fallback_mean=0.0)
        full_dists = _masked_distances(completed, bank, np.arange(query.size))
        all_rows = np.arange(len(bank), dtype=np.int64)
        cand = np.unique(all_rows[_rank_candidates(all_rows, full_dists, count)])
        dists = _masked_distances(query, bank, observed, cand)
        order = _rank_candidates(cand, dists, count)
        selected, selected_dists = cand[order], dists[order]

    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(selected_dists - selected_dists.min()) / float(temperature))
    weights /= weights.sum()
    result = query.copy()
    result[~obs_mask] = weights @ bank[selected][:, ~obs_mask]
    result[obs_mask] = query[obs_mask]
    return result
