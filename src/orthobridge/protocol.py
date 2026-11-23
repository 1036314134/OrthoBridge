"""Evaluation protocol, window splitting, masking, calibration, and metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from .ortho_bridge import OrthoBridgeIndex

WINDOW_LENGTH = 512
FORECAST_HORIZON = 96
MODES = ("point", "block", "tail")
RATES = (0.1, 0.2, 0.3, 0.4, 0.5)
TEST_SEEDS = (42, 43, 44, 45, 46)
VALIDATION_SEED = 42
VALIDATION_MASK_SEED = 20261010

DEFAULT_CALIBRATION_SHRINKAGES = {
    "short": 0.25,
    "long_internal": 0.75,
    "long_edge": 1.0,
}


@dataclass(frozen=True)
class PreparedSeries:
    dataset_id: str
    paper_name: str
    domain: str
    period: int
    series: np.ndarray
    bank: np.ndarray
    center: float
    repair_scale: float
    score_scale: float
    donor_starts: np.ndarray
    all_training_starts: np.ndarray
    fit_end: int
    train_end: int
    validation_end: int
    validation_origins: tuple[int, ...]
    test_origins: tuple[int, ...]
    metadata: dict


@dataclass(frozen=True)
class PreparedWindow:
    origin: int
    split: str
    mode: str
    rate: float
    seed: int
    mask_seed: int
    inference_seed: int
    normalized_masked: np.ndarray
    raw_masked: np.ndarray
    truth: np.ndarray
    hidden: np.ndarray
    observed: np.ndarray


def _json_digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts) -> int:
    return int(_json_digest(parts)[:8], 16)


def evaluation_mask_seed(meta: dict, origin: int, mode: str, rate: float, seed: int, split: str) -> int:
    effective_seed = int(seed) if split == "test" else VALIDATION_MASK_SEED
    parts = (
        meta["resource_id"],
        meta["source_record_key"],
        int(meta["channel_index"]),
        int(origin),
        mode,
        float(rate),
        effective_seed,
    )
    return int(_json_digest(parts)[:15], 16)


def training_starts(target: np.ndarray, fit_end: int, window_length: int = WINDOW_LENGTH) -> list[int]:
    finite = np.isfinite(np.asarray(target).reshape(-1)[:fit_end])
    counts = np.r_[0, np.cumsum(finite, dtype=np.int64)]
    eligible = np.flatnonzero(counts[window_length:] - counts[:-window_length] >= 16)
    if len(eligible) > 512:
        eligible = eligible[np.floor(np.linspace(0, len(eligible) - 1, 512)).astype(int)]
    return eligible.tolist()


def mask_target(values: np.ndarray, mode: str, rate: float, seed: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or mode not in MODES or not 0.0 <= rate <= 1.0:
        raise ValueError("Invalid target mask parameters")
    observed = np.flatnonzero(np.isfinite(values))
    count = int(np.floor(rate * len(observed) + 0.5))
    hidden = np.zeros(values.shape, dtype=bool)
    if count:
        rng = np.random.default_rng(seed)
        if mode == "point":
            chosen = rng.choice(observed, count, replace=False)
        elif mode == "block":
            start = int(rng.integers(0, len(observed) - count + 1))
            chosen = observed[start:start + count]
        else:
            chosen = observed[-count:]
        hidden[chosen] = True
    return hidden


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def load_benchmark_series(
    dataset_key: str,
    data_dir: str | Path | None = None,
) -> PreparedSeries:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    meta_path = root / "datasets_metadata.json"
    archive_path = root / "orthobridge_datasets_uv512.npz"
    if not meta_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing dataset archive or metadata in {root}. Expected "
            f"{archive_path.name} and {meta_path.name}."
        )
    catalog = json.loads(meta_path.read_text(encoding="utf-8"))
    by_key = {}
    for item in catalog["datasets"]:
        by_key[item["dataset_id"]] = item
        by_key[item["paper_name"]] = item
        by_key[item["paper_name"].lower()] = item
    if dataset_key not in by_key:
        raise KeyError(f"Unknown dataset {dataset_key!r}")
    meta = by_key[dataset_key]
    did = meta["dataset_id"]
    with np.load(archive_path, allow_pickle=False) as archive:
        raw = np.asarray(archive[did], dtype=np.float64).reshape(-1).copy()

    n = len(raw)
    train_end = int(np.floor(0.7 * n))
    validation_end = int(np.floor(0.8 * n))
    fit_end = train_end - WINDOW_LENGTH
    starts = training_starts(raw, fit_end)
    donors = [s for s in starts if np.isfinite(raw[s:s + WINDOW_LENGTH]).all()]
    prefix = raw[:fit_end]
    center = float(np.nanmean(prefix))
    repair_scale = max(float(np.nanstd(prefix)), 1e-6)
    finite_train = raw[:train_end][np.isfinite(raw[:train_end])]
    score_scale = max(float(np.std(finite_train)), 1e-6)
    bank = (
        np.stack([(raw[s:s + WINDOW_LENGTH] - center) / repair_scale for s in donors])
        if donors
        else np.empty((0, WINDOW_LENGTH), dtype=np.float64)
    )
    for arr in (raw, bank):
        arr.flags.writeable = False
    return PreparedSeries(
        dataset_id=did,
        paper_name=meta["paper_name"],
        domain=meta["domain"],
        period=int(meta["period"]),
        series=raw,
        bank=bank,
        center=center,
        repair_scale=repair_scale,
        score_scale=score_scale,
        donor_starts=np.asarray(donors, dtype=np.int64),
        all_training_starts=np.asarray(starts, dtype=np.int64),
        fit_end=fit_end,
        train_end=train_end,
        validation_end=validation_end,
        validation_origins=tuple(int(o) for o in meta["validation_origins"]),
        test_origins=tuple(int(o) for o in meta["test_origins"]),
        metadata=meta,
    )


def make_masked_window(
    prepared: PreparedSeries,
    origin: int,
    mode: str,
    rate: float,
    seed: int = 42,
    split: str = "test",
) -> PreparedWindow:
    if origin < WINDOW_LENGTH or origin > len(prepared.series):
        raise ValueError("Origin out of series bounds")
    truth = prepared.series[origin - WINDOW_LENGTH:origin].copy()
    m_seed = evaluation_mask_seed(prepared.metadata, origin, mode, rate, seed, split)
    i_seed = stable_seed("inference", prepared.metadata["array_id"], split, int(origin), mode, float(rate), int(seed))
    hidden = mask_target(truth, mode, rate, m_seed)
    raw_masked = truth.copy()
    raw_masked[hidden] = np.nan
    observed = np.isfinite(raw_masked)
    normalized = (raw_masked - prepared.center) / prepared.repair_scale
    for arr in (normalized, raw_masked, truth, hidden, observed):
        arr.flags.writeable = False
    return PreparedWindow(
        origin=int(origin),
        split=split,
        mode=mode,
        rate=float(rate),
        seed=int(seed),
        mask_seed=m_seed,
        inference_seed=i_seed,
        normalized_masked=normalized,
        raw_masked=raw_masked,
        truth=truth,
        hidden=hidden,
        observed=observed,
    )


def restore_raw_scale(
    normalized_repaired: np.ndarray,
    normalized_query: np.ndarray,
    raw_masked: np.ndarray,
    center: float,
    repair_scale: float,
) -> np.ndarray:
    obs = np.isfinite(normalized_query)
    if not np.isfinite(normalized_repaired).all():
        raise ValueError("Nonfinite normalized repair")
    if not np.array_equal(normalized_repaired[obs].view("u8"), normalized_query[obs].view("u8")):
        raise ValueError("Imputer modified observed coordinates in normalized space")
    values = normalized_repaired * repair_scale + center
    values[obs] = raw_masked[obs]
    return values


def _gap_geometry(raw_masked: np.ndarray) -> str:
    edges = np.flatnonzero(np.diff(np.r_[False, np.isnan(raw_masked), False])).reshape(-1, 2)
    if not len(edges):
        return "short"
    lengths = edges[:, 1] - edges[:, 0]
    idx = int(np.argmax(lengths))
    start, end = int(edges[idx, 0]), int(edges[idx, 1])
    length = end - start
    if length <= 8:
        return "short"
    if start == 0 or end == len(raw_masked):
        return "long_edge"
    return "long_internal"


def select_shrinkage(raw_masked: np.ndarray, shrinkage_map: dict[str, float] | None = None) -> float:
    mapping = DEFAULT_CALIBRATION_SHRINKAGES if shrinkage_map is None else shrinkage_map
    return float(mapping[_gap_geometry(raw_masked)])


def calibrate_shrinkage(
    series_list: list[PreparedSeries],
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seed: int = 20261030,
) -> dict[str, float]:
    """Calibrate shrinkage strictly within the training prefix using earlier donors."""
    losses: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for prep in series_list:
        y = prep.series[:prep.fit_end]
        for offset in (512, 0):
            origin = len(y) - offset
            if origin < WINDOW_LENGTH:
                continue
            earlier = prep.donor_starts[prep.donor_starts + WINDOW_LENGTH <= origin - WINDOW_LENGTH]
            if len(earlier) < 1:
                continue
            sub_prefix = y[:origin - WINDOW_LENGTH]
            c = float(np.nanmean(sub_prefix))
            s = max(float(np.nanstd(sub_prefix)), 1e-6)
            bank = np.stack([(y[st:st + WINDOW_LENGTH] - c) / s for st in earlier])
            index = OrthoBridgeIndex(bank, period=prep.period)
            truth = y[origin - WINDOW_LENGTH:origin].copy()
            if np.isfinite(truth).sum() < 16:
                continue
            for mode in ("point", "block", "tail"):
                for rate in (0.1, 0.3, 0.5):
                    m_seed = stable_seed("calibration", prep.dataset_id, int(origin), mode, rate, int(seed))
                    hidden = mask_target(truth, mode, rate, m_seed)
                    if not hidden.any():
                        continue
                    raw_q = truth.copy()
                    raw_q[hidden] = np.nan
                    if not np.isfinite(raw_q).any():
                        continue
                    geom = _gap_geometry(raw_q)
                    norm_q = (raw_q - c) / s
                    for lam in grid:
                        rep = index.repair(norm_q, shrinkage=float(lam), backend="auto")
                        out = restore_raw_scale(rep.values, norm_q, raw_q, c, s)
                        nmae = float(np.mean(np.abs(out[hidden] - truth[hidden]) / prep.score_scale))
                        losses[geom][float(lam)][prep.dataset_id].append(nmae)
    chosen = dict(DEFAULT_CALIBRATION_SHRINKAGES)
    for geom, by_lam in losses.items():
        scores = {
            lam: statistics.fmean(statistics.fmean(vals) for vals in by_ds.values())
            for lam, by_ds in by_lam.items()
            if by_ds
        }
        if scores:
            chosen[geom] = min(scores, key=lambda lam: (scores[lam], lam))
    return chosen


def compute_metrics(values: np.ndarray, truth: np.ndarray, hidden: np.ndarray, score_scale: float) -> dict[str, float]:
    err = np.asarray(values, dtype=np.float64)[hidden] - np.asarray(truth, dtype=np.float64)[hidden]
    if err.size == 0 or not np.isfinite(err).all():
        raise ValueError("Invalid evaluation support or nonfinite error")
    scale = max(float(score_scale), 1e-6)
    return {
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "nmae": float(np.mean(np.abs(err) / scale)),
        "nmse": float(np.mean((err / scale) ** 2)),
    }
