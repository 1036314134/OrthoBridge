"""Quickstart demonstration of OrthoBridge on synthetic and real benchmark windows."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orthobridge import (
    OrthoBridgeIndex,
    compute_metrics,
    load_benchmark_series,
    make_masked_window,
    restore_raw_scale,
    select_shrinkage,
)


def run_synthetic_verification() -> None:
    rng = np.random.default_rng(107)
    t = np.arange(96)
    phases = np.arange(18)[:, None]
    bank = (
        2.0 * np.sin(2.0 * np.pi * (t + phases) / 12.0)
        + 0.02 * t
        + rng.normal(0.0, 0.12, (18, t.size)).cumsum(axis=1)
    )
    truth = 2.0 * np.sin(2.0 * np.pi * (t + 2) / 12.0) + 0.018 * t + 0.3 * np.cos(t / 9.0)
    index = OrthoBridgeIndex(bank, period=12, leaf_size=3)

    print("=== 1. Synthetic Mask Geometries & Scan/Tree Exactness ===")
    print("geometry   gaps   masked MAE   scan==tree   candidate evals (scan -> tree)")
    for name, mask in (
        ("block", slice(30, 57)),
        ("tail", slice(70, None)),
        ("leading", slice(None, 26)),
        ("point", [10, 24, 47, 80]),
    ):
        query = truth.copy()
        query[mask] = np.nan
        scan_res = index.repair(query, backend="scan")
        tree_res = index.repair(query, backend="tree")
        np.testing.assert_array_equal(scan_res.values, tree_res.values)
        observed = np.isfinite(query)
        np.testing.assert_array_equal(tree_res.values[observed].view("u8"), query[observed].view("u8"))
        mae = float(np.mean(np.abs(tree_res.values[mask] - truth[mask])))
        work_scan = scan_res.diagnostics["pair_evaluations"] + scan_res.diagnostics["one_evaluations"]
        work_tree = tree_res.diagnostics["pair_evaluations"] + tree_res.diagnostics["one_evaluations"]
        print(
            f"{name:8s}   {tree_res.diagnostics['gap_count']:4d}   {mae:10.6f}   "
            f"exact        {work_scan:4d} -> {work_tree:4d}"
        )


def run_real_benchmark_sample() -> None:
    print("\n=== 2. Real Benchmark Window (ERCOT Power-Grid Load, L=512) ===")
    prep = load_benchmark_series("ERCOT")
    index = OrthoBridgeIndex(prep.bank, period=prep.period)
    origin = prep.test_origins[0]
    for mode in ("block", "tail"):
        win = make_masked_window(prep, origin=origin, mode=mode, rate=0.3, seed=42, split="test")
        lam = select_shrinkage(win.raw_masked)
        rep = index.repair(win.normalized_masked, shrinkage=lam, backend="auto")
        imputed = restore_raw_scale(rep.values, win.normalized_masked, win.raw_masked, prep.center, prep.repair_scale)
        scores = compute_metrics(imputed, win.truth, win.hidden, prep.score_scale)
        pruned = ""
        if mode == "block" and len(prep.bank) > 0:
            total_pairs = (len(prep.bank) + 1) ** 2
            ratio = 1.0 - rep.diagnostics["pair_evaluations"] / total_pairs
            pruned = f"  pair_pruning={ratio * 100:.2f}%"
        print(
            f"ERCOT [{mode:5s}, r=30%]  NMAE={scores['nmae']:.4f}  NMSE={scores['nmse']:.4f}  "
            f"time={rep.diagnostics['query_seconds'] * 1000:.2f}ms{pruned}"
        )


if __name__ == "__main__":
    run_synthetic_verification()
    run_real_benchmark_sample()
