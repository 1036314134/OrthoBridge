"""Run component ablation and paired scan-vs-tree index efficiency (Section 6.4)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orthobridge import OrthoBridgeIndex
from orthobridge.baselines import build_fitted_method
from orthobridge.protocol import compute_metrics, load_benchmark_series, make_masked_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation_efficiency.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--modes", type=str, default="block,tail", help="Comma-separated ablation modes (e.g., block,tail or block,tail,point)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ablation_efficiency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    series_list = [load_benchmark_series(did, args.data_dir) for did in cfg["diagnostic_datasets"]]

    print("=== 1. Component Ablation (Validation Diagnostic Panel) ===")
    ablation_scores: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for prep in series_list:
        runners = {m: build_fitted_method(m, prep, backend="auto") for m in cfg["ablation_methods"]}
        for mode in modes:
            ds_by_m: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for rate in cfg["ablation_rates"]:
                for origin in prep.validation_origins:
                    win = make_masked_window(prep, origin=origin, mode=mode, rate=rate, seed=42, split="validation")
                    if not win.hidden.any() or not win.observed.any():
                        continue
                    for m, runner in runners.items():
                        out, _ = runner.impute_fn(win.raw_masked, win.inference_seed)
                        sc = compute_metrics(out, win.truth, win.hidden, prep.score_scale)
                        ds_by_m[m].append((sc["nmae"], sc["nmse"]))
            for m, vals in ds_by_m.items():
                ablation_scores[(m, mode)].append(
                    (statistics.fmean(v[0] for v in vals), statistics.fmean(v[1] for v in vals))
                )

    ablation_summary = {}
    print(f"{'Variant':20s}  {'Block NMAE/NMSE':>18s}  {'Tail NMAE/NMSE':>18s}")
    for m in cfg["ablation_methods"]:
        b_vals = ablation_scores[(m, "block")]
        t_vals = ablation_scores[(m, "tail")]
        b_nmae, b_nmse = statistics.fmean(v[0] for v in b_vals), statistics.fmean(v[1] for v in b_vals)
        t_nmae, t_nmse = statistics.fmean(v[0] for v in t_vals), statistics.fmean(v[1] for v in t_vals)
        entry = {
            "block_nmae": round(b_nmae, 6), "block_nmse": round(b_nmse, 6),
            "tail_nmae": round(t_nmae, 6), "tail_nmse": round(t_nmse, 6),
        }
        if (m, "point") in ablation_scores and ablation_scores[(m, "point")]:
            entry["point_nmae"] = round(statistics.fmean(v[0] for v in ablation_scores[(m, "point")]), 6)
        ablation_summary[m] = entry
        print(f"{m:20s}  {b_nmae:8.4f} / {b_nmse:7.4f}  {t_nmae:8.4f} / {t_nmse:7.4f}")

    print("\n=== 2. Paired Scan vs. Tree Efficiency (r=0.3, exact parity check) ===")
    repeats = int(cfg["efficiency_repeats"])
    rate = float(cfg["efficiency_rate"])
    timing_summary = {}
    for mode in ("block", "tail", "point"):
        scan_totals, tree_totals = [], []
        eval_scan, eval_tree = 0, 0
        max_diff = 0.0
        prebuilt = []
        for prep in series_list:
            idx = OrthoBridgeIndex(prep.bank, period=prep.period)
            idx._ensure_tree()
            if mode == "tail":
                pass
            origin = prep.validation_origins[0]
            win = make_masked_window(prep, origin=origin, mode=mode, rate=rate, seed=42, split="validation")
            prebuilt.append((idx, win))

        for rep_id in range(repeats):
            t0 = perf_counter()
            s_outs = [idx.repair(win.normalized_masked, backend="scan") for idx, win in prebuilt]
            scan_totals.append(perf_counter() - t0)

            t1 = perf_counter()
            t_outs = [idx.repair(win.normalized_masked, backend="tree") for idx, win in prebuilt]
            tree_totals.append(perf_counter() - t1)

            if rep_id == 0:
                for sr, tr in zip(s_outs, t_outs):
                    max_diff = max(max_diff, float(np.max(np.abs(sr.values - tr.values))))
                    eval_scan += sr.diagnostics["pair_evaluations"]
                    eval_tree += tr.diagnostics["pair_evaluations"]

        pruning = (1.0 - eval_tree / eval_scan) * 100.0 if eval_scan > 0 else 0.0
        timing_summary[mode] = {
            "scan_median_s": round(statistics.median(scan_totals), 6),
            "tree_median_s": round(statistics.median(tree_totals), 6),
            "pair_pruning_pct": round(pruning, 4) if eval_scan > 0 else None,
            "max_abs_diff": max_diff,
        }
        print(
            f"mode={mode:5s}  scan={timing_summary[mode]['scan_median_s']:.4f}s  "
            f"tree={timing_summary[mode]['tree_median_s']:.4f}s  "
            f"pair_pruning={pruning:6.2f}%  max_diff={max_diff:.1f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "ablation_efficiency_summary.json"
    out_file.write_text(
        json.dumps({"ablation": ablation_summary, "efficiency": timing_summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved ablation & efficiency results to {out_file}")


if __name__ == "__main__":
    main()
