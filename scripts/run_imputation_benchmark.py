"""Run the 20-dataset missing-value imputation benchmark (Sections 6.2 & 6.3)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orthobridge.baselines import ALL_SUPPORTED_METHODS, PAPER_14_METHODS, build_fitted_method
from orthobridge.protocol import (
    DEFAULT_CALIBRATION_SHRINKAGES,
    RATES,
    TEST_SEEDS,
    calibrate_shrinkage,
    compute_metrics,
    load_benchmark_series,
    make_masked_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Directory containing datasets_metadata.json and .npz")
    parser.add_argument("--datasets", type=str, default="principal20", help="Comma-separated dataset IDs/names, or 'principal20'")
    parser.add_argument("--methods", type=str, default="non_neural", help="Comma-separated methods, 'all14', or 'non_neural'")
    parser.add_argument("--modes", type=str, default="block,tail", help="Comma-separated mask modes (block,tail,point)")
    parser.add_argument("--rates", type=str, default="0.1,0.2,0.3,0.4,0.5", help="Comma-separated missing rates")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated test evaluation seeds")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--backend", choices=("auto", "tree", "scan"), default="auto")
    parser.add_argument("--recalibrate", action="store_true", help="Re-run training-prefix shrinkage calibration instead of frozen defaults")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device for neural baselines (e.g., cpu or cuda:0)")
    parser.add_argument("--neural-epochs", type=int, default=50, help="Training epochs for neural baselines")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/imputation_benchmark")
    return parser.parse_args()


def resolve_datasets(spec: str, data_dir: Path) -> list[str]:
    catalog = json.loads((data_dir / "datasets_metadata.json").read_text(encoding="utf-8"))
    if spec == "principal20":
        return [d["dataset_id"] for d in catalog["datasets"] if d["in_principal_20"]]
    return [item.strip() for item in spec.split(",") if item.strip()]


def resolve_methods(spec: str) -> list[str]:
    if spec == "all14":
        return list(PAPER_14_METHODS)
    if spec == "non_neural":
        return [
            "ortho_calibrated", "linear", "seasonal", "hankel_softimpute",
            "graph_laplacian", "observed_knn", "kdtree_knn", "seasonal_knn",
        ]
    chosen = [m.strip() for m in spec.split(",") if m.strip()]
    for m in chosen:
        if m not in ALL_SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {m!r}; valid options: {ALL_SUPPORTED_METHODS}")
    return chosen


def main() -> None:
    args = parse_args()
    dataset_keys = resolve_datasets(args.datasets, args.data_dir)
    methods = resolve_methods(args.methods)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.split == "test" else [42]

    prepared_list = [load_benchmark_series(k, args.data_dir) for k in dataset_keys]
    shrinkage_map = calibrate_shrinkage(prepared_list) if args.recalibrate else dict(DEFAULT_CALIBRATION_SHRINKAGES)

    # Hierarchical aggregation: mean over origins -> seeds -> rates -> modes -> equal datasets
    origin_Buckets: dict[tuple[str, str, str, float, int], list[tuple[float, float]]] = defaultdict(list)

    for prep in prepared_list:
        origins = prep.test_origins if args.split == "test" else prep.validation_origins
        if not origins or len(prep.bank) == 0:
            continue
        fitted = {
            m: build_fitted_method(
                m, prep, backend=args.backend, shrinkage_map=shrinkage_map,
                device=args.device, neural_epochs=args.neural_epochs,
            )
            for m in methods
        }
        print(f"[{prep.paper_name:13s}] T={len(prep.series):8d}, P={prep.period:5d}, donors={len(prep.bank):3d}, origins={len(origins)}")
        for seed in seeds:
            for mode in modes:
                for rate in rates:
                    for origin in origins:
                        win = make_masked_window(prep, origin=origin, mode=mode, rate=rate, seed=seed, split=args.split)
                        if not win.hidden.any() or not win.observed.any():
                            continue
                        for m, runner in fitted.items():
                            if m == "joint_bridge" and (np.isnan(win.raw_masked[0]) or np.isnan(win.raw_masked[-1])):
                                continue
                            imputed, _ = runner.impute_fn(win.raw_masked, win.inference_seed)
                            sc = compute_metrics(imputed, win.truth, win.hidden, prep.score_scale)
                            origin_Buckets[(m, prep.dataset_id, mode, rate, seed)].append((sc["nmae"], sc["nmse"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    print("\n=== Aggregate Benchmark Summary ===")
    print(f"{'Method':20s}  {'Overall NMAE':>12s}  {'Overall NMSE':>12s}  {'Block NMAE':>10s}  {'Tail NMAE':>10s}")
    for m in methods:
        mode_nmae, mode_nmse = {}, {}
        for mode in modes:
            ds_nmae, ds_nmse = [], []
            for prep in prepared_list:
                rate_nmae, rate_nmse = [], []
                for rate in rates:
                    seed_nmae, seed_nmse = [], []
                    for seed in seeds:
                        vals = origin_Buckets.get((m, prep.dataset_id, mode, rate, seed), [])
                        if vals:
                            seed_nmae.append(statistics.fmean(v[0] for v in vals))
                            seed_nmse.append(statistics.fmean(v[1] for v in vals))
                    if seed_nmae:
                        rate_nmae.append(statistics.fmean(seed_nmae))
                        rate_nmse.append(statistics.fmean(seed_nmse))
                if rate_nmae:
                    ds_nmae.append(statistics.fmean(rate_nmae))
                    ds_nmse.append(statistics.fmean(rate_nmse))
            if ds_nmae:
                mode_nmae[mode] = statistics.fmean(ds_nmae)
                mode_nmse[mode] = statistics.fmean(ds_nmse)
        overall_nmae = statistics.fmean(mode_nmae.values()) if mode_nmae else float("nan")
        overall_nmse = statistics.fmean(mode_nmse.values()) if mode_nmse else float("nan")
        row = {
            "method": m,
            "overall_nmae": round(overall_nmae, 6),
            "overall_nmse": round(overall_nmse, 6),
            "block_nmae": round(mode_nmae.get("block", float("nan")), 6),
            "block_nmse": round(mode_nmse.get("block", float("nan")), 6),
            "tail_nmae": round(mode_nmae.get("tail", float("nan")), 6),
            "tail_nmse": round(mode_nmse.get("tail", float("nan")), 6),
        }
        summary_rows.append(row)
        print(
            f"{m:20s}  {row['overall_nmae']:12.4f}  {row['overall_nmse']:12.4f}  "
            f"{row['block_nmae']:10.4f}  {row['tail_nmae']:10.4f}"
        )

    csv_path = args.output_dir / "summary_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps({"split": args.split, "shrinkage_map": shrinkage_map, "results": summary_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved summary table to {csv_path}")


if __name__ == "__main__":
    main()
