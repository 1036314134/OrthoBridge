"""Run downstream 96-step forecasting on masked and naturally incomplete series (Sections 6.5 & 6.6)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orthobridge.baselines import ALL_SUPPORTED_METHODS, build_fitted_method
from orthobridge.forecasters import DEFAULT_MODEL_IDS, FoundationForecaster
from orthobridge.protocol import WINDOW_LENGTH, load_benchmark_series, make_masked_window, stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/downstream_forecast.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--section", choices=("6.5", "6.6", "both"), default="6.6")
    parser.add_argument("--forecasters", type=str, default="chronos2,timesfm3,moirai2_small")
    parser.add_argument("--methods", type=str, default="ortho_calibrated,linear,seasonal,graph_laplacian,observed_knn,seasonal_knn")
    parser.add_argument("--datasets", type=str, default="", help="Optional comma-separated dataset subset for Section 6.5")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/downstream_forecast")
    return parser.parse_args()


def future_metrics(pred: np.ndarray, future_truth: np.ndarray, score_scale: float) -> tuple[float, float]:
    finite = np.isfinite(future_truth)
    err = (pred[finite] - future_truth[finite]) / max(float(score_scale), 1e-6)
    return float(np.mean(np.abs(err))), float(np.mean(err ** 2))


def select_natural_test_origins(prep, stride: int = 608, cap: int = 30) -> list[int]:
    x = prep.series
    horizon = 96

    def eligible(origin: int) -> bool:
        ctx = x[origin - WINDOW_LENGTH:origin]
        fut = x[origin:origin + horizon]
        return (
            ctx.size == WINDOW_LENGTH
            and 0 < np.isnan(ctx).sum() < WINDOW_LENGTH
            and fut.size == horizon
            and np.isfinite(fut).any()
        )

    reg = [o for o in prep.test_origins if eligible(o)]
    extra = [
        o for o in range(prep.validation_end + WINDOW_LENGTH, len(x) - horizon + 1, stride)
        if o not in prep.test_origins and eligible(o)
    ]
    room = max(0, cap - len(reg))
    if len(extra) > room and room > 0:
        idx = np.linspace(0, len(extra) - 1, room, dtype=int)
        extra = [extra[i] for i in idx]
    else:
        extra = extra[:room]
    return reg + extra


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method {m!r}")
    forecaster_keys = [k.strip() for k in args.forecasters.split(",") if k.strip()]
    for fk in forecaster_keys:
        if fk not in DEFAULT_MODEL_IDS:
            raise ValueError(f"Unsupported forecaster {fk!r}")

    forecasters = {
        fk: FoundationForecaster(fk, model_id_or_path=cfg["forecasters"].get(fk), device=args.device)
        for fk in forecaster_keys
    }
    results = {}

    if args.section in ("6.6", "both"):
        print("=== Section 6.6: Natural Missingness Downstream 96-Step Forecasting ===")
        nat_scores: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for did in cfg["natural_missing_datasets"]:
            prep = load_benchmark_series(did, args.data_dir)
            origins = select_natural_test_origins(
                prep, stride=int(cfg["supplemental_test_stride"]), cap=int(cfg["max_natural_origins_per_dataset"])
            )
            if not origins:
                continue
            runners = {m: build_fitted_method(m, prep, device=args.device) for m in methods}
            for m, runner in runners.items():
                repaired_batch, futures = [], []
                for origin in origins:
                    raw_hist = prep.series[origin - WINDOW_LENGTH:origin].copy()
                    req_seed = stable_seed("natural", prep.dataset_id, int(origin))
                    rep_hist, _ = runner.impute_fn(raw_hist, req_seed)
                    repaired_batch.append(rep_hist)
                    futures.append(prep.series[origin:origin + 96].copy())
                histories = np.stack(repaired_batch)
                for fk, fc in forecasters.items():
                    preds = fc.predict(histories)
                    vals = [future_metrics(p, f, prep.score_scale) for p, f in zip(preds, futures)]
                    nat_scores[(fk, m)].append(
                        (statistics.fmean(v[0] for v in vals), statistics.fmean(v[1] for v in vals))
                    )
        sec66 = {}
        for m in methods:
            sec66[m] = {
                fk: {
                    "nmae": round(statistics.fmean(v[0] for v in nat_scores[(fk, m)]), 6),
                    "nmse": round(statistics.fmean(v[1] for v in nat_scores[(fk, m)]), 6),
                }
                for fk in forecaster_keys
                if nat_scores[(fk, m)]
            }
            print(f"{m:18s} -> {sec66[m]}")
        results["section_6_6"] = sec66

    if args.section in ("6.5", "both"):
        print("\n=== Section 6.5: Controlled Block & Tail Downstream 96-Step Forecasting ===")
        catalog = json.loads((args.data_dir / "datasets_metadata.json").read_text(encoding="utf-8"))
        dids = (
            [d.strip() for d in args.datasets.split(",") if d.strip()]
            if args.datasets
            else [d["dataset_id"] for d in catalog["datasets"] if d["in_principal_20"]]
        )
        sec65_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for did in dids:
            prep = load_benchmark_series(did, args.data_dir)
            if len(prep.bank) == 0:
                continue
            runners = {m: build_fitted_method(m, prep, device=args.device) for m in methods}
            for mode in ("block", "tail"):
                for m, runner in runners.items():
                    repaired_batch, futures = [], []
                    for seed in (42, 43, 44, 45, 46):
                        for rate in (0.1, 0.2, 0.3, 0.4, 0.5):
                            for origin in prep.test_origins:
                                fut = prep.series[origin:origin + 96]
                                if fut.size < 96 or not np.isfinite(fut).any():
                                    continue
                                win = make_masked_window(prep, origin=origin, mode=mode, rate=rate, seed=seed, split="test")
                                rep_hist, _ = runner.impute_fn(win.raw_masked, win.inference_seed)
                                repaired_batch.append(rep_hist)
                                futures.append(fut)
                    if not repaired_batch:
                        continue
                    histories = np.stack(repaired_batch)
                    for fk, fc in forecasters.items():
                        preds = fc.predict(histories)
                        vals = [future_metrics(p, f, prep.score_scale)[0] for p, f in zip(preds, futures)]
                        sec65_scores[(fk, m, mode)].append(statistics.fmean(vals))
        sec65 = {}
        for m in methods:
            sec65[m] = {}
            for fk in forecaster_keys:
                b_list = sec65_scores[(fk, m, "block")]
                t_list = sec65_scores[(fk, m, "tail")]
                if b_list and t_list:
                    b_mean, t_mean = statistics.fmean(b_list), statistics.fmean(t_list)
                    sec65[m][fk] = {"all_nmae": round(0.5 * (b_mean + t_mean), 6), "tail_nmae": round(t_mean, 6)}
            print(f"{m:18s} -> {sec65[m]}")
        results["section_6_5"] = sec65

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "downstream_forecast_summary.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved downstream forecast summary to {out_path}")


if __name__ == "__main__":
    main()
