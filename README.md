# OrthoBridge

Reference implementation for **OrthoBridge: Drift-Free Orthogonal Gauss–Markov Bridging with Exact Branch-and-Bound Indexing for Time Series Imputation**.

`OrthoBridge` repairs interior (`block`), boundary (`tail` and `leading`), and isolated (`point`) missing intervals in univariate time series windows from complete historical segments of the same series without gradient training, while preserving every observed value bit-for-bit in IEEE 754 `float64`.

It combines four components:
1. **Drift-free ANOVA empirical-Bayes (`ANOVA-EB`) periodic decomposition**, mapping phase-shifted historical windows into a shared zero-mean anomaly bank $Z$ with fitted lag-1 persistence $\phi$.
2. **Soft-anchored discrete Ornstein–Uhlenbeck (`OU`) bridging**, unifying two-anchor ($|\partial G| = 2$) and single-anchor ($|\partial G| = 1$) intervals under closed-form conditional expectations and cumulative step variances.
3. **Calibrated multi-donor ensemble fusion**, using variance-normalized compatibility and row-discounted Gibbs weights on two-anchor gaps, and multi-anchor causal prefix replay with consensus disagreement damping on single-anchor boundary gaps.
4. **Exact multi-tier branch-and-bound indexing**, integrating PAA-guided median splits, same-row lag and step-difference envelopes, and exact unary shortlists to prune non-competitive donor pairs while guaranteeing identical lexicographic top-$K_2$ rankings as exhaustive enumeration under exact real arithmetic.

---

## Installation

Python 3.10 or later is required. Run commands from the repository root.

For `OrthoBridge`, `JointBridge`, quickstart demos, and pure-NumPy baselines (`Linear`, `Seasonal`, `Hankel-SoftImpute`, `Observed-kNN`, `Seasonal-kNN`):

```bash
python -m pip install -e .
```

For SciPy-backed classical baselines (`Graph-Laplacian`, `KDTree-kNN`):

```bash
python -m pip install -e ".[baselines]"
```

For deep neural imputation baselines (`BRITS`, `CSDI`, `SAITS`, `ImputeFormer`, `TimeMixer`, `HELIX`):

```bash
python -m pip install -e ".[neural]"
```

For downstream 96-step foundation-model forecasting (`Chronos-2`, `TimesFM3`, `Moirai-2-Small`):

```bash
python -m pip install -e ".[forecast]"
```

---

## Repository Structure

```text
configs/
    benchmark_20datasets.json       Main 20-dataset evaluation protocol & hyperparameters
    ablation_efficiency.json        Component ablation & scan-vs-tree index timing settings
    downstream_forecast.json        Downstream & natural-missingness 96-step forecast settings
data/
    orthobridge_datasets_uv512.npz  Compressed float64 benchmark series (20 principal + Beijing-Air)
    datasets_metadata.json          Split cutoffs, periods, frozen origins, scales & citations
examples/
    quickstart_demo.py              Synthetic geometry parity check & real ERCOT window example
scripts/
    run_imputation_benchmark.py     Main 20-dataset imputation evaluation (Sections 6.2 & 6.3)
    run_ablation_and_efficiency.py  Ablation study & paired scan/tree pruning benchmark (Section 6.4)
    run_downstream_forecast.py      Downstream & natural-missingness 96-step forecasting (Sections 6.5 & 6.6)
src/orthobridge/
    ortho_bridge.py                 Core OrthoBridgeIndex, multi-tier range tree & OU bridge operator
    joint_bridge.py                 Hard-anchored single-pair JointBridge control (internal gaps)
    protocol.py                     Chronological 70/10/20 loader, masking, calibration & metrics
    forecasters.py                  Foundation forecaster wrappers (Chronos-2, TimesFM3, Moirai-2)
    baselines/
        classical.py                Linear, Seasonal, Hankel-SoftImpute, Graph-Laplacian
        retrieval.py                Observed-kNN, KDTree-kNN, Seasonal-kNN
        neural.py                   BRITS, CSDI, SAITS, ImputeFormer, TimeMixer, HELIX
        timemixer/                  Vendored TimeMixer model & multiscale mixing layers
```

---

## Quickstart

Run the self-contained demonstration verifying `scan` vs. `tree` exactness across `block`, `tail`, `leading`, and `point` missingness as well as an `ERCOT` benchmark window:

```bash
python examples/quickstart_demo.py
```

Or invoke `OrthoBridgeIndex` directly in Python:

```python
import numpy as np
from orthobridge import OrthoBridgeIndex

rng = np.random.default_rng(42)
t = np.arange(512)
bank = np.sin(2 * np.pi * (t + np.arange(128)[:, None]) / 24) + 0.1 * rng.normal(size=(128, 512))

query = np.sin(2 * np.pi * (t + 7) / 24) + 0.05 * rng.normal(size=512)
query[180:240] = np.nan   # two-anchor interior gap
query[460:] = np.nan      # single-anchor trailing gap

index = OrthoBridgeIndex(bank, period=24)
result = index.repair(query, shrinkage=0.75, backend="auto")

assert np.isfinite(result.values).all()
obs = np.isfinite(query)
assert np.array_equal(result.values[obs].view("u8"), query[obs].view("u8"))
```

---

## Benchmark Datasets

All 21 real-world univariate series used in the paper (`20` principal datasets across eight domains plus `Beijing-Air` for natural-missingness evaluation) are bundled in `data/orthobridge_datasets_uv512.npz` (~10.0 MB compressed) alongside `data/datasets_metadata.json`.

Each series is split chronologically into `70%` training, `10%` validation, and `20%` test. Up to $N \le 512$ complete length-512 historical donor windows are extracted strictly inside the training fitting prefix $t_{\mathrm{fit}} = \lfloor 0.7T \rfloor - 512$.

| Paper Name | Dataset ID | Domain | Length $T$ | Freq. / $P_{\mathrm{meta}}$ | Donors $N$ | Source Reference |
|---|---|---|---:|---|---:|---|
| `ETT-h1` | `ett_hourly_ot` | Power Grid | 17,420 | 1h / 24 | 512 | Zhou et al., AAAI 2021 (Informer) |
| `GEFCom17` | `fev_proenfo_gfc17` | Power Grid | 17,544 | 1h / 24 | 512 | Hong et al., IJF 2019 (GEFCom2017) |
| `ERCOT` | `fev_ercot_1h` | Power Grid | 154,872 | 1h / 24 | 480 | Shchur et al., arXiv:2509.26468 (fev-bench) |
| `Aus-Demand` | `monash_australian_electricity_demand` | Power Grid | 230,784 | 30min / 48 | 512 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `SDWPF` | `fev_kdd_cup_2022_30t` | Wind Energy | 11,758 | 30min / 48 | 434 | Zhou et al., Sci. Data 2024 (SDWPF) |
| `Solar-1h` | `fev_solar_with_weather_1h` | Solar Energy | 49,648 | 1h / 24 | 512 | Shchur et al., arXiv:2509.26468 (fev-bench) |
| `Wind-1m` | `australian_wind_farms_minutely` | Wind Energy | 527,040 | 1min / 1440 | 504 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `Solar-4s` | `monash_solar_4_seconds` | Solar Energy | 7,397,222 | 4s / 21600 | 512 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `Jena-Weather` | `jena_weather_hourly` | Meteorology | 8,784 | 1h / 24 | 461 | Aksu et al., arXiv:2410.10393 (GIFT-Eval) |
| `KDD-Air` | `monash_kdd_cup_2018` | Air Quality | 10,898 | 1h / 24 | 2 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `Oikolab` | `monash_oikolab_weather` | Meteorology | 100,057 | 1h / 24 | 512 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `Seattle-Loop` | `seattle_loop_hourly` | Traffic | 8,760 | 1h / 24 | 512 | Cui et al., IEEE TITS 2020 |
| `Pedestrian` | `monash_pedestrian_counts` | Mobility | 26,443 | 1h / 24 | 512 | Godahewa et al., NeurIPS D&B 2021 (Monash) |
| `Cockatoo` | `fev_proenfo_cockatoo` | Smart Building | 17,544 | 1h / 24 | 512 | Miller et al., Sci. Data 2020 (BDG2) |
| `Ocean-Vel` | `time_current_velocity_h` | Oceanography | 5,327 | 1h / 24 | 512 | Qiao et al., arXiv:2602.12147 (TIME) |
| `Coastal-TS` | `time_coastal_t_s_h` | Oceanography | 6,673 | 1h / 24 | 512 | Qiao et al., arXiv:2602.12147 (TIME) |
| `Exchange` | `exchange_rate` | Finance | 7,588 | 1d / 5 | 512 | Lai et al., SIGIR 2018 |
| `M4-Daily` | `m4_daily` | Macroeconomics | 8,533 | 1d / 7 | 512 | Makridakis et al., IJF 2020 (M4) |
| `BizIT-App` | `gift_bizitobs_application` | Cloud IT | 8,834 | 10s / 8640 | 512 | Aksu et al., arXiv:2410.10393 (GIFT-Eval) |
| `THU-Motor` | `mcc5_thu_motor` | Industrial | 1,152,000 | step / 1 | 512 | Chen et al., arXiv:2601.02278 (MCC5-THU) |
| `Beijing-Air`* | `lotsa_beijing_air_quality` | Air Quality | 35,064 | 1h / 24 | 61 | Woo et al., ICML 2024 (LOTSA) |

*\*Evaluated in Section 6.6 on pre-existing natural sensor dropouts without artificial masking.*

---

## Reproducing Paper Experiments

### 1. Main 20-Dataset Imputation Benchmark (Sections 6.2 & 6.3)

Run `OrthoBridge` and all non-neural baselines across the 20 datasets, 5 test seeds (`42..46`), and missing rates `10%..50%` under `block` and `tail` modes:

```bash
python scripts/run_imputation_benchmark.py \
    --datasets principal20 \
    --methods non_neural \
    --modes block,tail \
    --rates 0.1,0.2,0.3,0.4,0.5 \
    --seeds 42,43,44,45,46 \
    --output-dir outputs/imputation_benchmark
```

To evaluate all 14 methods including the 6 neural baselines (`brits`, `csdi`, `saits`, `imputeformer`, `timemixer`, `helix`), install `.[neural]` and pass `--methods all14 --device cuda:0`. To include the paired two-anchor `JointBridge` control, include `joint_bridge` in `--methods`.

### 2. Component Ablation & Index Pruning Efficiency (Section 6.4)

Run the ablation study (`ortho_base`, `ortho_top1_hard`, `ortho_fixed`, `ortho_calibrated`) and paired 5-repeat `scan` vs. `tree` runtime and candidate-pair pruning benchmark:

```bash
python scripts/run_ablation_and_efficiency.py \
    --output-dir outputs/ablation_efficiency
```

### 3. Downstream & Real-Missingness 96-Step Forecasting (Sections 6.5 & 6.6)

After installing `.[forecast]` (and having HuggingFace access or local model paths configured in `configs/downstream_forecast.json` for `amazon/chronos-2`, `google/timesfm-2.5-200m-pytorch`, and `Salesforce/moirai-2.0-R-small`), run downstream 96-step forecasting:

```bash
# Section 6.6: Natural sensor missingness (zero artificial masking)
python scripts/run_downstream_forecast.py \
    --section 6.6 \
    --forecasters chronos2,timesfm3,moirai2_small \
    --device cuda:0 \
    --output-dir outputs/downstream_forecast

# Section 6.5: Controlled block & tail missingness across the 20 datasets
python scripts/run_downstream_forecast.py \
    --section 6.5 \
    --forecasters chronos2,timesfm3,moirai2_small \
    --device cuda:0 \
    --output-dir outputs/downstream_forecast
```
