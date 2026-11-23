"""Unified registry for OrthoBridge, ablation controls, JointBridge, and all 13 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..joint_bridge import JointBridgeIndex
from ..ortho_bridge import OrthoBridgeIndex
from ..protocol import DEFAULT_CALIBRATION_SHRINKAGES, PreparedSeries, restore_raw_scale, select_shrinkage
from .classical import GraphLaplacianImputer, HankelSoftImpute, linear_impute, seasonal_impute
from .neural import NEURAL_METHODS, NeuralBaselineImputer
from .retrieval import retrieval_impute

PAPER_14_METHODS = (
    "ortho_calibrated",
    "linear",
    "seasonal",
    "hankel_softimpute",
    "graph_laplacian",
    "observed_knn",
    "kdtree_knn",
    "seasonal_knn",
    "brits",
    "csdi",
    "saits",
    "imputeformer",
    "timemixer",
    "helix",
)

ABLATION_METHODS = (
    "ortho_base",
    "ortho_top1_hard",
    "ortho_fixed",
    "ortho_calibrated",
)

ALL_SUPPORTED_METHODS = PAPER_14_METHODS + ("ortho_base", "ortho_top1_hard", "ortho_fixed", "joint_bridge")


@dataclass
class FittedMethod:
    name: str
    impute_fn: Callable[[np.ndarray, int], tuple[np.ndarray, dict]]


def build_fitted_method(
    method: str,
    prepared: PreparedSeries,
    *,
    backend: str = "auto",
    shrinkage_map: dict[str, float] | None = None,
    device: str = "cpu",
    neural_epochs: int = 50,
    seed: int = 42,
) -> FittedMethod:
    """Prepare a method strictly on the training prefix of ``prepared``."""
    if method not in ALL_SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method!r}")

    prefix = prepared.series[:prepared.fit_end]
    c, s = prepared.center, prepared.repair_scale

    if method in ("ortho_base", "ortho_top1_hard", "ortho_fixed", "ortho_calibrated"):
        index = OrthoBridgeIndex(prepared.bank, period=prepared.period)

        def _run_ortho(raw_masked: np.ndarray, _req_seed: int = 42):
            norm_q = (raw_masked - c) / s
            if method == "ortho_base":
                rep = index.repair(norm_q, shrinkage=0.0, backend=backend)
            elif method == "ortho_top1_hard":
                rep = index.repair(
                    norm_q, anchor_window=1, anchor_decay=0.0, k_pairs=1, k=1,
                    shrinkage=0.75, backend=backend,
                )
            elif method == "ortho_fixed":
                rep = index.repair(norm_q, shrinkage=0.75, backend=backend)
            else:
                lam = select_shrinkage(raw_masked, shrinkage_map or DEFAULT_CALIBRATION_SHRINKAGES)
                rep = index.repair(norm_q, shrinkage=lam, backend=backend)
            out = restore_raw_scale(rep.values, norm_q, raw_masked, c, s)
            return out, rep.diagnostics

        return FittedMethod(method, _run_ortho)

    if method == "joint_bridge":
        jb_index = JointBridgeIndex(prepared.bank)

        def _run_jb(raw_masked: np.ndarray, _req_seed: int = 42):
            norm_q = (raw_masked - c) / s
            rep = jb_index.repair(norm_q, backend="tree" if backend != "scan" else "scan")
            out = restore_raw_scale(rep.values, norm_q, raw_masked, c, s)
            return out, rep.diagnostics

        return FittedMethod(method, _run_jb)

    if method == "linear":
        return FittedMethod(method, lambda q, _s=42: (linear_impute(q, fallback_mean=c), {}))

    if method == "seasonal":
        return FittedMethod(method, lambda q, _s=42: (seasonal_impute(q, period=prepared.period, fallback_mean=c), {}))

    if method == "hankel_softimpute":
        model = HankelSoftImpute().fit(prefix)
        return FittedMethod(method, lambda q, _s=42: (model.impute(q), {}))

    if method == "graph_laplacian":
        model = GraphLaplacianImputer(period=prepared.period).fit(prefix)
        return FittedMethod(method, lambda q, _s=42: (model.impute(q), {}))

    if method in ("observed_knn", "kdtree_knn", "seasonal_knn"):
        def _run_ret(raw_masked: np.ndarray, _req_seed: int = 42):
            norm_q = (raw_masked - c) / s
            rep = retrieval_impute(norm_q, prepared.bank, method, period=prepared.period)
            return restore_raw_scale(rep, norm_q, raw_masked, c, s), {}

        return FittedMethod(method, _run_ret)

    if method in NEURAL_METHODS:
        net = NeuralBaselineImputer(
            method, epochs=neural_epochs, device=device, seed=seed
        ).fit(prefix, prepared.all_training_starts)
        return FittedMethod(method, lambda q, req_s=42: (net.impute(q, request_seed=req_s), {}))

    raise AssertionError(f"Unhandled method {method}")
