"""OrthoBridge: Drift-Free Orthogonal Gauss-Markov Bridging with Exact Indexing."""

from .joint_bridge import BridgeRepairResult, JointBridgeIndex, joint_bridge_repair
from .ortho_bridge import (
    OrthoBridgeIndex,
    OrthoBridgeRepairResult,
    ortho_baseline,
    ortho_bridge_repair,
)
from .protocol import (
    DEFAULT_CALIBRATION_SHRINKAGES,
    MODES,
    RATES,
    TEST_SEEDS,
    PreparedSeries,
    PreparedWindow,
    calibrate_shrinkage,
    compute_metrics,
    load_benchmark_series,
    make_masked_window,
    restore_raw_scale,
    select_shrinkage,
    training_starts,
)

__all__ = [
    "OrthoBridgeIndex",
    "OrthoBridgeRepairResult",
    "ortho_baseline",
    "ortho_bridge_repair",
    "JointBridgeIndex",
    "BridgeRepairResult",
    "joint_bridge_repair",
    "DEFAULT_CALIBRATION_SHRINKAGES",
    "MODES",
    "RATES",
    "TEST_SEEDS",
    "PreparedSeries",
    "PreparedWindow",
    "calibrate_shrinkage",
    "compute_metrics",
    "load_benchmark_series",
    "make_masked_window",
    "restore_raw_scale",
    "select_shrinkage",
    "training_starts",
]
