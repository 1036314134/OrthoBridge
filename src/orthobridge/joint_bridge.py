"""Joint selection of two anchored donors for internal missing intervals.

Scan and tree search minimize the same finite pair objective. Tree pruning
uses conservative float64 guards, not a directed-rounding proof. Work
counters describe logical operations; they do not measure physical I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from numbers import Integral, Real
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class BridgeRepairResult:
    """Repaired query and accounting for joint donor selection."""

    values: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class _Node:
    minimum: np.ndarray
    maximum: np.ndarray
    left: int
    right: int
    rows: np.ndarray | None
    count: int


@dataclass(frozen=True)
class _Gap:
    start: int
    end: int
    left_context: np.ndarray
    right_context: np.ndarray
    left_query: np.ndarray
    right_query: np.ndarray
    left_costs: np.ndarray
    right_costs: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    left_value: float
    right_value: float


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _weight(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("compatibility_weight must be a finite nonnegative real number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("compatibility_weight must be a finite nonnegative real number")
    return value


def _query(values: np.ndarray, length: int) -> np.ndarray:
    query = np.asarray(values)
    if query.ndim != 1 or query.size != length or query.dtype.kind not in "iuf":
        raise ValueError("query must be a real vector matching the reference length")
    query = query.astype(np.float64, copy=True)
    if np.isinf(query).any() or np.isnan(query).all():
        raise ValueError("query must contain a finite observation and no infinity")
    if np.isnan(query[0]) or np.isnan(query[-1]):
        raise ValueError("joint bridge repair supports internal gaps only; edge gaps are unsupported")
    return query


def _finite(values: np.ndarray, operation: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{operation} overflow; rescale values with training statistics")
    return values


def _relative_interval(node: _Node, columns: np.ndarray, anchor: int, shift: float):
    """Enclose the computed shift + (value - anchor), including cancellation."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            low = shift + (node.minimum[columns] - node.maximum[anchor])
            high = shift + (node.maximum[columns] - node.minimum[anchor])
            scale = np.maximum(1.0, np.maximum(np.abs(node.minimum[columns]),
                                              np.abs(node.maximum[columns])))
            scale = np.maximum(scale, max(abs(float(node.minimum[anchor])),
                                          abs(float(node.maximum[anchor])), abs(shift)))
            # Original operand magnitudes matter when large levels cancel.
            margin = (64 * np.finfo(np.float64).eps) * scale
            low = np.nextafter(low - margin, -np.inf)
            high = np.nextafter(high + margin, np.inf)
    except FloatingPointError:
        return None
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        return None
    return low, high


def _interval_distance(low_a, high_a, low_b, high_b):
    """Downward-guarded mean squared distance between coordinate boxes."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            distance = np.maximum(0.0, np.maximum(low_a - high_b, low_b - high_a))
            scale = np.maximum(1.0, np.maximum(np.abs(low_a), np.abs(high_a)))
            scale = np.maximum(scale, np.maximum(np.abs(low_b), np.abs(high_b)))
            distance = np.maximum(0.0, distance - (64 * np.finfo(np.float64).eps) * scale)
            value = float(np.einsum("i,i->", distance, distance) / distance.size)
            guard = (64 * np.finfo(np.float64).eps * (distance.size + 4)
                     * max(1.0, abs(value)))
            lower = max(0.0, float(np.nextafter(value - guard, -np.inf)))
    except FloatingPointError:
        return None
    return lower if np.isfinite(lower) else None


class JointBridgeIndex:
    """Immutable complete reference bank with an optional lazy pair-search tree.

    Constructing the object validates and copies the bank. The coordinate-box
    tree is built only on the first nonempty tree query. Its preparation cost
    is reported separately, so scan queries never pay for tree construction.
    """

    def __init__(self, references: np.ndarray, *, leaf_size: int = 16):
        started = perf_counter()
        self.leaf_size = _positive_integer(leaf_size, "leaf_size")
        bank = np.asarray(references)
        if bank.ndim != 2 or min(bank.shape) == 0 or bank.dtype.kind not in "iuf":
            raise ValueError("references must be a nonempty real matrix")
        self._bank = bank.astype(np.float64, copy=True)
        if not np.isfinite(self._bank).all():
            raise ValueError("references must be complete and finite")
        self._bank.flags.writeable = False
        self.reference_count, self.length = self._bank.shape
        self._nodes: list[_Node] = []
        preparation = perf_counter() - started
        self.build_diagnostics = {
            "reference_preparation_seconds": preparation,
            "tree_build_seconds": 0.0, "build_seconds": preparation,
            "reference_copy_scalar_count": int(self._bank.size),
            "reference_bytes": int(self._bank.nbytes),
            "tree_nodes": 0, "summary_bytes": 0,
            "reference_count": self.reference_count, "leaf_size": self.leaf_size,
        }

    def _build(self, rows: np.ndarray) -> int:
        values = self._bank[rows]
        minimum, maximum = values.min(axis=0), values.max(axis=0)
        minimum.flags.writeable = maximum.flags.writeable = False
        node_id = len(self._nodes)
        self._nodes.append(_Node(minimum, maximum, -1, -1, None, rows.size))
        if rows.size <= self.leaf_size:
            rows.flags.writeable = False
            self._nodes[node_id] = _Node(minimum, maximum, -1, -1, rows, rows.size)
        else:
            scale = max(1.0, float(np.max(np.abs(values))))
            dimension = int(np.argmax(np.var(values / scale, axis=0)))
            ordered = rows[np.argsort(values[:, dimension], kind="stable")]
            middle = ordered.size // 2
            left = self._build(ordered[:middle])
            right = self._build(ordered[middle:])
            self._nodes[node_id] = _Node(minimum, maximum, left, right, None, rows.size)
        return node_id

    def _ensure_tree(self) -> float:
        if self._nodes:
            return 0.0
        started = perf_counter()
        self._build(np.arange(self.reference_count, dtype=np.int64))
        elapsed = perf_counter() - started
        self.build_diagnostics.update(
            tree_build_seconds=elapsed,
            build_seconds=self.build_diagnostics["reference_preparation_seconds"] + elapsed,
            tree_nodes=len(self._nodes),
            summary_bytes=sum(node.minimum.nbytes + node.maximum.nbytes
                              + (0 if node.rows is None else node.rows.nbytes)
                              for node in self._nodes),
        )
        return elapsed

    def _prepare_gap(self, query, observed, start, end, context_size, diagnostics):
        left, right = start - 1, end
        left_context = observed[observed < left][-context_size:]
        right_context = observed[observed > right][:context_size]
        costs, relative_queries = [], []
        try:
            with np.errstate(over="raise", invalid="raise"):
                for columns, anchor in ((left_context, left), (right_context, right)):
                    relative_query = query[columns] - query[anchor]
                    relative_queries.append(_finite(relative_query, "context preparation"))
                    if columns.size:
                        difference = (self._bank[:, columns] - self._bank[:, anchor, None]) - relative_query
                        cost = np.einsum("ij,ij->i", difference, difference) / columns.size
                        costs.append(_finite(cost, "context distance"))
                    else:
                        costs.append(np.zeros(self.reference_count, dtype=np.float64))
                    diagnostics["context_scalar_pairs"] += int(self.reference_count * columns.size)
                forward = query[left] + (self._bank[:, start:end] - self._bank[:, left, None])
                backward = query[right] + (self._bank[:, start:end] - self._bank[:, right, None])
                _finite(forward, "anchored continuation")
                _finite(backward, "anchored continuation")
        except FloatingPointError as error:
            raise ValueError("bridge preparation overflow; rescale values with training statistics") from error
        diagnostics["continuation_scalar_count"] += int(2 * forward.size)
        return _Gap(start, end, left_context, right_context, *relative_queries, *costs,
                    forward, backward, float(query[left]), float(query[right]))

    @staticmethod
    def _check_pair_range(gap, weight, diagnostics):
        """Reject unsafe all-pair arithmetic before either search backend.

        Coordinate extrema can be attained by different pairs, so this range
        check is deliberately conservative. It keeps rejection independent
        of whether a tree happens to prune an overflowing candidate.
        """
        diagnostics["range_check_scalar_values"] += int(gap.left_costs.size + gap.right_costs.size)
        try:
            with np.errstate(over="raise", invalid="raise"):
                upper = gap.left_costs.max() + gap.right_costs.max()
                if weight > 0:
                    maximum_difference = np.maximum(
                        gap.forward.max(axis=0) - gap.backward.min(axis=0),
                        gap.backward.max(axis=0) - gap.forward.min(axis=0),
                    )
                    compatibility_upper = (np.einsum("i,i->", maximum_difference, maximum_difference)
                                           / (gap.end - gap.start))
                    upper = upper + weight * compatibility_upper
                    diagnostics["range_check_scalar_values"] += int(4 * gap.forward.size
                                                                     + maximum_difference.size)
                _finite(np.asarray(upper), "pair arithmetic range")
        except FloatingPointError as error:
            raise ValueError("pair arithmetic range overflow; rescale values with training statistics") from error

    @staticmethod
    def _score_pairs(gap, left_rows, right_rows, weight, diagnostics, best):
        # Bound temporary pair-by-coordinate storage, without a Gram expansion.
        block = max(1, min(128, int(np.sqrt(262_144 / (gap.end - gap.start)))))
        for left_start in range(0, left_rows.size, block):
            rows_i = left_rows[left_start:left_start + block]
            for right_start in range(0, right_rows.size, block):
                rows_j = right_rows[right_start:right_start + block]
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        objectives = gap.left_costs[rows_i, None] + gap.right_costs[rows_j][None, :]
                        if weight > 0:
                            difference = gap.forward[rows_i, None, :] - gap.backward[rows_j][None, :, :]
                            compatibility = np.einsum("ijk,ijk->ij", difference, difference) / (gap.end - gap.start)
                            objectives = objectives + weight * compatibility
                            _finite(compatibility, "pair compatibility")
                            diagnostics["pair_coordinate_evaluations"] += int(difference.size)
                        else:
                            compatibility = None
                        _finite(objectives, "pair objective")
                except FloatingPointError as error:
                    raise ValueError("pair objective overflow; rescale values with training statistics") from error
                diagnostics["pair_evaluations"] += int(objectives.size)
                minimum = float(objectives.min())
                tied_i, tied_j = np.nonzero(objectives == minimum)
                order = np.lexsort((rows_j[tied_j], rows_i[tied_i]))
                i, j = int(tied_i[order[0]]), int(tied_j[order[0]])
                candidate = (minimum, int(rows_i[i]), int(rows_j[j]),
                             0.0 if compatibility is None else float(compatibility[i, j]))
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        return best

    def _node_info(self, node_id, gap, side, weight, diagnostics):
        node = self._nodes[node_id]
        if side == "left":
            columns, query = gap.left_context, gap.left_query
            anchor, shift = gap.start - 1, gap.left_value
        else:
            columns, query = gap.right_context, gap.right_query
            anchor, shift = gap.end, gap.right_value
        diagnostics["unary_node_bound_evaluations"] += 1
        diagnostics["bound_coordinate_evaluations"] += int(columns.size)
        unary = 0.0
        if columns.size:
            interval = _relative_interval(node, columns, anchor, 0.0)
            if interval is None:
                return None
            unary = _interval_distance(query, query, *interval)
            if unary is None:
                return None
        continuation = None
        if weight > 0:
            continuation = _relative_interval(node, np.arange(gap.start, gap.end), anchor, shift)
            diagnostics["bound_preparation_scalar_count"] += int(2 * (gap.end - gap.start))
            if continuation is None:
                return None
        return unary, continuation

    def _search(self, gap, weight, backend, diagnostics):
        if backend == "scan":
            rows = np.arange(self.reference_count, dtype=np.int64)
            return self._score_pairs(gap, rows, rows, weight, diagnostics, None)
        left_cache, right_cache = {}, {}

        def lower_bound(left_id, right_id):
            diagnostics["node_pair_bound_evaluations"] += 1
            if left_id not in left_cache:
                left_cache[left_id] = self._node_info(left_id, gap, "left", weight, diagnostics)
            if right_id not in right_cache:
                right_cache[right_id] = self._node_info(right_id, gap, "right", weight, diagnostics)
            left_info, right_info = left_cache[left_id], right_cache[right_id]
            if left_info is None or right_info is None:
                diagnostics["unreliable_bound_evaluations"] += 1
                return 0.0
            compatibility = 0.0
            if weight > 0:
                diagnostics["bound_coordinate_evaluations"] += gap.end - gap.start
                compatibility = _interval_distance(*left_info[1], *right_info[1])
                if compatibility is None:
                    diagnostics["unreliable_bound_evaluations"] += 1
                    return 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                value = left_info[0] + right_info[0] + weight * compatibility
                if not np.isfinite(value):
                    diagnostics["unreliable_bound_evaluations"] += 1
                    return 0.0
                guard = 128 * np.finfo(np.float64).eps * max(1.0, abs(value))
                value = max(0.0, float(np.nextafter(value - guard, -np.inf)))
            if not np.isfinite(value):
                diagnostics["unreliable_bound_evaluations"] += 1
                return 0.0
            return value

        pending = [(lower_bound(0, 0), 0, 0)]
        best = None
        dimensions = gap.left_context.size + gap.right_context.size + gap.end - gap.start
        while pending:
            lower, left_id, right_id = heapq.heappop(pending)
            diagnostics["node_pairs_visited"] += 1
            if best is not None:
                guard = 128 * np.finfo(np.float64).eps * (dimensions + 8) * max(1.0, abs(best[0]))
                if lower > best[0] + guard:
                    diagnostics["pruned_node_pairs"] += 1
                    continue
            left, right = self._nodes[left_id], self._nodes[right_id]
            if left.rows is not None and right.rows is not None:
                best = self._score_pairs(gap, left.rows, right.rows, weight, diagnostics, best)
            elif left.rows is None and (right.rows is not None or left.count >= right.count):
                for child in (left.left, left.right):
                    heapq.heappush(pending, (lower_bound(child, right_id), child, right_id))
            else:
                for child in (right.left, right.right):
                    heapq.heappush(pending, (lower_bound(left_id, child), left_id, child))
            diagnostics["max_pending_node_pairs"] = max(diagnostics["max_pending_node_pairs"], len(pending))
        return best

    def repair(self, query: np.ndarray, *, context_size: int = 32,
               compatibility_weight: float = 1.0, backend: str = "tree") -> BridgeRepairResult:
        """Exactly select an ordered donor pair for each internal gap.

        Contexts contain original observed coordinates strictly outside the
        two anchors; an empty context contributes zero cost. All gaps use the
        original query. Edge gaps reject the entire call. Zero compatibility
        weight skips compatibility arithmetic, including its diagnostic cost.
        """
        started = perf_counter()
        query = _query(query, self.length)
        context_size = _positive_integer(context_size, "context_size")
        weight = _weight(compatibility_weight)
        if not isinstance(backend, str) or backend not in ("scan", "tree"):
            raise ValueError("backend must be 'scan' or 'tree'")
        result = query.copy()
        missing = np.flatnonzero(np.isnan(query))
        observed = np.flatnonzero(~np.isnan(query))
        diagnostics = {
            "backend": backend, "reference_count": self.reference_count,
            "context_size": context_size, "compatibility_weight": weight,
            "gap_count": 0, "gaps": [], "pair_evaluations": 0,
            "node_pair_bound_evaluations": 0, "unary_node_bound_evaluations": 0,
            "node_pairs_visited": 0, "pruned_node_pairs": 0,
            "unreliable_bound_evaluations": 0, "max_pending_node_pairs": 0,
            "context_scalar_pairs": 0, "continuation_scalar_count": 0,
            "range_check_scalar_values": 0,
            "pair_coordinate_evaluations": 0, "bound_coordinate_evaluations": 0,
            "bound_preparation_scalar_count": 0, "fusion_scalar_count": 0,
            "preparation_seconds": 0.0, "search_seconds": 0.0,
            "tree_build_seconds_this_query": 0.0,
            "numerics": "float64_with_pruning_guards_not_formally_certified",
        }
        if missing.size:
            if backend == "tree":
                diagnostics["tree_build_seconds_this_query"] = self._ensure_tree()
            boundaries = np.flatnonzero(np.diff(missing) > 1) + 1
            groups = np.split(missing, boundaries)
            diagnostics["gap_count"] = len(groups)
            for positions in groups:
                prepare_started = perf_counter()
                gap = self._prepare_gap(query, observed, int(positions[0]), int(positions[-1] + 1),
                                        context_size, diagnostics)
                self._check_pair_range(gap, weight, diagnostics)
                preparation = perf_counter() - prepare_started
                diagnostics["preparation_seconds"] += preparation
                search_started = perf_counter()
                objective, i, j, compatibility = self._search(gap, weight, backend, diagnostics)
                search_seconds = perf_counter() - search_started
                diagnostics["search_seconds"] += search_seconds
                theta = (positions - (gap.start - 1)) / (gap.end - (gap.start - 1))
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        values = (1.0 - theta) * gap.forward[i] + theta * gap.backward[j]
                        result[positions] = _finite(values, "bridge fusion")
                except FloatingPointError as error:
                    raise ValueError("bridge fusion overflow; rescale values with training statistics") from error
                diagnostics["fusion_scalar_count"] += int(positions.size)
                diagnostics["gaps"].append({
                    "start": gap.start, "end": gap.end,
                    "left_anchor": gap.start - 1, "right_anchor": gap.end,
                    "left_context": gap.left_context.tolist(),
                    "right_context": gap.right_context.tolist(),
                    "selected_pair": [i, j], "objective": objective,
                    "left_cost": float(gap.left_costs[i]), "right_cost": float(gap.right_costs[j]),
                    "compatibility_cost": compatibility, "compatibility_evaluated": weight > 0,
                    "preparation_seconds": preparation, "search_seconds": search_seconds,
                })
        diagnostics["query_seconds"] = perf_counter() - started
        diagnostics["index_reference_preparation_seconds"] = self.build_diagnostics["reference_preparation_seconds"]
        diagnostics["index_tree_build_seconds"] = self.build_diagnostics["tree_build_seconds"]
        return BridgeRepairResult(result, diagnostics)


def joint_bridge_repair(query: np.ndarray, references: np.ndarray, *, context_size: int = 32,
                        compatibility_weight: float = 1.0) -> BridgeRepairResult:
    """Canonical exhaustive pair search, without constructing a tree."""
    return JointBridgeIndex(references).repair(query, context_size=context_size,
                                               compatibility_weight=compatibility_weight, backend="scan")
