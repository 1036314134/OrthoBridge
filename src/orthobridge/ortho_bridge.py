"""Seasonally adjusted OU bridges over complete historical references.

Both searches rank the same finite objectives. Tree lower bounds are real-
arithmetic inequalities evaluated with float64 slack, not certified intervals.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import heapq
from numbers import Integral
from time import perf_counter

import numpy as np

from ._validation import _integer, _query, _references, _scalar


@dataclass(frozen=True)
class OrthoBridgeRepairResult:
    values: np.ndarray
    diagnostics: dict


@contextmanager
def _arithmetic():
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            yield
    except FloatingPointError as error:
        raise ValueError("OrthoBridge arithmetic overflow; rescale values with training statistics") from error


def _unit(value, name):
    value = _scalar(value, name, positive=False)
    if value > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _period(value, length):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("period must be an integer")
    return int(value) if 2 <= value <= length // 2 else 1


def _template(values, observed, period, reg):
    """Only observed coordinates enter either local means or phase counts."""
    if period == 1:
        return np.zeros(1), None
    y = values[observed] - values[observed[0]]
    half = max(1, period // 2)
    lo = np.searchsorted(observed, observed - half)
    hi = np.searchsorted(observed, observed + half, side="right")
    prefix = np.r_[0., np.cumsum(y)]
    detrended = y - (prefix[hi] - prefix[lo]) / (hi - lo)
    phases = observed % period
    counts = np.bincount(phases, minlength=period)
    sums = np.bincount(phases, weights=detrended, minlength=period)
    raw = np.divide(sums, counts + reg, out=np.zeros(period), where=counts > 0)
    return raw - raw.mean(), detrended


def _strength(season, detrended, groups):
    """Pool within-window ANOVA sums, giving each row its own phase effects."""
    samples = season.shape[-1]
    if groups < 2 or samples <= groups:
        return 0.
    rows = season.size // samples
    msb = np.sum(season ** 2) / (rows * (groups - 1))
    msw = np.sum((detrended - season) ** 2) / (rows * (samples - groups))
    # 1 - MSW/MSB avoids overflow when the F statistic is very large.
    noise = max(msw, 1e-12)
    return float(1. - noise / msb) if msb > noise else 0.


def _persistence(previous, following):
    cross = np.sum(previous * following)
    return float(np.clip(cross / max(np.sum(previous ** 2), 1e-12), 0., .995))


@dataclass(frozen=True)
class _Frame:
    observed: np.ndarray
    offset: np.ndarray  # phase level plus the visible deseasonalized mean
    z: np.ndarray
    phi: float
    prior: float


def _frame(query, period, reg, strength, phi):
    observed = np.flatnonzero(~np.isnan(query))
    raw, detrended = _template(query, observed, period, reg)
    if strength is None:
        phases = observed % period
        strength = 0. if period == 1 else _strength(raw[phases], detrended, np.unique(phases).size)
    season = strength * raw[np.arange(query.size) % period]
    centered = query - season
    mean = centered[observed].mean()
    z = centered - mean
    if phi is None:
        adjacent = observed[:-1][np.diff(observed) == 1]
        phi = _persistence(z[adjacent], z[adjacent + 1])
    return _Frame(observed, season + mean, z, phi, max(float(np.var(z[observed])), 1e-6))


def _gaps(query):
    edges = np.flatnonzero(np.diff(np.r_[False, np.isnan(query), False]))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


@dataclass(frozen=True)
class _Anchor:
    t: int
    indices: np.ndarray
    weights: np.ndarray
    level: float


def _anchor(frame, t, left, window, decay):
    obs = frame.observed
    indices = obs[obs <= t][-window:] if left else obs[obs >= t][:window]
    indices = indices[np.abs(t - indices) <= max(8, window * 2)]
    weights = decay ** np.abs(indices - t)
    weights /= weights.sum()
    return _Anchor(t, indices, weights, float(np.sum(weights * frame.z[indices])))


def _baseline_anomaly(phi, cols, left, right):
    dl = cols - left.t
    if right is None:
        return phi ** dl * left.level
    span = right.t - left.t
    divisor = 1. - phi ** (2 * span)
    if phi >= .999 or divisor <= 1e-10:
        return (1. - dl / span) * left.level + dl / span * right.level
    dr = right.t - cols
    return ((phi ** dl - phi ** (2 * span - dl)) * left.level
            + (phi ** dr - phi ** (2 * span - dr)) * right.level) / divisor


def _variance(phi, distance):
    if phi >= .999:
        return distance.astype(np.float64)
    return (1. - phi ** (2 * distance)) / max(1. - phi * phi, 1e-12)


def _geometry(frame, start, end, window, decay):
    cols = np.arange(start, end)
    left = _anchor(frame, start - 1, True, window, decay)
    right = None if end == frame.z.size else _anchor(frame, end, False, window, decay)
    base = _baseline_anomaly(frame.phi, cols, left, right)
    return cols, left, right, base


def ortho_baseline(query, *, period=1, seasonal_reg=2., seasonal_strength=None,
                   phi=None, anchor_window=4, anchor_decay=.45):
    """Visible-only OU baseline. Supply fitted strength and phi to fix the model.

    Without phi, only consecutive visible pairs contribute to its estimate.
    Leading gaps use the same procedure in reversed time.
    """
    query = _query(query, np.size(query))
    period = _period(period, query.size)
    reg = _scalar(seasonal_reg, "seasonal_reg", positive=False)
    strength = None if seasonal_strength is None else _unit(seasonal_strength, "seasonal_strength")
    phi = None if phi is None else _unit(phi, "phi")
    window = _integer(anchor_window, "anchor_window")
    decay = _unit(anchor_decay, "anchor_decay")
    result = query.copy()
    frames = {}
    with _arithmetic():
        for start, end in _gaps(query):
            reverse = start == 0
            if reverse not in frames:
                frames[reverse] = _frame(query[::-1] if reverse else query, period, reg, strength, phi)
            frame = frames[reverse]
            if reverse:
                start, end = query.size - end, query.size
            cols, _, _, base = _geometry(frame, start, end, window, decay)
            target = result[::-1] if reverse else result
            target[cols] = frame.offset[cols] + base
    return result


def _windows(frame, anchor, length, horizon, limit):
    # Prefix counts test the entire closed interval including the replay anchor.
    missing = np.r_[0, np.cumsum(np.isnan(frame.z[:anchor + 1]))]
    for mode, h in (("preferred", min(length, horizon)), ("short", min(length, 16))):
        found = []
        stop = anchor
        while stop >= h and len(found) < limit:
            r = stop - h
            if missing[stop + 1] == missing[r]:
                found.append((r, stop))
                stop = r
            else:
                stop -= 1
        if found:
            return mode, found
    return "disabled", []


@dataclass(frozen=True)
class _Term:
    anchor: _Anchor
    cols: np.ndarray
    power: np.ndarray


def _term(phi, anchor, cols):
    return _Term(anchor, cols, phi ** np.abs(cols - anchor.t))


@dataclass(frozen=True)
class _Node:
    low: np.ndarray
    high: np.ndarray
    paa_low: np.ndarray
    paa_high: np.ndarray
    delta_low: np.ndarray
    delta_high: np.ndarray
    lag_low: np.ndarray
    lag_high: np.ndarray
    count: int
    children: tuple
    rows: np.ndarray | None


class OrthoBridgeIndex:
    """Complete reference bank with fitted seasonal strength and AR coefficient.

    Tree summaries and the reverse-time fit are built on first use. Row IDs
    always refer to the caller's original order. The optional null row is N.
    """

    def __init__(self, references, *, period=1, seasonal_reg=2., leaf_size=16, paa_block=16):
        started = perf_counter()
        with _arithmetic():
            self._bank = _references(references).copy(order="C")
            self._bank.flags.writeable = False
            self.reference_count, self.length = self._bank.shape
            self.period = _period(period, self.length)
            self.seasonal_reg = _scalar(seasonal_reg, "seasonal_reg", positive=False)
            self.leaf_size = _integer(leaf_size, "leaf_size")
            self.paa_block = _integer(paa_block, "paa_block")
            self.s_bank = np.zeros((self.reference_count, self.period))
            self.seasonal_strength = 0.
            positions = np.arange(self.length)
            if self.period > 1:
                detrended = np.empty_like(self._bank)
                for row, values in enumerate(self._bank):
                    self.s_bank[row], detrended[row] = _template(
                        values, positions, self.period, self.seasonal_reg)
                self.seasonal_strength = _strength(self.s_bank[:, positions % self.period],
                                                   detrended, self.period)
                self.s_bank *= self.seasonal_strength
            z = self._bank - self.s_bank[:, positions % self.period]
            z -= z.mean(axis=1, keepdims=True)
            self.z_bank = z
            self.s_bank.flags.writeable = self.z_bank.flags.writeable = False
            self.phi = _persistence(z[:, :-1], z[:, 1:])
            self._max_anomaly = np.max(np.abs(z))
        self._reverse = None
        self._nodes = []
        self.build_diagnostics = dict(reference_preparation_seconds=perf_counter() - started,
                                      reference_count=self.reference_count,
                                      reference_bytes=self._bank.nbytes,
                                      fitted_bytes=z.nbytes + self.s_bank.nbytes,
                                      tree_build_seconds=0., tree_nodes=0, summary_bytes=0)

    def _reversed(self):
        if self._reverse is None:
            # The lag-regression denominator is directional. Refit on reversal.
            self._reverse = OrthoBridgeIndex(self._bank[:, ::-1], period=self.period,
                                             seasonal_reg=self.seasonal_reg,
                                             leaf_size=self.leaf_size, paa_block=self.paa_block)
        return self._reverse

    def _ensure_tree(self):
        if self._nodes:
            return
        started = perf_counter()
        z = self.z_bank
        nodes = []
        starts = np.arange(0, self.length, min(self.paa_block, self.length))
        sizes = np.diff(np.r_[starts, self.length])
        paa = np.add.reduceat(z, starts, axis=1) / sizes
        # Same-row first differences preserve correlations lost by box subtraction.
        delta = np.diff(z, axis=1)
        lag_count = min(16, self.length - 1)

        def build(rows):
            values = z[rows]
            low, high = values.min(axis=0), values.max(axis=0)
            children = ()
            if rows.size > self.leaf_size:
                projection = paa[rows] if paa.shape[1] > 1 else values
                dim = np.argmax(np.ptp(projection, axis=0))
                order = rows[np.argsort(projection[:, dim], kind="stable")]
                mid = order.size // 2
                children = (build(order[:mid]), build(order[mid:]))
                first, second = (nodes[c] for c in children)
                lag_low = np.minimum(first.lag_low, second.lag_low)
                lag_high = np.maximum(first.lag_high, second.lag_high)
            else:
                lag_low = np.zeros((lag_count, self.length))
                lag_high = np.zeros_like(lag_low)
                for lag in range(1, lag_count + 1):
                    diff = values[:, lag:] - values[:, :-lag]
                    lag_low[lag - 1, :-lag] = diff.min(axis=0)
                    lag_high[lag - 1, :-lag] = diff.max(axis=0)
            blocks = paa[rows]
            diff = delta[rows]
            # Prefix sums of per-step extrema bound any same-row difference.
            node = _Node(low, high, blocks.min(axis=0), blocks.max(axis=0),
                         np.r_[0., np.cumsum(diff.min(axis=0))],
                         np.r_[0., np.cumsum(diff.max(axis=0))], lag_low, lag_high, rows.size,
                         children, None if children else rows)
            nodes.append(node)
            return len(nodes) - 1

        self._root = build(np.arange(self.reference_count))
        zero = np.zeros(self.length)
        zero_paa = np.zeros(starts.size)
        zero_lags = np.zeros((lag_count, self.length))
        self._null = len(nodes)
        nodes.append(_Node(zero, zero, zero_paa, zero_paa, zero, zero, zero_lags, zero_lags, 1,
                           (), np.array([self.reference_count])))
        self._nodes = nodes
        self.build_diagnostics.update(
            tree_build_seconds=perf_counter() - started, tree_nodes=len(nodes),
            summary_bytes=sum(a.nbytes for a in {id(a): a for n in nodes for a in
                              (n.low, n.high, n.paa_low, n.paa_high, n.delta_low,
                               n.delta_high, n.lag_low, n.lag_high, n.rows)
                               if a is not None}.values()))

    def repair(self, query, *, context_size=32, anchor_window=4, anchor_decay=.45,
               k_pairs=16, k=8, temperature=1., shrinkage=.75, compatibility_weight=1.,
               include_null_candidate=True, null_penalty=.05, replay_weight=.35,
               replay_horizon=48, replay_windows=3, backend="tree"):
        """Fill maximal original NaN runs and return values plus per-gap rankings."""
        started = perf_counter()
        query = _query(query, self.length)
        if not isinstance(backend, str) or backend not in ("tree", "scan", "auto"):
            raise ValueError("backend must be 'scan', 'tree' or 'auto'")
        if not isinstance(include_null_candidate, (bool, np.bool_)):
            raise ValueError("include_null_candidate must be boolean")
        context_size = _integer(context_size, "context_size")
        anchor_window = _integer(anchor_window, "anchor_window")
        anchor_decay = _unit(anchor_decay, "anchor_decay")
        k_pairs, k = _integer(k_pairs, "k_pairs"), _integer(k, "k")
        horizon = _integer(replay_horizon, "replay_horizon")
        windows = _integer(replay_windows, "replay_windows")
        replay_weight = _unit(replay_weight, "replay_weight")
        compatibility = _scalar(compatibility_weight, "compatibility_weight", positive=False)
        penalty = _scalar(null_penalty, "null_penalty", positive=False)
        temperature = _scalar(temperature, "temperature", positive=True)
        shrinkage = _unit(shrinkage, "shrinkage")
        result = query.copy()
        gaps = []
        totals = Counter()
        frames = {}
        indexes = {}
        with _arithmetic():
            for start, end in _gaps(query):
                reverse = start == 0
                model = self._reversed() if reverse else self
                indexes["reverse" if reverse else "forward"] = model
                if reverse not in frames:
                    frames[reverse] = _frame(query[::-1] if reverse else query, self.period,
                                             self.seasonal_reg, model.seasonal_strength, model.phi)
                frame = frames[reverse]
                lo, hi = (self.length - end, self.length) if reverse else (start, end)
                cols, left, right, base = _geometry(frame, lo, hi, anchor_window, anchor_decay)
                kind = "leading" if reverse else ("tail" if right is None else "internal")
                info = dict(start=start, end=end, kind=kind, phi=model.phi,
                            seasonal_strength=model.seasonal_strength)
                if shrinkage:
                    search = _GapSearch(model, frame, cols, left, right, context_size,
                                        anchor_window, anchor_decay, horizon, windows,
                                        replay_weight, compatibility, penalty,
                                        include_null_candidate, k if right is None else k_pairs)
                    route = ("tree" if right is not None else "scan") if backend == "auto" else backend
                    ranked = search.run(route)
                    anomaly = search.fuse(ranked, base, shrinkage, temperature)
                    info.update(search.report(ranked, reverse))
                    totals.update(search.counts)
                else:
                    anomaly = base
                target = result[::-1] if reverse else result
                target[cols] = frame.offset[cols] + anomaly
                gaps.append(info)
        diag = dict(backend=backend, gap_count=len(gaps), gaps=gaps, period=self.period,
                    phi=self.phi, seasonal_strength=self.seasonal_strength,
                    reference_count=self.reference_count, shrinkage=shrinkage,
                    numerics="float64_with_padding_not_formally_certified",
                    index_build={key: model.build_diagnostics.copy() for key, model in indexes.items()},
                    query_seconds=perf_counter() - started)
        diag.update({key: totals[key] for key in _COUNTERS})
        return OrthoBridgeRepairResult(result, diag)


_COUNTERS = ("reference_scalar_reads", "context_scalar_pairs", "replay_scalar_pairs",
             "pair_evaluations", "one_evaluations", "pair_coordinate_evaluations",
             "node_visits", "unary_bound_evaluations", "paa_bound_evaluations",
             "fine_bound_evaluations", "bound_coordinate_evaluations", "paa_prunes",
             "fine_prunes", "unreliable_bound_evaluations", "warm_start_evaluations",
             "full_scan_fallbacks")


class _GapSearch:
    """One oriented gap. Exact rows, envelopes and the K-th cutoff are query-local."""

    def __init__(self, model, frame, cols, left, right, context, window, decay,
                 horizon, windows, replay_weight, compatibility, penalty, include_null, k):
        self.model, self.frame, self.cols = model, frame, cols
        self.compatibility = compatibility
        self.n = model.reference_count + int(include_null)
        self.internal = right is not None
        self.k = min(k, self.n ** (2 if self.internal else 1))
        self.penalty = penalty * frame.prior if include_null else 0.
        self.heap = []
        self.counts = Counter()
        phi, obs = frame.phi, frame.observed
        left_cols = obs[obs < left.t][-context:]
        self.contexts = [_term(phi, left, left_cols)]
        self.paths = [_term(phi, left, cols)]
        self.replay = []
        self.replay_mode = "disabled"
        if self.internal:
            self.contexts.append(_term(phi, right, obs[obs > right.t][:context]))
            self.paths.append(_term(phi, right, cols))
            vl, vr = _variance(phi, cols - left.t), _variance(phi, right.t - cols)
            self.denom = vl + vr
            self.mix = vl / self.denom
        else:
            self.replay_mode, intervals = _windows(frame, left.t, len(cols), horizon, windows)
            for r, stop in intervals:
                anchor = _anchor(frame, r, True, window, decay)
                positions = np.arange(r + 1, stop + 1)
                self.replay.append((_term(phi, anchor, positions), _variance(phi, positions - r)))
        self.eta = replay_weight if self.replay else 0.
        self.rows = np.arange(self.n)
        self.row_costs = np.full((len(self.paths), self.n), np.nan)
        self.path_arrays = {}
        self.node_costs = {}
        self.node_paths = {}
        self.seed_rows = None

    def _predict_rows(self, rows, term):
        real = rows < self.model.reference_count
        selected = rows[real, None]
        z, a = self.model.z_bank, term.anchor
        soft = np.sum(z[selected, a.indices] * a.weights, axis=1)
        result = np.empty((rows.size, term.cols.size))
        result[real] = z[selected, term.cols] + (a.level - soft[:, None]) * term.power
        result[~real] = a.level * term.power
        self.counts["reference_scalar_reads"] += selected.size * (a.indices.size + term.cols.size)
        return result

    def _costs(self, side, rows):
        cache = self.row_costs[side]
        needed = rows[np.isnan(cache[rows])]
        if needed.size:
            term = self.contexts[side]
            cost = np.zeros(needed.size)
            if term.cols.size:
                error = self.frame.z[term.cols] - self._predict_rows(needed, term)
                cost = np.mean(error ** 2, axis=1)
                self.counts["context_scalar_pairs"] += error.size
            cost[needed == self.model.reference_count] += self.penalty
            if self.eta:
                replay = np.zeros_like(cost)
                for term, variance in self.replay:
                    error = self.frame.z[term.cols] - self._predict_rows(needed, term)
                    replay += np.mean(error ** 2 / variance, axis=1) / len(self.replay)
                    self.counts["replay_scalar_pairs"] += error.size
                cost = (1. - self.eta) * cost + self.eta * replay
            cache[needed] = cost
        return cache[rows]

    def _unary(self, side, row):
        return self._costs(side, np.array([row]))[0]

    def _path(self, side, row):
        if side in self.path_arrays:
            return self.path_arrays[side][row]
        return self._predict_rows(np.array([row]), self.paths[side])[0]

    def _offer(self, scores, groups, eligible):
        """Only a block's lexicographic top K can enter the global top K."""
        if not np.isfinite(scores[eligible]).all():
            raise ValueError("nonfinite bridge objective; rescale values with training statistics")
        if len(self.heap) == self.k:
            eligible &= scores <= -self.heap[0][0]
        slots = np.flatnonzero(eligible)
        if slots.size > self.k:
            values = scores.ravel()[slots]
            cutoff = np.partition(values, self.k - 1)[self.k - 1]
            slots = slots[values <= cutoff]
        if not slots.size:
            return
        rows = [g[pos] for g, pos in zip(groups, np.unravel_index(slots, scores.shape))]
        values = scores.ravel()[slots]
        order = np.lexsort((*reversed(rows), values))[:self.k]
        for pos in order:
            item = (-float(values[pos]), *(-int(r[pos]) for r in rows))
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, item)
            elif item > self.heap[0]:
                heapq.heapreplace(self.heap, item)
            else:
                break

    def _evaluate(self, groups, warm=False):
        groups = [np.asarray(g, dtype=np.intp) for g in groups]
        if warm:
            self.seed_rows = groups
        if not self.internal:
            rows = groups[0]
            if self.seed_rows is not None and not warm:
                rows = rows[~np.isin(rows, self.seed_rows[0])]
            scores = self._costs(0, rows)
            self._offer(scores, [rows], np.ones(rows.size, dtype=bool))
            self.counts["one_evaluations"] += rows.size
            return
        step = min(128, max(1, int(np.sqrt(262144 / self.cols.size))))
        for a in range(0, len(groups[0]), step):
            left = groups[0][a:a + step]
            for b in range(0, len(groups[1]), step):
                right = groups[1][b:b + step]
                scores = self.row_costs[0, left, None] + self.row_costs[1, right]
                eligible = np.ones(scores.shape, dtype=bool)
                if self.seed_rows is not None and not warm:
                    eligible &= ~(np.isin(left, self.seed_rows[0])[:, None]
                                  & np.isin(right, self.seed_rows[1]))
                count = int(np.count_nonzero(eligible))
                if self.compatibility and count:
                    f, r = self.path_arrays[0][left], self.path_arrays[1][right]
                    if count == scores.size:
                        delta = f[:, None, :] - r
                        np.square(delta, out=delta)
                        delta /= self.denom
                        scores += self.compatibility * delta.mean(axis=-1)
                    else:
                        i, j = np.nonzero(eligible)
                        delta = f[i] - r[j]
                        scores[i, j] += self.compatibility * np.mean(delta ** 2 / self.denom, axis=1)
                    self.counts["pair_coordinate_evaluations"] += delta.size
                self.counts["pair_evaluations"] += count
                self._offer(scores, [left, right], eligible)

    def _prepare_blocks(self):
        size = self.model.paa_block
        # Ignore partial blocks; they contribute a valid lower bound of zero.
        first, last = (int(self.cols[0]) + size - 1) // size, (int(self.cols[-1]) + 1) // size
        self.blocks = np.arange(first, max(first, last))
        begin = first * size - int(self.cols[0])
        stop = begin + len(self.blocks) * size
        if self.blocks.size:
            self.block_power = [t.power[begin:stop].reshape(-1, size).mean(axis=1)
                                for t in self.paths]
            self.block_denom = self.denom[begin:stop].reshape(-1, size).max(axis=1)

    def _box(self, node, term, side=None):
        """Intersect the coordinate box with a weighted same-row difference bound."""
        a = term.anchor
        soft_lo = np.sum(a.weights * node.low[a.indices])
        soft_hi = np.sum(a.weights * node.high[a.indices])
        if side is not None:
            p = self.block_power[side]
            lo, hi = node.paa_low[self.blocks], node.paa_high[self.blocks]
        else:
            p = term.power
            lo, hi = node.low[term.cols], node.high[term.cols]
        low = lo + p * (a.level - soft_hi)
        high = hi + p * (a.level - soft_lo)
        if side is None:
            # z[t]-sum(w*z[s]) is the convex sum of same-row differences.
            dl, dh = np.zeros_like(p), np.zeros_like(p)
            for s, w in zip(a.indices, a.weights):
                before = term.cols < s
                l = node.delta_low[term.cols] - node.delta_low[s]
                h = node.delta_high[term.cols] - node.delta_high[s]
                lower, upper = np.where(before, h, l), np.where(before, l, h)
                lag = np.abs(term.cols - s)
                near = (lag > 0) & (lag <= len(node.lag_low))
                key = lag[near] - 1, np.minimum(term.cols[near], s)
                local_low, local_high = node.lag_low[key], node.lag_high[key]
                lower[near] = np.where(before[near], -local_high, local_low)
                upper[near] = np.where(before[near], -local_low, local_high)
                dl += w * lower
                dh += w * upper
            low = np.maximum(low, (1. - p) * lo + p * (a.level + dl))
            high = np.minimum(high, (1. - p) * hi + p * (a.level + dh))
        self.counts["bound_coordinate_evaluations"] += len(p)
        low = np.nextafter(low - self.margin, -np.inf)
        high = np.nextafter(high + self.margin, np.inf)
        if np.any(low > high) or not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise FloatingPointError("unusable node envelope")
        return low, high

    @staticmethod
    def _distance(a, b):
        return np.maximum(0., np.maximum(a[0] - b[1], b[0] - a[1])) ** 2

    def _unary_bound(self, side, node_id):
        key = side, node_id
        if key not in self.node_costs:
            self.counts["unary_bound_evaluations"] += 1
            node = self.model._nodes[node_id]

            def mse(term, variance):
                target = self.frame.z[term.cols]
                return np.mean(self._distance((target, target), self._box(node, term)) / variance)

            term = self.contexts[side]
            cost = mse(term, 1.) if term.cols.size else 0.
            if node_id == self.model._null:
                cost += self.penalty
            if self.eta:
                replay = sum(mse(t, v) / len(self.replay) for t, v in self.replay)
                cost = (1. - self.eta) * cost + self.eta * replay
            self.node_costs[key] = cost
        return self.node_costs[key]

    def _path_bound(self, side, node_id, coarse):
        key = side, node_id, coarse
        if key not in self.node_paths:
            self.node_paths[key] = self._box(self.model._nodes[node_id], self.paths[side],
                                             side if coarse else None)
        return self.node_paths[key]

    def _bound(self, ids, coarse):
        lower = sum(self._unary_bound(side, n) for side, n in enumerate(ids))
        if self.internal and self.compatibility:
            if coarse and not self.blocks.size:
                return lower
            distance = self._distance(*(self._path_bound(s, n, coarse) for s, n in enumerate(ids)))
            if coarse:
                self.counts["paa_bound_evaluations"] += 1
                compat = np.sum(distance / self.block_denom) * (self.model.paa_block / self.cols.size)
            else:
                self.counts["fine_bound_evaluations"] += 1
                compat = np.mean(distance / self.denom)
            lower += self.compatibility * compat
        return lower

    def _prunes(self, lower):
        if not np.isfinite(lower):
            self.counts["unreliable_bound_evaluations"] += 1
            return False
        if len(self.heap) < self.k:
            return False
        worst = -self.heap[0][0]
        guard = 128 * np.finfo(float).eps * (self.model.length + 8) * max(1., worst)
        return lower > worst + guard

    def _warm_rows(self, side, roots, count):
        """Best unary boxes first; these exact seeds only initialize the cutoff."""
        if self.internal:
            return np.argsort(self.row_costs[side], kind="stable")[:count]
        pending = [(self._unary_bound(side, n), n) for n in roots]
        heapq.heapify(pending)
        rows = []
        while pending and len(rows) < count:
            _, n = heapq.heappop(pending)
            node = self.model._nodes[n]
            if node.rows is None:
                for child in node.children:
                    heapq.heappush(pending, (self._unary_bound(side, child), child))
            else:
                rows.extend(node.rows)
        rows = np.array(rows)
        return rows[np.lexsort((rows, self._costs(side, rows)))[:count]]

    def _tree_search(self):
        scale = max(self.model._max_anomaly, np.max(np.abs(self.frame.z[self.frame.observed])))
        # A pruned exact candidate must not hide an overflowing scan objective.
        # A failed range check routes to enumeration, which can still succeed.
        squared = (8. * scale) ** 2
        upper = squared * self.model.length + 2. * (squared + self.penalty)
        if self.internal:
            upper += self.compatibility * squared
        if not np.isfinite(upper):
            raise FloatingPointError("unreliable objective range")
        self.model._ensure_tree()
        if self.internal:
            self._prepare_blocks()
        nodes = self.model._nodes
        # Prefix differences can cancel after L additions of increments.
        self.margin = 128 * np.finfo(float).eps * (self.model.length + 4) ** 2 * max(1., scale)
        roots = [self.model._root]
        if self.n > self.model.reference_count:
            roots.append(self.model._null)
        if self.internal:
            # Nodes are stored in postorder. Each real row is read once per side.
            for side, costs in enumerate(self.row_costs):
                for n, node in enumerate(nodes):
                    if n == self.model._null and self.n == self.model.reference_count:
                        continue
                    self.node_costs[side, n] = (min(self.node_costs[side, c] for c in node.children)
                                                if node.children else np.min(costs[node.rows]))
                    self.counts["unary_bound_evaluations"] += 1
        count = min(8, int(np.ceil(np.sqrt(self.k))) if self.internal else self.k)
        warm = [self._warm_rows(s, roots, count) for s in range(len(self.paths))]
        self._evaluate(warm, warm=True)
        self.counts["warm_start_evaluations"] = int(np.prod([len(g) for g in warm]))
        if self.internal and self.n > 64 and len(self.heap) == self.k:
            cutoff = -self.heap[0][0]
            guard = 128 * np.finfo(float).eps * (self.model.length + 8) * max(1., cutoff)
            # All omitted rows exceed the cutoff even with the cheapest partner.
            costs = self.row_costs
            shortlist = [self.rows[costs[s] + costs[1-s].min() <= cutoff + guard] for s in (0, 1)]
            if len(shortlist[0]) * len(shortlist[1]) <= 4096:
                self._evaluate(shortlist)
                return
        pairs = [(i, j) for i in roots for j in roots] if self.internal else [(i,) for i in roots]
        pending = [(self._bound(ids, True), ids) for ids in pairs]
        heapq.heapify(pending)
        while pending:
            lower, ids = heapq.heappop(pending)
            self.counts["node_visits"] += 1
            if self._prunes(lower):
                key = "paa_prunes" if self.internal and self.compatibility and self.blocks.size else "fine_prunes"
                self.counts[key] += 1
                continue
            if self.internal and self.compatibility and self._prunes(self._bound(ids, False)):
                self.counts["fine_prunes"] += 1
                continue
            expandable = [s for s, n in enumerate(ids) if nodes[n].children]
            if not expandable:
                self._evaluate([nodes[n].rows for n in ids])
                continue
            side = max(expandable, key=lambda s: nodes[ids[s]].count)
            for child in nodes[ids[side]].children:
                group = ids[:side] + (child,) + ids[side + 1:]
                heapq.heappush(pending, (self._bound(group, True), group))

    def run(self, backend):
        if self.internal:
            for side, term in enumerate(self.paths):
                self._costs(side, self.rows)
                if self.compatibility:
                    self.path_arrays[side] = self._predict_rows(self.rows, term)
        if backend == "tree":
            try:
                self._tree_search()
            except FloatingPointError:
                # Summary arithmetic may overflow even for a finite exact score.
                # Recompute the ranking from scratch with the canonical scorer.
                self.counts["unreliable_bound_evaluations"] += 1
                self.counts["full_scan_fallbacks"] += 1
                self.heap.clear()
                self.seed_rows = None
                backend = "scan"
        if backend == "scan":
            self._evaluate([range(self.n)] * len(self.paths))
        return sorted(tuple(-v for v in item) for item in self.heap)

    def fuse(self, ranked, base, shrinkage, temperature):
        scores = np.array([item[0] for item in ranked])
        # Divide successively to avoid underflow in temperature * scale.
        mid = len(scores) // 2
        median = .5 * scores[mid] + .5 * scores[(len(scores) - 1) // 2]
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(-((scores - scores[0]) / max(median, 1e-6)) / temperature)
        rows = np.array([item[1:] for item in ranked])
        if self.internal:
            for group in rows.T:
                _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
                weights /= np.sqrt(counts[inverse])
        weights /= weights.sum()
        self.weights = weights
        paths = np.stack([self._path(0, i) for i in rows[:, 0]])
        if self.internal:
            other = np.stack([self._path(1, j) for j in rows[:, 1]])
            paths = (1. - self.mix) * paths + self.mix * other
        mean = np.sum(weights[:, None] * paths, axis=0)
        if self.internal:
            return (1. - shrinkage) * base + shrinkage * mean
        disagreement = np.sum(weights[:, None] * (paths - mean) ** 2, axis=0)
        damping = self.frame.prior / (self.frame.prior + disagreement)
        return base + shrinkage * damping * (mean - base)

    def report(self, ranked, reverse):
        def coords(cols):
            return ((self.model.length - 1 - cols) if reverse else cols).tolist()

        info = dict(objectives=[v[0] for v in ranked], weights=self.weights.tolist())
        if self.internal:
            info.update(selected_pairs=[list(v[1:]) for v in ranked],
                        left_context=coords(self.contexts[0].cols),
                        right_context=coords(self.contexts[1].cols))
        else:
            info.update(selected_rows=[v[1] for v in ranked], replay_mode=self.replay_mode,
                        replay_horizon=len(self.replay[0][0].cols) if self.replay else 0,
                        replay_windows=[coords(np.array([t.anchor.t, t.cols[-1]]))
                                        for t, _ in self.replay])
            info["right_context" if reverse else "left_context"] = coords(self.contexts[0].cols)
        return info


def ortho_bridge_repair(query, references, *, period=1, seasonal_reg=2., leaf_size=16,
                        paa_block=16, backend="scan", **parameters):
    """One-shot repair. Reuse OrthoBridgeIndex to amortize fitting and indexing."""
    index = OrthoBridgeIndex(references, period=period, seasonal_reg=seasonal_reg,
                             leaf_size=leaf_size, paa_block=paa_block)
    return index.repair(query, backend=backend, **parameters)
