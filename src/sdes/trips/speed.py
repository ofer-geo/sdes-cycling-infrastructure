"""Riding speed inside vs. outside cycling infrastructure (thesis 5.1.3 and Appendix E).

Each GPS route is split into runs of pings inside / outside the buffered infrastructure.
A transition point is placed midway (in space and time) between the last ping of one run
and the first of the next, which is the same as splitting that step's distance and
duration 50/50 between the two runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from sdes.config import Speed
from sdes.geo import ragged_offsets


@dataclass
class Segment:
    inside: bool
    length_m: float
    duration_s: float
    first: int          # index of the first ping in the run
    last: int           # index of the last ping in the run


def trim_idle(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """Index range that excludes repeated identical pings at the start and end of a ride
    (unlocking, helmet, parking photo...)."""
    moved = (np.diff(x) != 0) | (np.diff(y) != 0)
    if not moved.any():
        return 0, len(x) - 1
    return int(np.argmax(moved)), int(len(moved) - np.argmax(moved[::-1]))


def segment_route(x, y, t_s, inside) -> list[Segment]:
    """Split a route into inside/outside runs with midpoint transitions."""
    step_len = np.hypot(np.diff(x), np.diff(y))
    step_dur = np.diff(t_s)
    change = inside[1:] != inside[:-1]
    # share of each step that belongs to the run of its *starting* ping
    own = np.where(change, 0.5, 1.0)

    bounds = np.flatnonzero(change) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds - 1, [len(x) - 1]])
    segments = []
    for first, last in zip(starts, ends, strict=True):
        # full steps inside the run, plus half of the bridging steps on either side
        length = step_len[first:last].sum()
        duration = step_dur[first:last].sum()
        if first > 0:
            length += step_len[first - 1] * 0.5
            duration += step_dur[first - 1] * 0.5
        if last < len(x) - 1:
            length += step_len[last] * own[last]
            duration += step_dur[last] * own[last]
        segments.append(Segment(bool(inside[first]), length, duration, first, last))
    return segments


def absorb_short_segments(segments: list[Segment], min_length_m: float) -> list[Segment]:
    """Short runs are GPS artefacts (crossing a bike lane, a brief gap in it): give them the
    type of their neighbour, then dissolve consecutive runs of the same type."""
    if len(segments) < 2:
        return segments
    types = [s.inside for s in segments]
    for i, seg in enumerate(segments):
        if seg.length_m < min_length_m:
            neighbour = types[i - 1] if i > 0 else types[1]
            types[i] = neighbour
    merged: list[Segment] = []
    for seg, inside in zip(segments, types, strict=True):
        if merged and merged[-1].inside == inside:
            prev = merged[-1]
            merged[-1] = Segment(inside, prev.length_m + seg.length_m,
                                 prev.duration_s + seg.duration_s, prev.first, seg.last)
        else:
            merged.append(Segment(inside, seg.length_m, seg.duration_s, seg.first, seg.last))
    return merged


class RouteSegmenter:
    """Computes inside/outside-infrastructure distance and time for every trip."""

    def __init__(self, infra: shapely.Geometry, rules: Speed,
                 slow_zones: shapely.Geometry | None = None):
        self.infra = infra
        self.rules = rules
        self.slow_zones = slow_zones
        shapely.prepare(self.infra)
        if slow_zones is not None:
            shapely.prepare(self.slow_zones)

    def summarize(self, trips: gpd.GeoDataFrame) -> pd.DataFrame:
        geoms = trips.geometry.values
        coords = shapely.get_coordinates(geoms)
        n = shapely.get_num_points(geoms)
        offsets = ragged_offsets(n)
        inside_all = shapely.contains_xy(self.infra, coords[:, 0], coords[:, 1])
        crosses_slow = (shapely.intersects(self.slow_zones, geoms)
                        if self.slow_zones is not None else np.zeros(len(trips), bool))

        rows = []
        for i, (off, k, ts) in enumerate(zip(offsets, n, trips["ts"], strict=True)):
            x, y = coords[off:off + k, 0], coords[off:off + k, 1]
            a, b = trim_idle(x, y)
            x, y = x[a:b + 1], y[a:b + 1]
            t_s = np.asarray(ts, dtype=np.float64)[a:b + 1] / 1000
            segments = segment_route(x, y, t_s, inside_all[off + a:off + b + 1])
            segments = absorb_short_segments(segments, self.rules.min_segment_m)
            if crosses_slow[i]:
                segments = [self._remove_slow_zone(s, x, y) for s in segments]
            rows.append(self._totals(segments))
        out = pd.DataFrame(rows, index=trips.index)
        out["inside_kmh"] = out["inside_m"] / out["inside_s"] * 3.6
        out["outside_kmh"] = out["outside_m"] / out["outside_s"] * 3.6
        moving_m, moving_s = out["inside_m"] + out["outside_m"], out["inside_s"] + out["outside_s"]
        out["moving_kmh"] = moving_m / moving_s * 3.6
        out["infra_pct"] = out["inside_m"] / moving_m * 100
        return out.replace([np.inf, -np.inf], np.nan)

    def _remove_slow_zone(self, seg: Segment, x, y) -> Segment:
        """Drop the part of a segment inside speed-capped zones (assumes it was ridden at the
        zone's speed cap), so capped riding doesn't bias the infrastructure comparison."""
        line = shapely.linestrings(np.c_[x[seg.first:seg.last + 1], y[seg.first:seg.last + 1]]) \
            if seg.last > seg.first else None
        if line is None:
            return seg
        slow_m = min(shapely.length(shapely.intersection(line, self.slow_zones)), seg.length_m)
        slow_s = min(slow_m / (self.rules.slow_zone_kmh / 3.6), seg.duration_s)
        return Segment(seg.inside, seg.length_m - slow_m, seg.duration_s - slow_s,
                       seg.first, seg.last)

    @staticmethod
    def _totals(segments: list[Segment]) -> dict:
        tot = {"inside_m": 0.0, "inside_s": 0.0, "outside_m": 0.0, "outside_s": 0.0}
        for s in segments:
            side = "inside" if s.inside else "outside"
            tot[f"{side}_m"] += s.length_m
            tot[f"{side}_s"] += s.duration_s
        tot["n_segments"] = len(segments)
        return tot
