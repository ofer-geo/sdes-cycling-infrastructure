"""Chosen route vs. shortest path: how much cycling infrastructure riders seek out
(thesis 5.1.2 and Appendices C.2, D.3)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import shapely

from sdes.config import Config, MapMatching, TripRules
from sdes.report import FilterReport
from sdes.trips.map_matching import MapMatcher, MatchError


def route_choice_sample(trips: gpd.GeoDataFrame, rules: TripRules,
                        report: FilterReport) -> gpd.GeoDataFrame:
    """Drop trips where minimising distance clearly wasn't the rider's goal."""
    report.start(len(trips))
    coords = shapely.get_coordinates(trips.geometry.values)
    n = shapely.get_num_points(trips.geometry.values)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    aerial = np.hypot(*(coords[first + n - 1] - coords[first]).T)

    circular = aerial == 0
    report.add("circular (start = end)", int(circular.sum()))
    trips = trips.loc[~circular].assign(aerial_m=aerial[~circular])

    ratio = trips["length_m"] / trips["aerial_m"]
    irrational = ratio >= rules.max_route_to_aerial_ratio
    report.add(f"route/aerial ratio >= {rules.max_route_to_aerial_ratio:g}",
               int(irrational.sum()))
    return trips.loc[~irrational]


# --------------------------------------------------------------------------- matching + SP

_matcher: MapMatcher | None = None


def _init_worker(graph: nx.Graph, rules: MapMatching) -> None:
    global _matcher
    _matcher = MapMatcher(graph, rules)


def _match_batch(batch: tuple[np.ndarray, np.ndarray]) -> list[dict]:
    trip_ids, wkbs = batch
    out = []
    for trip_id, route in zip(trip_ids, shapely.from_wkb(wkbs), strict=True):
        row = {"trip_id": trip_id}
        try:
            m = _matcher.match(route)
            sp_len, sp_path = nx.bidirectional_dijkstra(_matcher.graph, m.start_node,
                                                        m.end_node, weight="length")
            row.update(matched=m.geometry.wkb, start_node=m.start_node, end_node=m.end_node,
                       reconnected=m.reconnected,
                       shortest_path=_matcher.path_geometry(sp_path).wkb)
        except (MatchError, nx.NetworkXNoPath) as err:
            row["match_error"] = str(err) or type(err).__name__
        out.append(row)
    return out


WORKER_MEMORY_GB = 0.6   # measured: ~0.2 GB graph/index state + interpreter and imports


def default_workers() -> int:
    """As many workers as CPUs allow, but never more than fit in free RAM: every worker
    holds its own copy of the city graph, and swapping makes the run ~100x slower."""
    free_gb = psutil.virtual_memory().available / 1e9
    return max(1, min((os.cpu_count() or 2) - 2, int((free_gb - 1) / WORKER_MEMORY_GB)))


def match_and_route(trips: gpd.GeoDataFrame, graph: nx.Graph, rules: MapMatching,
                    workers: int | None = None, batch_size: int = 1000) -> pd.DataFrame:
    """Map-match every trip and compute the shortest path between the matched end nodes."""
    workers = workers or default_workers()
    print(f"  map-matching {len(trips):,} trips with {workers} workers", flush=True)
    ids, wkbs = trips["trip_id"].to_numpy(), shapely.to_wkb(trips.geometry.values)
    batches = [(ids[i:i + batch_size], wkbs[i:i + batch_size])
               for i in range(0, len(ids), batch_size)]
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(graph, rules)) as pool:
        for i, result in enumerate(pool.map(_match_batch, batches), 1):
            rows.extend(result)
            if i % 20 == 0 or i == len(batches):
                print(f"  matched {min(i * batch_size, len(ids)):,}/{len(ids):,}", flush=True)
    return pd.DataFrame(rows)


def decode_wkb(values: pd.Series) -> np.ndarray:
    """WKB -> geometries; missing values (trips that failed to match) become None."""
    values = values.to_numpy(dtype=object)
    return shapely.from_wkb(np.where(pd.isna(values), None, values))


def drop_bad_matches(trips: gpd.GeoDataFrame, rules: MapMatching,
                     report: FilterReport) -> gpd.GeoDataFrame:
    """Remove trips whose matched route deviates too much from the GPS route."""
    failed = trips["matched"].isna()
    report.add("could not be map-matched", int(failed.sum()))
    trips = trips.loc[~failed].copy()

    trips["matched_length_m"] = shapely.length(trips["matched"].values)
    diff = (trips["matched_length_m"] - trips["length_m"]).abs()
    short = trips["length_m"] <= rules.short_trip_m
    bad_short = short & (diff > rules.short_trip_max_abs_diff_m)
    bad_long = ~short & (diff / trips["length_m"] > rules.long_trip_max_rel_diff)
    report.add(f"short trip, |matched - GPS| > {rules.short_trip_max_abs_diff_m:g} m",
               int(bad_short.sum()))
    report.add(f"long trip, |matched - GPS| > {rules.long_trip_max_rel_diff:.0%}",
               int(bad_long.sum()))
    return trips.loc[~(bad_short | bad_long)]


# --------------------------------------------------------------------------- infrastructure

def infrastructure_share(lines: np.ndarray, infra: shapely.Geometry) -> tuple[np.ndarray,
                                                                              np.ndarray]:
    """Length (m) and share (%) of each line inside the buffered cycling infrastructure."""
    inside = shapely.length(shapely.intersection(lines, infra))
    total = shapely.length(lines)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, inside / total * 100, 0.0)
    return inside, share


def compare_routes(trips: gpd.GeoDataFrame, infra: shapely.Geometry) -> gpd.GeoDataFrame:
    trips = trips.copy()
    matched, sp = trips["matched"].values, trips["shortest_path"].values
    trips["infra_m"], trips["infra_pct"] = infrastructure_share(matched, infra)
    trips["sp_infra_m"], trips["sp_infra_pct"] = infrastructure_share(sp, infra)
    trips["sp_length_m"] = shapely.length(sp)
    trips["detour_ratio"] = trips["matched_length_m"] / trips["sp_length_m"]

    # A matched route can be shorter than the network SP (missing OSM links / shortcuts);
    # then the chosen route *is* the shortest path.
    shorter = trips["detour_ratio"] < 1
    for col_sp, col_matched in [("shortest_path", "matched"), ("sp_length_m", "matched_length_m"),
                                ("sp_infra_m", "infra_m"), ("sp_infra_pct", "infra_pct")]:
        trips.loc[shorter, col_sp] = trips.loc[shorter, col_matched]
    trips.loc[shorter, "detour_ratio"] = 1.0
    return trips


def run_route_choice(cfg: Config, trips: gpd.GeoDataFrame, graph: nx.Graph,
                     infra: shapely.Geometry, workers: int | None = None,
                     ) -> tuple[gpd.GeoDataFrame, FilterReport]:
    report = FilterReport("Route-choice sample")
    trips = route_choice_sample(trips, cfg.trips, report)
    matched = match_and_route(trips, graph, cfg.map_matching, workers=workers)
    trips = trips.merge(matched, on="trip_id", how="left")
    for col in ("matched", "shortest_path"):
        trips[col] = decode_wkb(trips[col]) if col in trips else None
    trips = drop_bad_matches(trips, cfg.map_matching, report)
    trips = compare_routes(trips, infra).reset_index(drop=True)
    trips = gpd.GeoDataFrame(trips, geometry="geometry", crs=cfg.crs.projected)
    trips["matched"] = gpd.GeoSeries(trips["matched"], crs=cfg.crs.projected)
    trips["shortest_path"] = gpd.GeoSeries(trips["shortest_path"], crs=cfg.crs.projected)
    return trips, report
