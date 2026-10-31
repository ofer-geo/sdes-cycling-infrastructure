"""Trip sample consolidation (thesis Appendix C).

Every rule is vectorized over a flat array of all GPS pings in a chunk; trips are
handled as ragged slices of that array instead of Python objects.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from sdes.config import Config
from sdes.geo import Layers, lonlat_transformer, max_step_per_line
from sdes.report import FilterReport

KEEP_COLUMNS = ["trip_id", "device_id", "trip_duration", "trip_distance",
                "start_time", "end_time", "n_points", "ts"]


class TripCleaner:
    """Turns raw MDS trips into clean, projected route LineStrings."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rules = cfg.trips
        self.to_projected = lonlat_transformer(cfg)
        self.city = Layers(cfg).city_boundary.buffer(self.rules.city_buffer_m)
        shapely.prepare(self.city)

    def clean(self, trips: pd.DataFrame, report: FilterReport) -> gpd.GeoDataFrame:
        report.start(len(trips))
        start, end = pd.Timestamp(self.cfg.study.start), pd.Timestamp(self.cfg.study.end)

        trips = self._drop(trips, trips["vehicle_type"] != self.cfg.study.vehicle_type,
                           "not an e-scooter", report)
        trips = self._drop(trips, ~trips["start_time"].between(start, end, inclusive="left"),
                           "starts outside study period", report)
        trips = self._drop(trips, trips["n_points"] < 2, "no route geometry", report)

        lengths = trips["n_points"].to_numpy()
        x, y = self.to_projected.transform(np.concatenate(trips["lon"].to_numpy()),
                                           np.concatenate(trips["lat"].to_numpy()))
        gap_ok = max_step_per_line(x, y, lengths) <= self.rules.max_ping_gap_m

        lines = shapely.linestrings(x, y, indices=np.repeat(np.arange(len(trips)), lengths))
        trips = gpd.GeoDataFrame(trips[KEEP_COLUMNS], geometry=lines, crs=self.cfg.crs.projected)
        trips = self._drop(trips, ~gap_ok, "low-resolution route (ping gap)", report)

        trips["length_m"] = trips.length
        trips = self._drop(trips, trips["length_m"] <= self.rules.min_length_m,
                           "too short", report)
        trips = self._drop(trips, trips["length_m"] >= self.rules.max_length_m,
                           "too long", report)
        trips = self._drop(trips, ~shapely.contains(self.city, trips.geometry.values),
                           "outside city", report)
        return trips

    @staticmethod
    def _drop(df, mask, step: str, report: FilterReport):
        mask = np.asarray(mask)
        report.add(step, int(mask.sum()))
        return df.loc[~mask]


def add_trip_attributes(trips: gpd.GeoDataFrame, layers: Layers) -> gpd.GeoDataFrame:
    """Temporal attributes, origin/destination TAZ and average speed."""
    trips = trips.copy()
    coords = shapely.get_coordinates(trips.geometry.values)
    n = shapely.get_num_points(trips.geometry.values)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    origins = gpd.points_from_xy(*coords[first].T, crs=trips.crs)
    dests = gpd.points_from_xy(*coords[first + n - 1].T, crs=trips.crs)

    taz = layers.taz
    for col, points in [("taz_start", origins), ("taz_end", dests)]:
        joined = gpd.sjoin(gpd.GeoDataFrame(geometry=points, index=trips.index), taz,
                           how="left", predicate="within")
        trips[col] = joined.loc[~joined.index.duplicated(), "TAZ"]

    trips["start_hour"] = trips["start_time"].dt.hour
    trips["weekday"] = trips["start_time"].dt.day_name()
    trips["day_type"] = np.select(
        [trips["weekday"] == "Friday", trips["weekday"] == "Saturday"],
        ["Friday", "Saturday"], default="working day")
    trips["speed_kmh"] = trips["length_m"] / trips["trip_duration"] * 3.6
    return trips


# --------------------------------------------------------------------------- batch runner

_cleaner: TripCleaner | None = None


def _init_worker(cfg: Config) -> None:
    global _cleaner
    _cleaner = TripCleaner(cfg)


def _clean_part(path: Path) -> tuple[gpd.GeoDataFrame, FilterReport]:
    report = FilterReport("trips")
    return _cleaner.clean(pq.read_table(path).to_pandas(), report), report


def clean_all(cfg: Config, parts_dir: Path, workers: int = 8) -> tuple[gpd.GeoDataFrame,
                                                                        FilterReport]:
    """Clean every Parquet part in parallel and merge the per-part funnels."""
    parts = sorted(Path(parts_dir).glob("*.parquet"))
    report, frames = FilterReport("Trips sample consolidation"), []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for gdf, part_report in pool.map(_clean_part, parts):
            frames.append(gdf)
            report.start(part_report.initial)
            for step, n in part_report.steps:
                report.add(step, n)
    trips = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(trips, geometry="geometry", crs=cfg.crs.projected), report
