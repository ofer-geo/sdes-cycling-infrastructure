"""GIS layer loading and small vectorized geometry helpers."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from pyproj import Transformer

from sdes.config import Config


def read_layer(path: Path, crs: str) -> gpd.GeoDataFrame:
    """Read any OGR layer (zipped shapefiles are read in place) and reproject it.

    Date fields are skipped: some municipal exports carry corrupt dates that break parsing.
    """
    uri = f"zip://{path}" if path.suffix.lower() == ".zip" else str(path)
    info = pyogrio.read_info(uri)
    columns = [f for f, t in zip(info["fields"], info["dtypes"], strict=True)
               if not str(t).startswith("datetime")]
    return gpd.read_file(uri, columns=columns).to_crs(crs)


class Layers:
    """Lazy, cached access to the study's GIS layers, all in the projected CRS."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.crs = cfg.crs.projected

    def _load(self, name: str) -> gpd.GeoDataFrame:
        path = getattr(self.cfg.layers, name)
        if path is None:
            raise FileNotFoundError(f"Layer '{name}' is not configured (config/default.yaml)")
        return read_layer(path, self.crs)

    @cached_property
    def city_boundary(self) -> shapely.Polygon:
        return self._load("city_boundary").union_all()

    @cached_property
    def taz(self) -> gpd.GeoDataFrame:
        return self._load("taz")[["TAZ", "geometry"]]

    @cached_property
    def areas(self) -> gpd.GeoDataFrame:
        return self._load("areas")[["area", "geometry"]]

    @cached_property
    def bike_lanes(self) -> gpd.GeoDataFrame:
        return self._load("bike_lanes")

    @cached_property
    def bike_lanes_buffer(self) -> shapely.Geometry:
        """Dissolved buffer around all cycling infrastructure (flat line ends)."""
        buffered = self.bike_lanes.buffer(self.cfg.infrastructure.buffer_m, cap_style="flat")
        geom = buffered.union_all()
        shapely.prepare(geom)
        return geom

    @cached_property
    def slow_zones(self) -> shapely.Geometry:
        return self._load("slow_zones").union_all()

    @cached_property
    def neighbourhoods(self) -> gpd.GeoDataFrame:
        return self._load("neighbourhoods")

    @cached_property
    def open_space(self) -> shapely.Geometry:
        return self._load("open_space").union_all()


def lonlat_transformer(cfg: Config) -> Transformer:
    return Transformer.from_crs(cfg.crs.geographic, cfg.crs.projected, always_xy=True)


def ragged_offsets(lengths: np.ndarray) -> np.ndarray:
    """Start offset of each ragged sub-array given their lengths."""
    return np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)


def max_step_per_line(x: np.ndarray, y: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Largest distance between consecutive vertices of each line in a flat ragged array."""
    step = np.hypot(np.diff(x), np.diff(y))
    starts = ragged_offsets(lengths)
    # the step that bridges two different lines is invalid; zero it out
    step[starts[1:] - 1] = 0.0
    step = np.append(step, 0.0)
    return np.maximum.reduceat(step, starts)
