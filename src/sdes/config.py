"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_trips: Path
    raw_status_changes: Path
    raw_status_changes_previous: Path
    interim: Path
    processed: Path
    layers: Path
    figures: Path


@dataclass(frozen=True)
class Layers:
    city_boundary: Path | None
    taz: Path | None
    areas: Path | None
    bike_lanes: Path | None
    slow_zones: Path | None
    neighbourhoods: Path | None
    open_space: Path | None


@dataclass(frozen=True)
class CRS:
    geographic: str
    projected: str


@dataclass(frozen=True)
class Study:
    start: str
    end: str
    vehicle_type: str
    osm_place: str


@dataclass(frozen=True)
class TripRules:
    city_buffer_m: float
    max_ping_gap_m: float
    min_length_m: float
    max_length_m: float
    max_route_to_aerial_ratio: float


@dataclass(frozen=True)
class MapMatching:
    buffer_m: float
    min_edge_coverage: float
    short_trip_m: float
    short_trip_max_abs_diff_m: float
    long_trip_max_rel_diff: float


@dataclass(frozen=True)
class Infrastructure:
    buffer_m: float


@dataclass(frozen=True)
class Speed:
    min_segment_m: float
    slow_zone_kmh: float
    speed_quantiles: tuple[float, float]


@dataclass(frozen=True)
class Parking:
    same_location_m: float
    merge_gap_s: float
    low_battery_window_s: float
    min_duration_s: float
    trip_match_time_s: float
    trip_match_distance_m: float
    open_space_buffer_m: float
    distance_groups_m: tuple[float, ...]


@dataclass(frozen=True)
class Config:
    paths: Paths
    layers: Layers
    crs: CRS
    study: Study
    trips: TripRules
    map_matching: MapMatching
    infrastructure: Infrastructure
    speed: Speed
    parking: Parking

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        sections = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for name in sections:
            section_cls = globals()[sections[name]]
            values = raw[name]
            if section_cls in (Paths, Layers):
                values = {k: _resolve(v) if v else None for k, v in values.items()}
            values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
            kwargs[name] = section_cls(**values)
        return cls(**kwargs)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
