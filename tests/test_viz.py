"""Smoke tests: every figure renders from small synthetic inputs."""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import shapely  # noqa: E402

from sdes import viz  # noqa: E402

rng = np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _close_figures():
    viz.apply_style()
    yield
    plt.close("all")


def test_length_bins_edges():
    b = viz.length_bins(pd.Series([250, 999, 1000, 4999, 14000]))
    assert list(b.astype(str)) == ["0-1", "0-1", "1-2", "4-5", "5+"]


def test_funnel():
    rep = pd.DataFrame({"step": ["initial", "a", "b"], "removed": [0, 10, 5],
                        "remaining": [100, 90, 85]})
    viz.funnel({"x": rep, "y": rep})


def test_hourly_trips():
    n = 500
    trips = pd.DataFrame({"day_type": rng.choice(viz.DAY_TYPES, n),
                          "start_hour": rng.integers(0, 24, n)})
    viz.hourly_trips(trips)


def _routes(n=200):
    length = rng.uniform(300, 7000, n)
    sp = length / rng.uniform(1.0, 1.3, n)
    line = shapely.LineString([(0, 0), (1000, 500), (2000, 0)])
    sp_line = shapely.LineString([(0, 0), (2000, 0)])
    return pd.DataFrame({
        "length_m": length, "matched_length_m": length, "sp_length_m": sp,
        "infra_m": length * 0.6, "sp_infra_m": sp * 0.45,
        "infra_pct": rng.uniform(40, 90, n), "sp_infra_pct": rng.uniform(0, 50, n),
        "detour_ratio": length / sp, "reconnected": False,
        "geometry": [line] * n, "matched": [line] * n, "shortest_path": [sp_line] * n,
    })


def test_infra_share_and_example():
    routes = _routes()
    viz.infra_share_by_length(routes)
    row = viz.pick_example(routes.assign(length_m=2000.0, detour_ratio=1.2))
    lanes = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (2000, 500)])])
    viz.route_example(row, lanes)


def test_speed_by_length():
    n = 300
    speed = pd.DataFrame({"length_m": rng.uniform(300, 7000, n),
                          "inside_m": rng.uniform(0, 3000, n), "inside_s": rng.uniform(1, 900, n),
                          "outside_m": rng.uniform(0, 3000, n),
                          "outside_s": rng.uniform(1, 900, n)})
    viz.speed_by_length(speed)


def test_parking_by_distance():
    n = 400
    events = pd.DataFrame({"end_reason": "on_trip", "dist_to_infra_m": rng.uniform(0, 150, n),
                           "area": rng.choice(viz.AREAS, n),
                           "duration_s": rng.uniform(60, 30000, n)})
    viz.parking_by_distance(events)


def test_density_map():
    city = shapely.box(0, 0, 1000, 2000)
    xy = rng.uniform([0, 0], [1000, 2000], (5000, 2))
    lanes = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 0), (1000, 2000)])])
    viz.density_map(xy, city, lanes, n_trips=10, gridsize=20)
