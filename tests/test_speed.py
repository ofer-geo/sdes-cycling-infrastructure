import geopandas as gpd
import numpy as np
import pytest
import shapely

from sdes.config import Speed
from sdes.trips.speed import RouteSegmenter, absorb_short_segments, segment_route, trim_idle

RULES = Speed(min_segment_m=80, slow_zone_kmh=15, speed_quantiles=(0.01, 0.99))


def test_trim_idle_drops_repeated_start_and_end_pings():
    x = np.array([0, 0, 0, 10, 20, 30, 30.0])
    y = np.zeros(7)
    assert trim_idle(x, y) == (2, 5)


def test_segment_route_splits_transition_steps_in_half():
    x = np.array([0, 100, 200, 300.0])
    y = np.zeros(4)
    t = np.array([0, 10, 20, 40.0])
    inside = np.array([False, False, True, True])
    out_seg, in_seg = segment_route(x, y, t, inside)
    assert (out_seg.inside, in_seg.inside) == (False, True)
    assert out_seg.length_m == pytest.approx(150)   # 100 + half of the bridging step
    assert out_seg.duration_s == pytest.approx(15)
    assert in_seg.length_m == pytest.approx(150)
    assert in_seg.duration_s == pytest.approx(25)


def test_short_segment_takes_neighbour_type_and_merges():
    x = np.array([0, 200, 230, 260, 500.0])
    inside = np.array([False, False, True, False, False])
    segs = segment_route(x, np.zeros(5), np.arange(5.0) * 10, inside)
    assert [s.inside for s in segs] == [False, True, False]
    merged = absorb_short_segments(segs, 80)
    assert len(merged) == 1 and not merged[0].inside
    assert merged[0].length_m == pytest.approx(500)
    assert merged[0].duration_s == pytest.approx(40)


def test_segmenter_end_to_end_with_slow_zone():
    infra = shapely.box(250, -50, 1100, 50)          # bike lane along the 2nd half
    slow = shapely.box(900, -50, 1100, 50)           # last 100 m is a 15 km/h zone
    line = shapely.linestrings([(0, 0), (0, 0), (500, 0), (1000, 0)])
    trips = gpd.GeoDataFrame({"ts": [[0, 5_000, 105_000, 205_000]]}, geometry=[line])
    out = RouteSegmenter(infra, RULES, slow_zones=slow).summarize(trips).iloc[0]
    assert out["outside_m"] == pytest.approx(250)
    assert out["inside_m"] == pytest.approx(650)     # 750 inside minus 100 slow-zone metres
    assert out["inside_s"] == pytest.approx(150 - 100 / (15 / 3.6))
    assert out["infra_pct"] == pytest.approx(650 / 900 * 100)
