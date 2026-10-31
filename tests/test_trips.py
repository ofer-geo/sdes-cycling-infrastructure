import networkx as nx
import numpy as np
import pytest
import shapely

from sdes.config import MapMatching
from sdes.geo import max_step_per_line
from sdes.report import FilterReport
from sdes.trips.map_matching import MapMatcher, MatchError
from sdes.trips.network import to_simple_undirected

RULES = MapMatching(buffer_m=25, min_edge_coverage=0.9, short_trip_m=1000,
                    short_trip_max_abs_diff_m=100, long_trip_max_rel_diff=0.05)


def grid(n: int = 5, step: float = 100.0) -> nx.Graph:
    """n x n street grid, node id = row * n + col."""
    g = nx.Graph()
    for r in range(n):
        for c in range(n):
            g.add_node(r * n + c, x=c * step, y=r * step)
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                g.add_edge(r * n + c, r * n + c + 1, length=step)
            if r + 1 < n:
                g.add_edge(r * n + c, (r + 1) * n + c, length=step)
    return g


def test_match_follows_the_ridden_streets():
    # L-shaped ride: east along y=0, then north along x=300
    route = shapely.LineString([(2, 3), (150, -4), (298, 1), (303, 150), (299, 296)])
    result = MapMatcher(grid(), RULES).match(route)
    assert (result.start_node, result.end_node) == (0, 18)
    assert not result.reconnected
    assert result.geometry.length == pytest.approx(600)
    assert result.geometry.equals(shapely.LineString(
        [(0, 0), (100, 0), (200, 0), (300, 0), (300, 100), (300, 200), (300, 300)]))


def test_match_bridges_a_missing_link():
    g = grid()
    g.remove_edge(1, 2)  # the rider uses a link that is missing from OSM
    route = shapely.LineString([(0, 0), (400, 0)])
    result = MapMatcher(g, RULES).match(route)
    assert result.reconnected
    assert result.geometry.length == pytest.approx(400)


def test_match_rejects_degenerate_route():
    with pytest.raises(MatchError):
        MapMatcher(grid(), RULES).match(shapely.LineString([(1, 1), (3, 2)]))


def test_simple_undirected_keeps_shortest_parallel_edge():
    g = nx.MultiDiGraph()
    g.add_node(1, x=0, y=0)
    g.add_node(2, x=1, y=0)
    g.add_edge(1, 2, length=5.0)
    g.add_edge(2, 1, length=3.0)
    g.add_edge(1, 2, length=9.0)
    assert to_simple_undirected(g)[1][2]["length"] == 3.0


def test_max_step_per_line_ignores_jumps_between_lines():
    x = np.array([0, 10, 20, 1000, 1001, 1100.0])
    y = np.zeros(6)
    assert max_step_per_line(x, y, np.array([3, 3])).tolist() == [10, 99]


def test_filter_report_funnel():
    r = FilterReport("t")
    r.start(100)
    r.add("a", 10)
    r.add("b", 5)
    r.add("a", 1)
    assert r.remaining == 84
    assert r.to_frame()["remaining"].tolist() == [100, 89, 84]


def test_decode_wkb_keeps_failed_matches_as_none():
    import pandas as pd
    import shapely

    from sdes.trips.route_choice import decode_wkb

    line = shapely.LineString([(0, 0), (1, 1)])
    out = decode_wkb(pd.Series([line.wkb, float("nan"), None]))
    assert out[0].equals(line)
    assert out[1] is None and out[2] is None
