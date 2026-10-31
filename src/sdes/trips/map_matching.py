"""Buffer-and-shortest-path map-matching (thesis 5.1.2.1 and Appendix D).

1. Buffer the GPS route and keep the OSM edges that lie (mostly) inside the buffer.
2. Snap the route's start/end to the nearest nodes of that subgraph.
3. If the subgraph is disconnected (rider used a link missing from OSM), bridge the
   components through their closest node pairs.
4. The shortest path through the subgraph is the matched route.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree

from sdes.config import MapMatching


@dataclass(frozen=True)
class MatchResult:
    geometry: shapely.LineString
    start_node: int
    end_node: int
    reconnected: bool


class MatchError(Exception):
    """The route could not be matched; ``str(err)`` is the reason."""


class MapMatcher:
    def __init__(self, graph: nx.Graph, rules: MapMatching):
        self.rules = rules
        self.graph = graph
        self.node_xy = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}

        u, v, length = zip(*graph.edges(data="length"), strict=True)
        self.u, self.v, self.length = np.array(u), np.array(v), np.array(length)
        ends = np.array([[self.node_xy[a], self.node_xy[b]] for a, b in zip(u, v, strict=True)])
        self.edges = shapely.linestrings(ends)
        self.edge_len = shapely.length(self.edges)
        self.tree = shapely.STRtree(self.edges)

    # ------------------------------------------------------------------ public
    def match(self, route: shapely.LineString) -> MatchResult:
        subgraph = self._candidate_subgraph(route)
        nodes = np.fromiter(subgraph.nodes, dtype=np.int64)
        xy = np.array([self.node_xy[n] for n in nodes])
        (sx, sy), (ex, ey) = shapely.get_coordinates(route)[[0, -1]]
        start = int(nodes[np.argmin(np.hypot(xy[:, 0] - sx, xy[:, 1] - sy))])
        end = int(nodes[np.argmin(np.hypot(xy[:, 0] - ex, xy[:, 1] - ey))])
        if start == end:
            raise MatchError("start and end snap to the same node")

        reconnected = False
        if not nx.has_path(subgraph, start, end):
            self._bridge_components(subgraph)
            reconnected = True

        path = nx.bidirectional_dijkstra(subgraph, start, end, weight="length")[1]
        return MatchResult(self.path_geometry(path), start, end, reconnected)

    def path_geometry(self, path: list[int]) -> shapely.LineString:
        # simplify=False graph: every edge is a straight segment between its nodes
        return shapely.linestrings([self.node_xy[n] for n in path])

    # ------------------------------------------------------------------ internals
    def _candidate_subgraph(self, route: shapely.LineString) -> nx.Graph:
        buffer = shapely.buffer(route, self.rules.buffer_m)
        idx = self.tree.query(buffer, predicate="intersects")
        if len(idx) == 0:
            raise MatchError("no OSM edges near route")
        inside = shapely.length(shapely.intersection(self.edges[idx], buffer))
        with np.errstate(invalid="ignore", divide="ignore"):
            coverage = np.where(self.edge_len[idx] > 0, inside / self.edge_len[idx], 1.0)
        idx = idx[coverage >= self.rules.min_edge_coverage]
        if len(idx) == 0:
            raise MatchError("no OSM edge sufficiently covered by route buffer")
        subgraph = nx.Graph()
        subgraph.add_weighted_edges_from(
            zip(self.u[idx], self.v[idx], self.length[idx], strict=True), weight="length")
        return subgraph

    def _bridge_components(self, subgraph: nx.Graph) -> None:
        """Greedily join components through their closest pair of nodes until connected."""
        components = [np.fromiter(c, dtype=np.int64) for c in nx.connected_components(subgraph)]
        main, rest = components[0], components[1:]
        while rest:
            tree = cKDTree([self.node_xy[n] for n in main])
            best = None
            for ci, comp in enumerate(rest):
                dist, nearest = tree.query([self.node_xy[n] for n in comp])
                k = int(np.argmin(dist))
                if best is None or dist[k] < best[0]:
                    best = (dist[k], ci, int(main[nearest[k]]), int(comp[k]))
            dist, ci, a, b = best
            subgraph.add_edge(a, b, length=float(dist))
            main = np.concatenate([main, rest.pop(ci)])
