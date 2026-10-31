"""The OpenStreetMap walking network used for map-matching and shortest paths."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox

from sdes.config import Config


def load_walk_graph(cfg: Config) -> nx.MultiDiGraph:
    """Download (once, then cache) the unsimplified OSM walk network, projected.

    The *walk* network is used on purpose: it also contains pedestrian-only links,
    which e-scooter riders use as shortcuts. ``simplify=False`` keeps every OSM node,
    so each edge is a straight two-point segment.
    """
    cache = Path(cfg.paths.interim) / "osm_walk.graphml"
    if cache.exists():
        graph = ox.load_graphml(cache)
    else:
        graph = ox.graph_from_place(cfg.study.osm_place, network_type="walk", simplify=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache)
    return ox.project_graph(graph, to_crs=cfg.crs.projected)


def to_simple_undirected(graph: nx.MultiDiGraph) -> nx.Graph:
    """Collapse to a simple undirected graph, keeping the *shortest* of parallel edges.

    (``nx.Graph(multigraph)`` would silently keep whichever parallel edge came last.)
    """
    simple = nx.Graph()
    simple.add_nodes_from((n, {"x": d["x"], "y": d["y"]}) for n, d in graph.nodes(data=True))
    for u, v, length in graph.edges(data="length"):
        if u == v:
            continue
        if not simple.has_edge(u, v) or simple[u][v]["length"] > length:
            simple.add_edge(u, v, length=length)
    return simple
