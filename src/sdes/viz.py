"""Figures for the README / showcase notebook (static PNGs, one consistent style).

Every plotting function takes already-loaded data and returns a ``Figure`` so it can be
used both from ``scripts/make_figures.py`` and interactively in the notebook.
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# ---------------------------------------------------------------- style tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # chosen route / inside infrastructure
ORANGE = "#eb6834"    # shortest path / outside infrastructure
AQUA = "#1baf7a"
YELLOW = "#eda100"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DENSITY_CMAP = LinearSegmentedColormap.from_list("density", BLUE_RAMP)

DAY_TYPES = ["working day", "Friday", "Saturday"]
AREAS = ["Center", "North", "East", "South"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 22, "legend.frameon": False, "lines.linewidth": 2,
        "lines.markersize": 5, "lines.solid_capstyle": "round",
    })


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _end_label(ax, x, y, text, color) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK, fontsize=9, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"})
    ax.plot([x], [y], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=5)


def length_bins(length_m: pd.Series, edges_km=(0, 1, 2, 3, 4, 5, 15)) -> pd.Series:
    labels = [f"{a}-{b}" if b < 15 else f"{a}+" for a, b in zip(edges_km[:-1], edges_km[1:],
                                                               strict=True)]
    return pd.cut(length_m / 1000, edges_km, labels=labels, right=False)


# ---------------------------------------------------------------- funnels
def funnel(reports: dict[str, pd.DataFrame]) -> Figure:
    """One panel per cleaning funnel (FilterReport CSVs): rows remaining after each step."""
    fig, axes = plt.subplots(1, len(reports), figsize=(6.2 * len(reports), 4.2),
                             gridspec_kw={"wspace": 1.0})
    for ax, (name, rep) in zip(np.atleast_1d(axes), reports.items(), strict=True):
        rep = rep.iloc[::-1]
        y = np.arange(len(rep))
        share = rep["remaining"] / rep["remaining"].iloc[-1]
        ax.barh(y, rep["remaining"], height=0.72, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(y, rep["step"].str.replace(" (", "\n(", regex=False), fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
        for yi, (n, s) in enumerate(zip(rep["remaining"], share, strict=True)):
            ax.text(n, yi, f"  {n:,.0f} ({s:.0%})", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, rep["remaining"].max() * 1.45)
        ax.set_title(name)
        _subtitle(ax, "records remaining after each cleaning rule")
    return fig


# ---------------------------------------------------------------- trips
def hourly_trips(trips: pd.DataFrame) -> Figure:
    """Share of each day type's trips by start hour (thesis Fig. 6.2)."""
    share = (trips.groupby(["day_type", "start_hour"]).size()
             .groupby(level=0).transform(lambda s: s / s.sum() * 100).unstack(0))
    n = trips["day_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for day, color in zip(DAY_TYPES, CATEGORICAL, strict=False):
        ax.plot(share.index, share[day], color=color, label=f"{day} (n={n[day]:,})")
        _end_label(ax, 23, share[day].iloc[-1], day, color)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(0, 25.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("% of the day type's trips")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("When are e-scooters ridden?")
    _subtitle(ax, "Trips by start hour, July 2023")
    return fig


def infra_share_by_length(routes: pd.DataFrame) -> Figure:
    """Share of the route on cycling infrastructure: chosen route vs shortest path
    (thesis Fig. 6.4a), with the detour ratio by length beside it (Fig. 6.5b)."""
    b = length_bins(routes["length_m"])
    g = routes.groupby(b, observed=True)
    stats = pd.DataFrame({
        "chosen": g["infra_m"].sum() / g["matched_length_m"].sum() * 100,
        "sp": g["sp_infra_m"].sum() / g["sp_length_m"].sum() * 100,
        "detour": (g["detour_ratio"].mean() - 1) * 100,
        "n": g.size(),
    })
    x = np.arange(len(stats))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3,
                                                                        "width_ratios": [3, 2]})
    w = 0.38
    ax.bar(x - w / 2, stats["chosen"], w, color=BLUE, label="chosen route",
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x + w / 2, stats["sp"], w, color=ORANGE, label="shortest path",
           edgecolor=SURFACE, linewidth=2)
    for xi, (c, s) in enumerate(zip(stats["chosen"], stats["sp"], strict=True)):
        ax.text(xi - w / 2, c + 1, f"{c:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, s + 1, f"{s:.0f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(stats.index, stats["n"], strict=True)],
                  fontsize=8)
    ax.set_xlabel("route length (km)")
    ax.set_ylabel("% of route on cycling infrastructure")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Riders detour onto cycling infrastructure")
    _subtitle(ax, "Map-matched chosen route vs. shortest path between the same nodes")

    ax2.plot(x, stats["detour"], color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2)
    for xi, d in zip(x, stats["detour"], strict=True):
        ax2.annotate(f"+{d:.0f}%", (xi, d), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8)
    ax2.set_xticks(x, stats.index, fontsize=8)
    ax2.set_xlabel("route length (km)")
    ax2.set_ylabel("chosen route longer than shortest path (%)")
    ax2.set_ylim(0, stats["detour"].max() * 1.3)
    ax2.set_title("...and longer trips detour more")
    _subtitle(ax2, "Mean detour ratio by route length")
    return fig


def pick_example(routes: pd.DataFrame) -> pd.Series:
    """A clear, typical detour: 1.5-3 km, modest detour, big gain in infrastructure share."""
    c = routes[routes["length_m"].between(1500, 3000) & routes["detour_ratio"].between(1.08, 1.35)
               & ~routes["reconnected"]]
    gain = c["infra_pct"] - c["sp_infra_pct"]
    return c.loc[gain.idxmax()]


def route_example(row: pd.Series, bike_lanes: gpd.GeoDataFrame, pad_m: float = 250) -> Figure:
    """One trip: raw GPS, map-matched route and shortest path over the bike network
    (thesis Fig. 6.6)."""
    geoms = [row["geometry"], row["matched"], row["shortest_path"]]
    minx, miny, maxx, maxy = shapely.bounds(shapely.union_all(geoms))
    span = max(maxx - minx, maxy - miny) / 2 + pad_m
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lanes = bike_lanes.cx[cx - span:cx + span, cy - span:cy + span]
    lanes.plot(ax=ax, color="#9fd9c2", linewidth=4, capstyle="round", zorder=1)
    gpd.GeoSeries([row["shortest_path"]]).plot(ax=ax, color=ORANGE, linewidth=2.5,
                                                linestyle=(0, (4, 2)), zorder=3)
    gpd.GeoSeries([row["matched"]]).plot(ax=ax, color=BLUE, linewidth=2.5, zorder=4)
    xy = shapely.get_coordinates(row["geometry"])
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color=INK_2, zorder=5, linewidths=0)
    for (x, y), label in [(xy[0], "start"), (xy[-1], "end")]:
        ax.plot(x, y, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=6)
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=9,
                    fontweight="bold")
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.set_axis_off()
    handles = [
        mpl.lines.Line2D([], [], color="#9fd9c2", lw=4, label="cycling infrastructure"),
        mpl.lines.Line2D([], [], color=INK_2, marker="o", lw=0, ms=3, label="GPS pings"),
        mpl.lines.Line2D([], [], color=BLUE, lw=2.5,
                         label=f"chosen route ({row['matched_length_m'] / 1000:.2f} km, "
                               f"{row['infra_pct']:.0f}% on infra)"),
        mpl.lines.Line2D([], [], color=ORANGE, lw=2.5, ls=(0, (4, 2)),
                         label=f"shortest path ({row['sp_length_m'] / 1000:.2f} km, "
                               f"{row['sp_infra_pct']:.0f}% on infra)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2, fontsize=8)
    ax.set_title("Map-matching and the shortest-path alternative")
    _subtitle(ax, "One July 2023 trip; the rider adds distance to stay on the bike lane")
    return fig


def speed_by_length(speed: pd.DataFrame) -> Figure:
    """Speed inside vs outside cycling infrastructure by route length (thesis Fig. 6.7).

    Speeds are distance-weighted (total metres / total seconds per bin), which is robust
    to the very short segments whose per-trip speeds are noisy.
    """
    g = speed.groupby(length_bins(speed["length_m"]), observed=True)
    stats = pd.DataFrame({
        "inside": g["inside_m"].sum() / g["inside_s"].sum() * 3.6,
        "outside": g["outside_m"].sum() / g["outside_s"].sum() * 3.6,
        "n": g.size(),
    })
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for col, color, label in [("inside", BLUE, "inside infrastructure"),
                              ("outside", ORANGE, "outside infrastructure")]:
        ax.plot(x, stats[col], color=color, marker="o", ms=8, mec=SURFACE, mew=2)
        _end_label(ax, x[-1], stats[col].iloc[-1], label, color)
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(stats.index, stats["n"], strict=True)],
                  fontsize=8)
    ax.set_xlim(-0.3, len(x) + 0.7)
    ax.set_xlabel("route length (km)")
    ax.set_ylabel("average speed (km/h)")
    ax.set_title("Riders are faster on cycling infrastructure, on longer trips")
    _subtitle(ax, "Distance-weighted speed of route segments inside / outside a 25 m buffer")
    return fig


# ---------------------------------------------------------------- parking
def parking_by_distance(events: pd.DataFrame, groups_m=(0, 25, 50, 75, 100)) -> Figure:
    """Mean parking duration before a ride by distance to infrastructure and area
    (thesis Fig. 6.12): shorter wait = higher demand."""
    ev = events[(events["end_reason"] == "on_trip") & (events["dist_to_infra_m"] < groups_m[-1])]
    labels = [f"{a}-{b}" for a, b in zip(groups_m[:-1], groups_m[1:], strict=True)]
    band = pd.cut(ev["dist_to_infra_m"], groups_m, labels=labels, right=False)
    hours = ev.groupby([ev["area"], band], observed=True)["duration_s"].mean().div(3600).unstack()
    n = ev["area"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 4.4))
    areas = [a for a in AREAS if a in hours.index]
    x = np.arange(len(areas))
    w = 0.8 / len(labels)
    ramp = ["#256abf", "#5598e7", "#86b6ef", "#b7d3f6"]
    for i, (lab, color) in enumerate(zip(labels, ramp, strict=True)):
        ax.bar(x - 0.4 + w * (i + 0.5), hours.loc[areas, lab], w, color=color,
               edgecolor=SURFACE, linewidth=2, label=f"{lab} m")
    ax.set_xticks(x, [f"{a}\nn={n[a]:,}" for a in areas])
    ax.set_ylabel("mean parking time before the next ride (h)")
    ax.legend(title="distance to cycling infrastructure", ncols=4, loc="upper left",
              fontsize=8, title_fontsize=8)
    ax.set_ylim(0, hours.to_numpy().max() * 1.25)
    ax.set_title("Scooters parked next to a bike lane are picked up sooner")
    _subtitle(ax, "Parking events ending in a ride, within 100 m of cycling infrastructure")
    return fig


def density_map(xy: np.ndarray, city: shapely.Geometry, bike_lanes: gpd.GeoDataFrame,
                n_trips: int, gridsize: int = 220) -> Figure:
    """Where e-scooters ride: hexbin density of GPS pings over the bike network."""
    fig, ax = plt.subplots(figsize=(7, 9))
    gpd.GeoSeries([city]).plot(ax=ax, color="#f0efec", edgecolor=GRID, linewidth=0.8)
    hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=gridsize, bins="log", mincnt=3,
                   cmap=DENSITY_CMAP, linewidths=0, zorder=2)
    bike_lanes.plot(ax=ax, color=ORANGE, linewidth=0.5, alpha=0.8, zorder=3)
    minx, miny, maxx, maxy = city.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cb = fig.colorbar(hb, ax=ax, shrink=0.45, pad=0.01)
    cb.set_label("GPS pings per cell (log)", color=INK_2)
    cb.outline.set_visible(False)
    ax.legend(handles=[mpl.lines.Line2D([], [], color=ORANGE, lw=1.5,
                                        label="cycling infrastructure")], loc="lower left")
    ax.set_title("Where Tel Aviv-Yafo rides e-scooters")
    _subtitle(ax, f"GPS pings of {n_trips:,} cleaned trips, July 2023")
    return fig
