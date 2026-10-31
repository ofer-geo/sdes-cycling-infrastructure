"""Render the README figures from the processed outputs into docs/figures/.

    python scripts/make_figures.py [--only NAME ...]

Each figure reads only the columns it needs (the processed files are several GB in memory).
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from sdes import viz
from sdes.config import Config
from sdes.geo import Layers
from sdes.io import read_geoparquet


def route_choice_file(processed: Path) -> Path:
    """The full run if it exists, else the largest --limit run."""
    full = processed / "route_choice.parquet"
    if full.exists():
        return full
    partial = sorted(processed.glob("route_choice_[0-9]*.parquet"),
                     key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not partial:
        raise FileNotFoundError("run scripts/run_route_choice.py first")
    return partial[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    cfg = Config.load()
    processed, out = cfg.paths.processed, cfg.paths.figures
    out.mkdir(parents=True, exist_ok=True)
    layers = Layers(cfg)
    viz.apply_style()

    def wanted(name: str) -> bool:
        return not args.only or name in args.only

    def save(fig, name: str) -> None:
        fig.savefig(out / f"{name}.png")
        plt.close(fig)
        print(f"  {name}.png", flush=True)

    t0 = time.perf_counter()
    trips_file = processed / "trips_clean.parquet"

    if wanted("funnels"):
        reports = {
            "Trips": pd.read_csv(processed / "trips_clean_report.csv"),
            "Parking events": pd.read_csv(processed / "parking_events_report.csv"),
        }
        save(viz.funnel(reports), "funnels")

    if wanted("hourly_trips"):
        trips = pd.read_parquet(trips_file, columns=["day_type", "start_hour"])
        save(viz.hourly_trips(trips), "hourly_trips")

    if wanted("infra_share") or wanted("route_example"):
        path = route_choice_file(processed)
        print(f"  (route choice from {path.name})")
        cols = ["length_m", "matched_length_m", "sp_length_m", "infra_m", "sp_infra_m",
                "infra_pct", "sp_infra_pct", "detour_ratio", "reconnected"]
        if wanted("infra_share"):
            save(viz.infra_share_by_length(pd.read_parquet(path, columns=cols)), "infra_share")
        if wanted("route_example"):
            routes = read_geoparquet(path, columns=[*cols, "trip_id", "matched", "shortest_path"])
            save(viz.route_example(viz.pick_example(routes), layers.bike_lanes), "route_example")

    if wanted("speed"):
        speed = pd.read_parquet(processed / "speed.parquet",
                                columns=["length_m", "inside_m", "inside_s", "outside_m",
                                         "outside_s"])
        save(viz.speed_by_length(speed), "speed")

    if wanted("parking"):
        events = pd.read_parquet(processed / "parking_events.parquet",
                                 columns=["end_reason", "dist_to_infra_m", "area", "duration_s"])
        save(viz.parking_by_distance(events), "parking")

    if wanted("density_map"):
        n_trips = 60_000
        trips = read_geoparquet(trips_file, columns=[], limit=n_trips)
        xy = shapely.get_coordinates(trips.geometry.values)
        xy = xy[np.random.default_rng(0).random(len(xy)) < 0.5]  # thin for rendering speed
        save(viz.density_map(xy, layers.city_boundary, layers.bike_lanes, len(trips)),
             "density_map")

    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
