"""Generate notebooks/results_overview.ipynb (source of truth is this file, so the notebook
stays reviewable in diffs). Execute it with:

    jupyter nbconvert --to notebook --execute --inplace notebooks/results_overview.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Cycling infrastructure and shared e-scooter behaviour — results overview

Tel Aviv-Yafo, July 2023. ~1.36 M MDS trips and ~3.5 M vehicle status changes, provided by the
municipality. This notebook loads the outputs of the pipeline (`scripts/run_*.py`) and walks
through the three questions of the thesis:

1. **Route choice** — do riders go out of their way to use cycling infrastructure?
2. **Speed** — are they faster on it?
3. **Demand** — are scooters parked next to it picked up sooner?

All heavy lifting lives in the `sdes` package; every figure here is a function in `sdes.viz`."""),
    code("""import pandas as pd
from IPython.display import display

from sdes import viz
from sdes.config import Config
from sdes.geo import Layers
from sdes.io import read_geoparquet

cfg = Config.load()
P = cfg.paths.processed
layers = Layers(cfg)
viz.apply_style()"""),
    md("""## 1. Data cleaning

Every rule is logged by `FilterReport`, so each funnel is auditable. The biggest cut is the
*low-resolution route* rule: a trip is kept only if no two consecutive GPS pings are more than
70 m apart (10 s ping interval × 25 km/h cap), otherwise its geometry can't be trusted for
route-level analysis."""),
    code("""reports = {"Trips": pd.read_csv(P / "trips_clean_report.csv"),
           "Parking events": pd.read_csv(P / "parking_events_report.csv")}
viz.funnel(reports);"""),
    code("""trips = pd.read_parquet(P / "trips_clean.parquet",
                        columns=["day_type", "start_hour", "length_m", "trip_duration",
                                 "speed_kmh"])
# quartiles, not mean/max: a handful of operator records carry a timestamp in trip_duration
summary = pd.DataFrame({"length (km)": trips["length_m"] / 1000,
                        "duration (min)": trips["trip_duration"] / 60,
                        "speed (km/h)": trips["speed_kmh"]})
display(summary.quantile([0.25, 0.5, 0.75]).rename(index=lambda q: f"{q:.0%}").round(1))
viz.hourly_trips(trips);"""),
    md("""## 2. Route choice

Each trip is **map-matched** to the OpenStreetMap walk network (buffer the GPS track by 25 m,
keep edges ≥ 90 % inside the buffer, bridge disconnected pieces, take the shortest path through
that sub-network). The result is compared with the **shortest path** between the same two
nodes on the full network; both are intersected with a 25 m buffer around the municipal
cycling-infrastructure layer."""),
    code("""routes = read_geoparquet(P / "route_choice.parquet", columns=[
    "trip_id", "length_m", "matched_length_m", "sp_length_m", "infra_m", "sp_infra_m",
    "infra_pct", "sp_infra_pct", "detour_ratio", "reconnected", "matched", "shortest_path"])
display(pd.read_csv(P / "route_choice_report.csv"))
print(f"share on infrastructure: chosen {routes.infra_pct.mean():.1f}% "
      f"vs shortest path {routes.sp_infra_pct.mean():.1f}%; "
      f"mean detour {100 * (routes.detour_ratio.mean() - 1):.1f}%")"""),
    code("viz.infra_share_by_length(routes);"),
    code("viz.route_example(viz.pick_example(routes), layers.bike_lanes);"),
    md("""## 3. Speed inside vs. outside cycling infrastructure

Each GPS route is cut into runs of pings inside / outside the infrastructure buffer, with the
transition placed midway between the last ping of one run and the first of the next. Runs
shorter than 80 m are treated as GPS noise and absorbed by their neighbour."""),
    code("""speed = pd.read_parquet(P / "speed.parquet")
print(f"mean {speed.moving_kmh.mean():.2f} km/h, median {speed.moving_kmh.median():.2f} km/h")
viz.speed_by_length(speed);"""),
    md("""## 4. Parking demand

A *parking event* is the time a scooter stands ready to ride at one spot (built from the Status
Changes feed: consecutive `available`/`reserved` records within 25 m). For events that end in
a ride, a shorter parking time means higher demand at that location."""),
    code("""events = pd.read_parquet(P / "parking_events.parquet",
                         columns=["end_reason", "dist_to_infra_m", "area", "duration_s"])
display(events.end_reason.value_counts(normalize=True).mul(100).round(1).rename("% of events"))
viz.parking_by_distance(events);"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "results_overview.ipynb"
nbf.write(nb, out)
print(out)
