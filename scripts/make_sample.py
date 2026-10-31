"""Write a small, anonymised sample of the raw MDS exports to data/sample/.

    python scripts/make_sample.py [--day 2023-07-12] [--vehicles 10] [--max-trips 100]

The full data belongs to the Tel Aviv-Yafo municipality and is not redistributed. The sample
keeps the raw CSV layout (so `sdes.io` reads it unchanged) but removes anything that identifies
an operator or a vehicle:

* vehicle ids -> ``vehicle_01`` ...; trip ids -> fresh random UUIDs (random, not hashed, so they
  can't be matched back to the originals; the mapping is never saved);
* ``accuracy`` and ``mds_version`` are blanked (both differ between operators);
* timestamps are rounded to whole seconds and coordinates to 5 decimals (~1 m): operators
  report different precision;
* the export's row index is dropped.
"""

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from sdes.config import REPO_ROOT, Config
from sdes.io import parse_route

OUT = REPO_ROOT / "data" / "sample"


def route_repr(lon, lat, ts_ms) -> str:
    """Rebuild the MDS ``route`` field in the export's own (Python-repr) style."""
    features = [{"type": "Feature", "properties": {"timestamp": int(t)},
                 "geometry": {"type": "Point", "coordinates": [float(x), float(y)]}}
                for x, y, t in zip(lon, lat, ts_ms, strict=True)]
    return repr({"type": "FeatureCollection", "features": features})


def pick_vehicles(cfg: Config, day: pd.Timestamp, n: int, rng) -> list[str]:
    """Vehicles with a handful of clean trips that day (so every pipeline stage has data)."""
    clean = pd.read_parquet(cfg.paths.processed / "trips_clean.parquet",
                            columns=["device_id", "start_time"])
    clean = clean[clean["start_time"].dt.normalize() == day]
    counts = clean["device_id"].value_counts()
    candidates = counts[counts.between(5, 12)].index.to_numpy()
    return list(rng.choice(candidates, size=n, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="2023-07-12")
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--max-trips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = Config.load()
    rng = np.random.default_rng(args.seed)
    day = pd.Timestamp(args.day)
    next_day = day + pd.Timedelta(days=1)

    devices = pick_vehicles(cfg, day, args.vehicles, rng)
    vehicle_alias = {d: f"vehicle_{i:02d}" for i, d in enumerate(sorted(devices), 1)}

    # ---- trips: every raw trip of those vehicles that day (incl. ones cleaning will drop)
    trips = ds.dataset(cfg.paths.interim / "trips_2023-07").to_table(
        filter=pc.field("device_id").isin(devices)
        & (pc.field("start_time") >= day) & (pc.field("start_time") < next_day)).to_pandas()
    trips = trips.sort_values("start_time").head(args.max_trips)
    trip_alias = {t: str(uuid.UUID(bytes=rng.bytes(16), version=4)) for t in trips["trip_id"]}

    sample_trips = pd.DataFrame({
        "vehicle_type": trips["vehicle_type"],
        "propulsion_types": "electric",
        "device_id": trips["device_id"].map(vehicle_alias),
        "trip_id": trips["trip_id"].map(trip_alias),
        "trip_duration": trips["trip_duration"],
        "trip_distance": trips["trip_distance"],
        "start_time": trips["start_time"].dt.floor("s"),
        "end_time": trips["end_time"].dt.floor("s"),
        "accuracy": np.nan,
        "route": [route_repr(np.round(lon, 5), np.round(lat, 5),
                             np.round(np.asarray(ts) / 1000) * 1000)
                  for lon, lat, ts in zip(trips["lon"], trips["lat"], trips["ts"], strict=True)],
        "mds_version": np.nan,
    })

    # ---- status changes of the same vehicles that day
    status = pd.read_parquet(cfg.paths.interim / "status_changes_2023-07.parquet")
    status = status[status["device_id"].isin(devices)
                    & (status["event_time"] >= day) & (status["event_time"] < next_day)]
    sample_status = pd.DataFrame({
        "vehicle_state": status["vehicle_state"].astype(str),
        "event_types": status["event_types"].astype(str),
        "event_time": status["event_time"].dt.floor("s"),
        "device_id": status["device_id"].astype(str).map(vehicle_alias),
        "vehicle_type": status["vehicle_type"].astype(str),
        "propulsion_types": "electric",
        "lat": status["lat"].round(5), "lon": status["lon"].round(5),
        "mds_version": np.nan,
    }).sort_values(["event_time", "device_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    sample_trips.to_csv(OUT / "trips_sample.csv", index=False)
    sample_status.to_csv(OUT / "status_changes_sample.csv", index=False)

    # sanity: nothing original survives, and the routes still parse
    text = (OUT / "trips_sample.csv").read_text() + (OUT / "status_changes_sample.csv").read_text()
    leaked = [x for x in [*devices, *trip_alias] if x in text]
    assert not leaked, f"original ids leaked: {leaked[:3]}"
    parse_route(sample_trips["route"].iloc[0])
    print(f"{len(sample_trips)} trips, {len(sample_status)} status changes, "
          f"{len(devices)} vehicles, {day:%Y-%m-%d} -> {Path(OUT).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
