"""Build the parking-events dataset from Status Changes.

    python scripts/run_parking.py [--no-seed]
"""

import argparse
import time

import pandas as pd

from sdes.config import Config
from sdes.geo import Layers
from sdes.io import write_geoparquet
from sdes.parking.events import ParkingEventBuilder, load_trip_starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-seed", action="store_true",
                        help="don't use the previous month to seed vehicles' initial state")
    args = parser.parse_args()
    cfg = Config.load()
    interim, out = cfg.paths.interim, cfg.paths.processed
    t0 = time.perf_counter()

    status = pd.read_parquet(interim / "status_changes_2023-07.parquet")
    seed = None if args.no_seed else pd.read_parquet(interim / "status_changes_2023-06.parquet")
    trip_starts = load_trip_starts(interim / "trips_2023-07", cfg)

    builder = ParkingEventBuilder(cfg, Layers(cfg))
    events = builder.run(status, seed, trip_starts)
    write_geoparquet(events, out / "parking_events.parquet")
    builder.report.to_frame().to_csv(out / "parking_events_report.csv", index=False)

    print(builder.report)
    print("\nend reasons:\n" + events["end_reason"].value_counts(normalize=True)
          .mul(100).round(1).to_string())
    rides = events[events["end_reason"] == "on_trip"]
    print(f"\nevents ending in a ride: {len(rides):,}; "
          f"mean parking {rides['duration_s'].mean() / 3600:.2f} h, "
          f"median {rides['duration_s'].median() / 3600:.2f} h")
    print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
