"""Stage 1 of the trips pipeline: clean the raw trips and add attributes.

    python scripts/run_trips.py
"""

import time

from sdes.config import Config
from sdes.geo import Layers
from sdes.io import write_geoparquet
from sdes.trips.filtering import add_trip_attributes, clean_all


def main() -> None:
    cfg = Config.load()
    out = cfg.paths.processed
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trips, report = clean_all(cfg, cfg.paths.interim / "trips_2023-07")
    trips = add_trip_attributes(trips, Layers(cfg))
    write_geoparquet(trips, out / "trips_clean.parquet")
    report.to_frame().to_csv(out / "trips_clean_report.csv", index=False)

    print(report)
    print(f"\n{len(trips):,} trips written in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
