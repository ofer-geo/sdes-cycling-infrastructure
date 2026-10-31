"""Stage 3 of the trips pipeline: speed inside vs. outside cycling infrastructure.

    python scripts/run_speed.py
"""

import time

from sdes.config import Config
from sdes.geo import Layers
from sdes.io import read_geoparquet
from sdes.report import FilterReport
from sdes.trips.speed import RouteSegmenter


def main() -> None:
    cfg = Config.load()
    out = cfg.paths.processed
    layers = Layers(cfg)
    t0 = time.perf_counter()

    trips = read_geoparquet(out / "trips_clean.parquet")
    try:
        slow = layers.slow_zones
    except FileNotFoundError:
        slow = None
        print("slow-speed zones layer not configured: skipping slow-zone correction")

    speed = RouteSegmenter(layers.bike_lanes_buffer, cfg.speed, slow).summarize(trips)
    speed = trips[["trip_id", "start_time", "start_hour", "weekday", "day_type",
                   "length_m"]].join(speed)

    report = FilterReport("Speed sample")
    report.start(len(speed))
    lo, hi = speed["moving_kmh"].quantile(list(cfg.speed.speed_quantiles))
    outlier = ~speed["moving_kmh"].between(lo, hi, inclusive="neither")
    report.add(f"speed outside [{lo:.1f}, {hi:.1f}] km/h (1st-99th pct)", int(outlier.sum()))
    speed = speed.loc[~outlier]

    speed.to_parquet(out / "speed.parquet", compression="zstd", index=False)
    report.to_frame().to_csv(out / "speed_report.csv", index=False)
    print(report)
    print(f"\nmean speed {speed['moving_kmh'].mean():.2f} km/h, "
          f"median {speed['moving_kmh'].median():.2f}")
    print(f"inside infra {speed['inside_kmh'].mean():.2f} km/h, "
          f"outside {speed['outside_kmh'].mean():.2f} km/h")
    print(f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
