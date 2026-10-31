"""One-off conversion of the raw MDS exports to Parquet (data/interim).

    python scripts/convert_raw.py [--only trips|status]
"""

import argparse
import time

from sdes.config import Config
from sdes.io import convert_trips, load_status_changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["trips", "status"])
    args = parser.parse_args()
    cfg = Config.load()
    interim = cfg.paths.interim
    interim.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "status"):
        for name, src in [("status_changes_2023-06", cfg.paths.raw_status_changes_previous),
                          ("status_changes_2023-07", cfg.paths.raw_status_changes)]:
            t0 = time.perf_counter()
            df = load_status_changes(src)
            df.to_parquet(interim / f"{name}.parquet", compression="zstd", index=False)
            print(f"{name}: {len(df):,} rows in {time.perf_counter() - t0:.0f}s")

    if args.only in (None, "trips"):
        t0 = time.perf_counter()
        n = convert_trips(cfg.paths.raw_trips, interim / "trips_2023-07")
        print(f"trips: {n:,} rows in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
