"""Stage 2 of the trips pipeline: map-matching, shortest paths, infrastructure share.

    python scripts/run_route_choice.py [--limit N] [--chunk-rows 25000] [--workers N]

Trips are processed in chunks; each finished chunk is saved under data/interim, so an
interrupted run (e.g. killed for memory) resumes where it stopped. Only one chunk of trips
is in memory at a time.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sdes.config import Config
from sdes.geo import Layers
from sdes.io import iter_geoparquet, write_geoparquet
from sdes.report import FilterReport
from sdes.trips.network import load_walk_graph, to_simple_undirected
from sdes.trips.route_choice import run_route_choice

# per-ping timestamps aren't needed here and are ~1/3 of the file
COLUMNS = ["trip_id", "device_id", "start_time", "start_hour", "weekday", "day_type",
           "length_m", "trip_duration", "taz_start", "taz_end", "speed_kmh"]


def merge_parts(parts: list[Path], out_file: Path) -> None:
    """Concatenate the chunk files one at a time (constant memory)."""
    writer = None
    for part in parts:
        table = pq.read_table(part)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_file, schema, compression="zstd")
        writer.write_table(table.select(schema.names).cast(schema))
    if writer is not None:
        writer.close()


def merge_reports(report_files: list[Path]) -> FilterReport:
    report = FilterReport("Route-choice sample")
    for f in report_files:
        df = pd.read_csv(f)
        report.start(int(df.loc[0, "remaining"]))
        for step, removed in zip(df["step"].iloc[1:], df["removed"].iloc[1:], strict=True):
            report.add(step, int(removed))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="process only the first N trips (testing)")
    parser.add_argument("--chunk-rows", type=int, default=25_000)
    parser.add_argument("--workers", type=int,
                        help="cap parallel workers (default: as many as free RAM allows)")
    args = parser.parse_args()
    cfg = Config.load()
    out = cfg.paths.processed
    suffix = f"_{args.limit}" if args.limit else ""
    parts_dir = cfg.paths.interim / f"route_choice_parts{suffix}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    graph = to_simple_undirected(load_walk_graph(cfg))
    infra = Layers(cfg).bike_lanes_buffer
    print(f"network ready in {time.perf_counter() - t0:.0f}s "
          f"({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges)", flush=True)

    chunks = iter_geoparquet(out / "trips_clean.parquet", columns=COLUMNS,
                             chunk_rows=args.chunk_rows)
    done = 0
    for i, trips in enumerate(chunks):
        if args.limit:
            trips = trips.head(args.limit - done)
        part, part_report = parts_dir / f"part-{i:04d}.parquet", parts_dir / f"part-{i:04d}.csv"
        done += len(trips)
        if part_report.exists():  # the report is written last, so it marks a finished chunk
            print(f"chunk {i}: already done", flush=True)
        else:
            t1 = time.perf_counter()
            routes, report = run_route_choice(cfg, trips, graph, infra, workers=args.workers)
            write_geoparquet(routes.drop(columns="match_error", errors="ignore"), part)
            report.to_frame().to_csv(part_report, index=False)
            print(f"chunk {i}: {len(trips):,} trips -> {len(routes):,} routes "
                  f"in {time.perf_counter() - t1:.0f}s ({done:,} trips so far)", flush=True)
        if args.limit and done >= args.limit:
            break

    n_chunks = i + 1
    parts = [parts_dir / f"part-{k:04d}.parquet" for k in range(n_chunks)]
    merge_parts(parts, out / f"route_choice{suffix}.parquet")
    report = merge_reports([p.with_suffix(".csv") for p in parts])
    report.to_frame().to_csv(out / f"route_choice_report{suffix}.csv", index=False)

    summary = pq.read_table(out / f"route_choice{suffix}.parquet",
                            columns=["reconnected", "infra_pct", "sp_infra_pct",
                                     "detour_ratio"]).to_pandas()
    print(report)
    print(f"\nreconnected subgraphs: {summary['reconnected'].mean():.1%}")
    print(f"infra share  chosen: {summary['infra_pct'].mean():.1f}%  "
          f"SP: {summary['sp_infra_pct'].mean():.1f}%  "
          f"mean detour ratio: {summary['detour_ratio'].mean():.3f}")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
