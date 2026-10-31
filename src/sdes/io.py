"""Loading the raw MDS exports (Trips, Status Changes) and converting them to Parquet.

The raw exports are large CSVs shipped inside archives (the July trips file is ~15 GB
uncompressed), so everything here streams: archives are decompressed to a pipe and
read in chunks, never unpacked to disk.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRIP_COLUMNS = [
    "device_id", "trip_id", "vehicle_type", "trip_duration", "trip_distance",
    "start_time", "end_time", "accuracy", "route", "mds_version",
]
STATUS_COLUMNS = [
    "device_id", "vehicle_state", "event_types", "event_time", "vehicle_type", "lat", "lon",
    "mds_version",
]


# --------------------------------------------------------------------------- archives

def _bsdtar() -> str:
    candidates = [Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "tar.exe"]
    if sys.platform != "win32":
        candidates = [Path(p) for p in (shutil.which("bsdtar"), shutil.which("tar")) if p]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError("bsdtar is required to read .rar archives")


@contextlib.contextmanager
def open_tabular(path: str | Path):
    """Open a CSV that may be plain or packed in a .zip/.rar archive (single member)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "rb") as f:
            yield f
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            yield f
    elif suffix == ".rar":
        proc = subprocess.Popen([_bsdtar(), "-xOf", str(path)], stdout=subprocess.PIPE,
                                bufsize=1 << 24)
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
    else:
        raise ValueError(f"Unsupported file type: {path}")


def iter_csv_chunks(path: str | Path, usecols: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    with open_tabular(path) as f:
        yield from pd.read_csv(f, usecols=usecols, chunksize=chunksize)


# --------------------------------------------------------------------------- trips

def parse_route(route: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the MDS ``route`` FeatureCollection into (lon, lat, timestamp_ms) arrays.

    The export stores it as a Python ``repr`` (single quotes), so quotes are swapped
    before JSON parsing.
    """
    features = orjson.loads(route.replace("'", '"'))["features"]
    n = len(features)
    lon, lat = np.empty(n), np.empty(n)
    ts = np.empty(n, dtype=np.int64)
    for i, feat in enumerate(features):
        lon[i], lat[i] = feat["geometry"]["coordinates"][:2]
        ts[i] = feat["properties"]["timestamp"]
    return lon, lat, ts


_TRIPS_SCHEMA = pa.schema([
    ("trip_id", pa.string()),
    ("device_id", pa.string()),
    ("vehicle_type", pa.string()),
    ("trip_duration", pa.float64()),
    ("trip_distance", pa.float64()),
    ("start_time", pa.timestamp("s")),
    ("end_time", pa.timestamp("s")),
    ("accuracy", pa.float64()),
    ("mds_version", pa.float64()),
    ("n_points", pa.int32()),
    ("lon", pa.list_(pa.float64())),
    ("lat", pa.list_(pa.float64())),
    ("ts", pa.list_(pa.int64())),
])


def _convert_trip_chunk(args: tuple[pd.DataFrame, Path]) -> int:
    chunk, out_file = args
    lons, lats, tss = [], [], []
    for route in chunk["route"]:
        try:
            lon, lat, ts = parse_route(route) if isinstance(route, str) else ((), (), ())
        except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
            lon, lat, ts = (), (), ()
        lons.append(lon)
        lats.append(lat)
        tss.append(ts)
    chunk = chunk.drop(columns="route")
    chunk["start_time"] = pd.to_datetime(chunk["start_time"])
    chunk["end_time"] = pd.to_datetime(chunk["end_time"])
    chunk["n_points"] = np.fromiter((len(x) for x in lons), dtype=np.int32, count=len(lons))
    chunk["lon"], chunk["lat"], chunk["ts"] = lons, lats, tss
    table = pa.Table.from_pandas(chunk[_TRIPS_SCHEMA.names], schema=_TRIPS_SCHEMA,
                                 preserve_index=False)
    pq.write_table(table, out_file, compression="zstd")
    return len(chunk)


def convert_trips(archive: str | Path, out_dir: str | Path, chunksize: int = 20_000,
                  workers: int | None = None) -> int:
    """Stream the raw trips CSV into a partitioned Parquet dataset (one file per chunk).

    Route GeoJSON is parsed once here into coordinate/timestamp arrays, which is what
    every downstream step needs and is ~10x smaller than the raw text.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    total = 0
    chunks = iter_csv_chunks(archive, TRIP_COLUMNS, chunksize)
    jobs = ((chunk, out_dir / f"part-{i:04d}.parquet") for i, chunk in enumerate(chunks))
    with ProcessPoolExecutor(workers) as pool:
        # bounded look-ahead so the reader never holds more than ~2x workers chunks in memory
        pending = []
        for job in jobs:
            pending.append(pool.submit(_convert_trip_chunk, job))
            if len(pending) >= 2 * workers:
                total += pending.pop(0).result()
                print(f"  {total:,} trips converted", flush=True)
        for fut in pending:
            total += fut.result()
    print(f"  {total:,} trips converted", flush=True)
    return total


def load_trips(parquet_dir: str | Path, columns: list[str] | None = None,
               filters=None) -> pd.DataFrame:
    return pq.read_table(parquet_dir, columns=columns, filters=filters).to_pandas()


# --------------------------------------------------------------------------- GeoParquet

ROW_GROUP_SIZE = 20_000


def write_geoparquet(gdf, path: str | Path) -> None:
    """GeoParquet in row groups, so readers can take only the first N rows or a few columns."""
    gdf.to_parquet(path, compression="zstd", row_group_size=ROW_GROUP_SIZE)


def read_geoparquet(path: str | Path, columns: list[str] | None = None,
                    limit: int | None = None, geometry: str = "geometry"):
    """Read selected columns (and optionally only the first ``limit`` rows) of a GeoParquet.

    ``geopandas.read_parquet`` always materialises the whole file; with ~1 GB of route WKB
    that is too much for a 16 GB machine when other columns aren't needed.
    """
    pf = pq.ParquetFile(path)
    meta = _geo_metadata(pf)
    batches, n = [], 0
    for batch in pf.iter_batches(batch_size=ROW_GROUP_SIZE,
                                 columns=_with_geometry(columns, geometry)):
        batches.append(batch)
        n += batch.num_rows
        if limit is not None and n >= limit:
            break
    table = pa.Table.from_batches(batches, schema=batches[0].schema if batches else None)
    if limit is not None:
        table = table.slice(0, limit)
    return _to_geodataframe(table, meta, geometry)


def iter_geoparquet(path: str | Path, columns: list[str] | None = None,
                    chunk_rows: int = 25_000, geometry: str = "geometry"):
    """Yield consecutive GeoDataFrame chunks, so only one chunk is in memory at a time."""
    pf = pq.ParquetFile(path)
    meta = _geo_metadata(pf)
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=_with_geometry(columns, geometry)):
        yield _to_geodataframe(pa.Table.from_batches([batch]), meta, geometry)


def _geo_metadata(pf: pq.ParquetFile) -> dict:
    return orjson.loads(pf.schema_arrow.metadata[b"geo"])


def _with_geometry(columns: list[str] | None, geometry: str) -> list[str] | None:
    return columns if columns is None or geometry in columns else [*columns, geometry]


def _to_geodataframe(table: pa.Table, meta: dict, geometry: str):
    import geopandas as gpd
    import shapely

    geom_cols = set(meta["columns"])
    df = table.drop([c for c in table.column_names if c in geom_cols]).to_pandas()
    for col in geom_cols & set(table.column_names):
        df[col] = gpd.GeoSeries(shapely.from_wkb(table[col].to_numpy(zero_copy_only=False)),
                                crs=meta["columns"][col].get("crs"), index=df.index)
    return gpd.GeoDataFrame(df, geometry=geometry)


# --------------------------------------------------------------------------- status changes

def load_status_changes(archive: str | Path) -> pd.DataFrame:
    """Read a monthly Status Changes export (~3.5M rows; fits comfortably in memory)."""
    with open_tabular(archive) as f:
        df = pd.read_csv(
            f, usecols=STATUS_COLUMNS,
            dtype={"device_id": "string", "vehicle_state": "category",
                   "event_types": "category", "vehicle_type": "category"},
        )
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df
