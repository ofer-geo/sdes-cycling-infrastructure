"""The published sample must load with the same readers as the raw export, and stay anonymous."""

import re

import pytest

from sdes.config import REPO_ROOT
from sdes.io import STATUS_COLUMNS, TRIP_COLUMNS, iter_csv_chunks, load_status_changes, parse_route

SAMPLE = REPO_ROOT / "data" / "sample"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample not generated")


def test_trips_sample_parses():
    trips = next(iter_csv_chunks(SAMPLE / "trips_sample.csv", TRIP_COLUMNS, chunksize=1000))
    assert len(trips) > 0
    lon, lat, ts = parse_route(trips["route"].iloc[0])
    assert len(lon) == len(lat) == len(ts) > 1


def test_status_sample_parses():
    status = load_status_changes(SAMPLE / "status_changes_sample.csv")
    assert set(STATUS_COLUMNS) <= set(status.columns)


def test_sample_is_anonymous():
    trips = next(iter_csv_chunks(SAMPLE / "trips_sample.csv", TRIP_COLUMNS, chunksize=1000))
    status = load_status_changes(SAMPLE / "status_changes_sample.csv")
    for ids in (trips["device_id"], status["device_id"]):
        assert ids.str.fullmatch(r"vehicle_\d{2}").all()
    assert trips["accuracy"].isna().all() and trips["mds_version"].isna().all()
    # at most 5 decimals in any coordinate
    text = "".join((SAMPLE / f).read_text() for f in ("trips_sample.csv",
                                                      "status_changes_sample.csv"))
    assert not re.search(r"\d+\.\d{6,}", text)
