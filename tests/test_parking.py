import geopandas as gpd
import pandas as pd
import pytest

from sdes.config import Config
from sdes.parking.events import ParkingEventBuilder

CFG = Config.load()


def records(rows) -> gpd.GeoDataFrame:
    df = pd.DataFrame(rows, columns=["device_id", "vehicle_state", "minute", "x"])
    df["event_time"] = pd.Timestamp("2023-07-02") + pd.to_timedelta(df["minute"], "min")
    df["event_types"] = "x"
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], [0] * len(df)),
                            crs=CFG.crs.projected)


def test_build_events_folds_reservations_and_drops_orphans():
    rec = records([
        ("A", "available", 0, 0),
        ("A", "reserved", 30, 5),
        ("A", "available", 35, 5),        # reservation cancelled: same event continues
        ("A", "reserved", 60, 3),
        ("A", "on_trip", 62, 3),          # reservation fulfilled: event ends with a ride
        ("A", "available", 80, 1000),
        ("A", "non_operational", 200, 1000),
        ("B", "reserved", 0, 0),          # orphan reservation (no preceding 'available')
        ("B", "on_trip", 1, 0),
        ("B", "available", 20, 500),      # no following record: censored
    ])
    builder = ParkingEventBuilder(CFG, layers=None)
    ev = builder.build_events(rec)

    assert ev[["device_id", "end_reason", "n_records"]].values.tolist() == [
        ["A", "on_trip", 4], ["A", "non_operational", 1]]
    assert (ev["end_time"] - ev["start_time"]).dt.total_seconds().tolist() == [62 * 60, 120 * 60]
    removed = dict(builder.report.steps)
    assert removed["starts with an orphan reservation"] == 1
    assert removed["no following record (censored at month end)"] == 1


def test_vehicle_seen_elsewhere_ends_event_as_unknown():
    rec = records([("A", "available", 0, 0), ("A", "available", 60, 400),
                   ("A", "on_trip", 90, 400)])
    ev = ParkingEventBuilder(CFG, layers=None).build_events(rec)
    assert ev["end_reason"].tolist() == ["unknown", "on_trip"]


def test_short_interruptions_are_merged():
    rec = records([("A", "available", 0, 0), ("A", "on_trip", 60, 0),
                   ("A", "available", 62, 10),   # 2-min "ride" that went nowhere
                   ("A", "on_trip", 120, 10)])
    builder = ParkingEventBuilder(CFG, layers=None)
    ev = builder.merge_interruptions(builder.build_events(rec))
    assert len(ev) == 1
    assert (ev["end_time"] - ev["start_time"]).iloc[0] == pd.Timedelta(minutes=120)
    assert ev["end_reason"].iloc[0] == "on_trip"


@pytest.mark.parametrize("gap_min", [6, 60 * 24 + 2])
def test_long_gaps_are_not_merged(gap_min):
    # a gap of 1 day + 2 min must not look like 2 minutes (old `.seconds` bug)
    rec = records([("A", "available", 0, 0), ("A", "on_trip", 60, 0),
                   ("A", "available", 60 + gap_min, 0), ("A", "on_trip", 200 + gap_min, 0)])
    builder = ParkingEventBuilder(CFG, layers=None)
    assert len(builder.merge_interruptions(builder.build_events(rec))) == 2
