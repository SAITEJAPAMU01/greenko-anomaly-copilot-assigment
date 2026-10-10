import asyncio
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline import run_pipeline, validate  # noqa: E402

DATA = ROOT / "data" / "asset_telemetry.csv"


@pytest.fixture(scope="module")
def res():
    return run_pipeline(pd.read_csv(DATA))


def test_finds_the_two_known_events(res):
    spans = {(e["asset_id"], e["start_index"], e["end_index"]) for e in res.events}
    assert ("SOLAR_INV_01", 67, 69) in spans
    assert ("WTG_07", 173, 174) in spans


def test_no_false_alarms_on_normal_hours(res):
    assert int(res.frame["anomaly"].sum()) == 5


def test_diagnosis_matches_signature(res):
    top = {e["asset_id"]: e["hypotheses"][0]["id"] for e in res.events}
    assert top["SOLAR_INV_01"] == "S1"   # heat + vibration while idle -> cooling fault
    assert top["WTG_07"] == "W1"         # heat + vibration + derate -> drivetrain


def test_explanation_is_grounded_in_data(res):
    e = next(e for e in res.events if e["asset_id"] == "WTG_07")
    assert "46.9" in e["explanation"] and "3.8 mm/s" in e["explanation"]
    assert len(e["recommended_checks"]) >= 3


def test_night_zero_power_is_not_flagged(res):
    solar = res.frame[res.frame["asset_id"] == "SOLAR_INV_01"]
    night = solar[solar["irradiance_wm2"] == 0]
    assert not night[~night["timestamp_id"].between(67, 69)]["anomaly"].any()


def test_daytime_trip_is_flagged():
    df = pd.read_csv(DATA)
    i = df[(df.asset_id == "SOLAR_INV_01") & (df.timestamp_id == 36)].index[0]  # noon, 800 W/m2
    df.loc[i:i + 1, "power_mw"] = 0.0
    r = run_pipeline(df)
    ev = [e for e in r.events if e["asset_id"] == "SOLAR_INV_01" and e["start_index"] <= 37 and e["end_index"] >= 36]
    assert ev and ev[0]["channels"].get("power output") == "low"
    assert ev[0]["hypotheses"][0]["id"] in {"S2", "S3"}


def test_validation_errors():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError):
        run_pipeline(pd.read_csv(DATA).head(10))


# ---- API layer: endpoint functions called directly --------------------------
def test_api_functions():
    pytest.importorskip("fastapi")
    from backend import app as api

    assert api.health() == {"status": "ok"}
    assert len(api.events()) == 2
    assert len(api.events(severity="critical", asset_id="WTG_07")) == 1
    ev = api.event("WTG_07-E1")
    assert ev["evidence"] and ev["hypotheses"]
    with pytest.raises(Exception):
        api.event("nope")
    out = api.ask(api.Query(text="what should I check first on the turbine?"))
    assert out["intent"] == "next_checks" and out["events"] == ["WTG_07-E1"]
    assert api.ask(api.Query(text="why")).get("intent") == "explain"
    assert api.asset("WTG_07")["series"][0]["expected_power_mw"] is not None

    class F:
        async def read(self):
            return DATA.read_bytes()
    up = asyncio.run(api.analyze(F()))
    assert len(up["events"]) == 2
