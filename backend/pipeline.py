"""Anomaly-detection pipeline for renewable-asset telemetry.

Design in one paragraph
-----------------------
Raw channels (power, temperature, vibration) are NOT compared against a fixed
global band, because "normal" depends on the operating point: a solar inverter
making 0 MW at noon is a fault, at midnight it is normal. So we first build a
per-asset *expected-behaviour model* (physics-informed baselines), turn every
channel into a robust z-score of its residual, and then run two complementary
detectors on those residuals:

  1. transparent statistical rules (robust z-score + engineering thresholds)
  2. an IsolationForest over the residual feature vector (catches combinations
     that no single rule fires on)

The detectors are ensembled, consecutive flagged hours are merged into
*events*, and every event is explained in operator language by matching its
"deviation signature" against a small fault knowledge base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

REQUIRED_COLUMNS = [
    "timestamp_id", "asset_id", "hour", "irradiance_wm2",
    "wind_speed_ms", "temperature_c", "vibration_mm_s", "power_mw",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    z_thresh: float = 3.5          # robust-z needed for a statistical rule to fire
    if_gate_z: float = 2.5         # IsolationForest flag only counts if some channel is >= this
    if_contamination: float = 0.03  # expected anomaly share used to set the IF cut-off
    vib_warn: float = 2.5          # mm/s, absolute engineering threshold (site/ISO-class specific!)
    vib_alarm: float = 3.5         # mm/s
    temp_alarm: float = 60.0       # degC absolute
    power_alarm_frac: float = 0.20  # power shortfall >= 20 % of rated => "major" loss
    min_scale_power_frac: float = 0.03  # noise floor for power residual, fraction of rated
    min_scale_temp: float = 1.0    # degC
    min_scale_vib: float = 0.08    # mm/s
    merge_gap_hours: int = 1       # flagged points <= this apart are one event
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def asset_type(asset_id: str) -> str:
    a = asset_id.upper()
    if a.startswith("WTG") or "WIND" in a:
        return "wind"
    return "solar"


def _robust_z(x: pd.Series, min_scale: float) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / max(mad, min_scale)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    for c in REQUIRED_COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp_id"] = pd.to_numeric(df["timestamp_id"], errors="coerce")
    df = df.dropna(subset=["timestamp_id", "asset_id", "hour"])
    df["timestamp_id"] = df["timestamp_id"].astype(int)
    df["hour"] = df["hour"].astype(int)
    return df.sort_values(["asset_id", "timestamp_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Expected-behaviour models
# --------------------------------------------------------------------------- #
def expected_power(g: pd.DataFrame, kind: str, rated: float) -> pd.Series:
    """Expected MW given the driver (irradiance for solar, wind speed for wind).

    Both fits are robust: a first fit is made, the worst 5 % of residuals are
    discarded, and the model is refit - so a fault in the training window does
    not drag the baseline toward itself.
    """
    if kind == "solar":
        x, y = g["irradiance_wm2"].to_numpy(float), g["power_mw"].to_numpy(float)
        day = x > 50
        if day.sum() < 3:
            return pd.Series(np.zeros(len(g)), index=g.index)
        slope = np.median(y[day] / x[day])  # MW per W/m2, through the origin
        return pd.Series(np.clip(slope * x, 0, None), index=g.index)

    x, y = g["wind_speed_ms"].to_numpy(float), g["power_mw"].to_numpy(float)
    keep = np.ones(len(x), bool)
    for _ in range(2):
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(x[keep], y[keep])
        res = np.abs(y - iso.predict(x))
        keep = res <= np.quantile(res, 0.95)
    return pd.Series(iso.predict(x), index=g.index)


def temperature_baseline(g: pd.DataFrame) -> pd.Series:
    """Per-asset diurnal baseline: median temperature for the same hour-of-day."""
    by_hour = g.groupby("hour")["temperature_c"].transform("median")
    counts = g.groupby("hour")["temperature_c"].transform("count")
    fallback = g["temperature_c"].rolling(25, center=True, min_periods=3).median()
    return by_hour.where(counts >= 3, fallback).fillna(g["temperature_c"].median())


# --------------------------------------------------------------------------- #
# Feature engineering + detection for one asset
# --------------------------------------------------------------------------- #
def featurize(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = g.sort_values("timestamp_id").reset_index(drop=True).copy()
    kind = asset_type(g["asset_id"].iloc[0])
    rated = float(np.quantile(g["power_mw"].dropna(), 0.99)) or 1.0

    g["kind"] = kind
    g["expected_power_mw"] = expected_power(g, kind, rated)
    g["power_resid"] = g["power_mw"] - g["expected_power_mw"]
    g["expected_temp_c"] = temperature_baseline(g)
    g["temp_resid"] = g["temperature_c"] - g["expected_temp_c"]
    g["vib_baseline"] = g["vibration_mm_s"].median()
    g["idle"] = g["expected_power_mw"] < 0.02 * rated

    g["power_z"] = _robust_z(g["power_resid"], cfg.min_scale_power_frac * rated)
    g["temp_z"] = _robust_z(g["temp_resid"], cfg.min_scale_temp)
    g["vib_z"] = _robust_z(g["vibration_mm_s"], cfg.min_scale_vib)

    # temporal context: persistence and rate of change
    g["power_z_roll3"] = g["power_z"].rolling(3, min_periods=1).mean()
    g["temp_z_roll3"] = g["temp_z"].rolling(3, min_periods=1).mean()
    g["vib_z_roll3"] = g["vib_z"].rolling(3, min_periods=1).mean()
    g["temp_rate_c_per_h"] = g["temperature_c"].diff().fillna(0)
    g["rated_mw"] = rated
    return g


FEATURES = ["power_z", "temp_z", "vib_z", "power_z_roll3", "temp_z_roll3",
            "vib_z_roll3", "temp_rate_c_per_h"]


def detect(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = g.copy()
    rated = g["rated_mw"].iloc[0]

    # ---- 1. statistical rules (each is explainable on its own) ----------
    g["rule_power"] = g["power_z"].abs() >= cfg.z_thresh
    g["rule_temp"] = (g["temp_z"].abs() >= cfg.z_thresh) | (g["temperature_c"] >= cfg.temp_alarm)
    g["rule_vib"] = (g["vib_z"] >= cfg.z_thresh) | (g["vibration_mm_s"] >= cfg.vib_warn)
    g["rule_flag"] = g[["rule_power", "rule_temp", "rule_vib"]].any(axis=1)

    # ---- 2. IsolationForest on the residual feature vector ---------------
    X = g[FEATURES].to_numpy(float)
    iso = IsolationForest(n_estimators=300, contamination="auto",
                          random_state=cfg.random_state).fit(X)
    score = -iso.score_samples(X)  # higher = more anomalous
    g["if_score"] = score
    cut = np.quantile(score, 1 - cfg.if_contamination)
    g["if_flag"] = score >= cut

    # ---- 3. ensemble ------------------------------------------------------
    max_z = g[["power_z", "temp_z", "vib_z"]].abs().max(axis=1)
    g["max_abs_z"] = max_z
    g["anomaly"] = g["rule_flag"] | (g["if_flag"] & (max_z >= cfg.if_gate_z))

    # per-point channel states for the explainer: -1 low, 0 normal, +1 high
    def state(z, rule):
        return np.where(rule, np.sign(z), 0).astype(int)
    g["power_state"] = state(g["power_z"], g["rule_power"])
    g["temp_state"] = state(g["temp_z"], g["rule_temp"])
    g["vib_state"] = state(g["vib_z"], g["rule_vib"])
    # a point flagged only by IF/ensemble still gets channel states from a softer threshold
    soft = g["anomaly"] & ~g["rule_flag"]
    for ch in ("power", "temp", "vib"):
        g.loc[soft & (g[f"{ch}_z"].abs() >= cfg.if_gate_z), f"{ch}_state"] = \
            np.sign(g.loc[soft & (g[f"{ch}_z"].abs() >= cfg.if_gate_z), f"{ch}_z"]).astype(int)
    g["major_power_loss"] = (-g["power_resid"] >= cfg.power_alarm_frac * rated) & ~g["idle"]
    return g


# --------------------------------------------------------------------------- #
# Fault knowledge base (retrieval target for the explainer)
# --------------------------------------------------------------------------- #
# signature: expected state of each channel (-1 low, +1 high, 0 normal, None = don't care)
KB: List[dict] = [
    # ---- solar inverter / plant ----
    dict(id="S1", kind="solar", title="Inverter thermal / cooling-system fault",
         sig=dict(power=None, temp=1, vib=1), urgency="high",
         causes=["Cooling fan failure or fan bearing wear (adds heat and vibration)",
                 "Blocked air filters / heat-sink fouling",
                 "Degraded thermal paste or failing power module / capacitor bank"],
         checks=["Check inverter fault/alarm log for over-temperature or fan-fault codes",
                 "Inspect the cooling fans (spin freely? abnormal noise?) and clean air filters",
                 "Thermal-camera scan of the cabinet and DC/AC terminals for hot spots",
                 "Check mounting bolts and fan imbalance; compare with the neighbouring inverter"]),
    dict(id="S2", kind="solar", title="Inverter trip / derating / grid-side event",
         sig=dict(power=-1, temp=0, vib=0), urgency="high",
         causes=["Inverter tripped on grid over/under-voltage or frequency",
                 "DC isolator / fuse open, or communication loss forcing a curtailment",
                 "Operator or grid-operator curtailment not flagged in the data"],
         checks=["Read the inverter event log around these hours",
                 "Confirm there was no scheduled curtailment / grid outage",
                 "Check AC breaker, DC isolators and string fuses"]),
    dict(id="S3", kind="solar", title="Soiling, shading or string-level underperformance",
         sig=dict(power=-1, temp=0, vib=0), urgency="medium",
         causes=["Dust/bird-dropping soiling or transient shading",
                 "One or more strings down (blown fuse, connector, module failure)",
                 "Irradiance sensor fouled or mis-calibrated (power looks low relative to reading)"],
         checks=["Compare string currents; look for a step change in one string",
                 "Visual inspection / IV-curve test on suspect strings",
                 "Clean and cross-check the pyranometer against a reference cell"]),
    dict(id="S4", kind="solar", title="Over-temperature without mechanical symptoms",
         sig=dict(power=None, temp=1, vib=0), urgency="medium",
         causes=["Poor cabinet ventilation or a blocked inlet",
                 "Temperature sensor drift", "Load/ambient heat not captured by the baseline"],
         checks=["Verify the sensor against a handheld thermometer",
                 "Inspect ventilation path, louvres and filters"]),
    dict(id="S5", kind="solar", title="Abnormal vibration / mechanical looseness",
         sig=dict(power=None, temp=0, vib=1), urgency="medium",
         causes=["Loose mounting or enclosure panels", "Fan imbalance", "Transformer / reactor hum from loose laminations"],
         checks=["Walk-down: listen for rattling, check torque on mounting hardware",
                 "Re-measure vibration with a handheld meter to rule out a sensor fault"]),
    dict(id="S6", kind="solar", title="Output above expected - metering or sensor error",
         sig=dict(power=1, temp=0, vib=0), urgency="low",
         causes=["Irradiance sensor under-reading", "Power meter / CT calibration error"],
         checks=["Cross-check the energy meter against the revenue meter", "Recalibrate the pyranometer"]),
    # ---- wind turbine ----
    dict(id="W1", kind="wind", title="Drivetrain (gearbox / bearing) degradation",
         sig=dict(power=-1, temp=1, vib=1), urgency="high",
         causes=["Gearbox or main-bearing wear, lubrication breakdown",
                 "Generator bearing fault or misalignment",
                 "Cooling/oil-circuit problem raising component temperatures"],
         checks=["Pull condition-monitoring (CMS) vibration spectra: look for bearing / gear-mesh tones",
                 "Check gearbox oil level, oil temperature trend, and send an oil sample",
                 "Review SCADA for over-temperature or derate flags; consider a controlled stop if it persists",
                 "Borescope / visual inspection of the drivetrain at the next access window"]),
    dict(id="W2", kind="wind", title="Rotor imbalance, blade icing or blade damage",
         sig=dict(power=-1, temp=0, vib=1), urgency="high",
         causes=["Ice or dirt build-up on blades", "Pitch-angle mismatch between blades", "Blade crack / leading-edge damage"],
         checks=["Check blade-icing alarms and ambient temperature", "Compare blade pitch angles in SCADA",
                 "Drone or ground-based blade inspection"]),
    dict(id="W3", kind="wind", title="Pitch / yaw misalignment or curtailment",
         sig=dict(power=-1, temp=0, vib=0), urgency="medium",
         causes=["Yaw error (nacelle not facing the wind)", "Pitch controller derate", "Grid or noise curtailment"],
         checks=["Check yaw-misalignment and wind-vane calibration", "Review curtailment commands and control-mode logs"]),
    dict(id="W4", kind="wind", title="Component over-temperature (cooling / lubrication)",
         sig=dict(power=None, temp=1, vib=0), urgency="medium",
         causes=["Cooler fan or pump degraded", "Low oil level or clogged filter", "Temperature sensor drift"],
         checks=["Inspect the cooling system and oil filters", "Verify the sensor against a handheld reading"]),
    dict(id="W5", kind="wind", title="Tower / nacelle looseness or resonance",
         sig=dict(power=None, temp=0, vib=1), urgency="medium",
         causes=["Loose bolts or foundation fatigue", "Operating in a resonance band"],
         checks=["Bolt torque audit", "Review vibration vs. rotor speed for a resonance band"]),
    dict(id="W6", kind="wind", title="Output above power curve - anemometer or metering error",
         sig=dict(power=1, temp=0, vib=0), urgency="low",
         causes=["Anemometer under-reading (iced, worn bearing)", "Meter calibration error"],
         checks=["Cross-check against the met mast / neighbouring turbine", "Inspect or replace the anemometer"]),
]

CH_NAME = {"power": "power output", "temp": "temperature", "vib": "vibration"}


def match_kb(kind: str, states: Dict[str, int]) -> List[dict]:
    """Rank KB entries by agreement between the event's channel signature and each entry's.

    +1 for each channel whose state matches, 0 if the entry doesn't care,
    -1 for a mismatch. A small bonus rewards entries that explain *more* deviating
    channels so the most specific diagnosis wins.
    """
    ranked = []
    for e in KB:
        if e["kind"] != kind:
            continue
        score, maxscore = 0.0, 0.0
        for ch, want in e["sig"].items():
            if want is None:
                continue
            maxscore += 1
            score += 1 if states[ch] == want else -1
        explained = sum(1 for ch, w in e["sig"].items() if w not in (None, 0) and states[ch] == w)
        deviating = sum(1 for s in states.values() if s != 0)
        score += 0.25 * explained
        maxscore += 0.25 * max(deviating, 1)
        match = max(0.0, min(1.0, score / maxscore)) if maxscore else 0.0
        ranked.append(dict(id=e["id"], title=e["title"], match=round(match, 2),
                           urgency=e["urgency"], causes=e["causes"], checks=e["checks"]))
    ranked.sort(key=lambda r: -r["match"])
    return ranked


# --------------------------------------------------------------------------- #
# Events + operator-language explanation
# --------------------------------------------------------------------------- #
def _dir(v: int, up: str, down: str) -> str:
    return up if v > 0 else down


def _severity(ev_rows: pd.DataFrame, cfg: Config) -> str:
    n_ch = sum(int(ev_rows[f"{c}_state"].ne(0).any()) for c in ("power", "temp", "vib"))
    hard = ((ev_rows["vibration_mm_s"] >= cfg.vib_alarm) | (ev_rows["temperature_c"] >= cfg.temp_alarm)
            | ev_rows["major_power_loss"]).any()
    if n_ch >= 2 and hard:
        return "critical"
    if n_ch >= 2 or hard or ev_rows["max_abs_z"].max() >= 2 * cfg.z_thresh:
        return "warning"
    return "watch"


def _explain(asset: str, kind: str, ev: pd.DataFrame, day: int, states: Dict[str, int],
             top: dict, severity: str) -> str:
    start_h, end_h = int(ev["hour"].iloc[0]), int(ev["hour"].iloc[-1])
    dur = len(ev)
    when = f"Day {day}, {start_h:02d}:00" + (f"-{end_h:02d}:00" if dur > 1 else "")
    unit_word = "inverter" if kind == "solar" else "turbine"
    idle = bool(ev["idle"].all())
    parts = []
    if states["temp"]:
        pk = ev.loc[ev["temp_resid"].abs().idxmax()]
        base = float(pk["expected_temp_c"])
        parts.append(f"temperature peaked at {pk['temperature_c']:.1f} degC, about "
                     f"{abs(pk['temperature_c'] - base):.0f} degC {_dir(states['temp'], 'above', 'below')} "
                     f"normal for that hour of day ({base:.0f} degC)")
    if states["vib"]:
        pk = ev.loc[ev["vibration_mm_s"].idxmax()]
        b = float(pk["vib_baseline"])
        parts.append(f"vibration peaked at {pk['vibration_mm_s']:.1f} mm/s, roughly "
                     f"{pk['vibration_mm_s'] / b:.1f}x its usual {b:.1f} mm/s")
    if states["power"]:
        pk = ev.loc[ev["power_resid"].abs().idxmax()]
        short, exp = float(pk["power_resid"]), float(pk["expected_power_mw"])
        parts.append(f"output was {abs(short):.2f} MW {_dir(states['power'], 'above', 'below')} what the "
                     f"{'sunlight' if kind == 'solar' else 'wind speed'} should give ({exp:.2f} MW expected)")
    body = "; ".join(parts) if parts else "several readings drifted together in an unusual combination"

    idle_note = (f" The {unit_word} was idle (no sunlight) at the time, so this is not a production issue - "
                 "the equipment is stressed while doing no work.") if idle and kind == "solar" and (states["temp"] or states["vib"]) else ""
    lead = (f"{asset} - {when} ({dur} h) - {severity.upper()}. "
            f"During this window {body}.{idle_note} ")
    diag = (f"Most likely: {top['title'].lower()} (signature match {int(top['match'] * 100)}%). "
            f"Typical causes: {top['causes'][0].lower()}; {top['causes'][1].lower()}.")
    return lead + diag


def build_events(g: pd.DataFrame, cfg: Config) -> List[dict]:
    asset = g["asset_id"].iloc[0]
    kind = g["kind"].iloc[0]
    idx = np.flatnonzero(g["anomaly"].to_numpy())
    if idx.size == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= cfg.merge_gap_hours:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    events = []
    for n, grp in enumerate(groups, 1):
        ev = g.iloc[grp[0]: grp[-1] + 1]
        flagged = g.iloc[grp]
        # channel state = majority sign among flagged points (0 if never flagged)
        states = {}
        for ch in ("power", "temp", "vib"):
            s = flagged[f"{ch}_state"]
            states[ch] = int(np.sign(s.sum())) if (s != 0).any() else 0
        ranked = match_kb(kind, states)
        top = ranked[0]
        sev = _severity(flagged, cfg)
        # day number: index of the 24h block within this asset's series
        day = int(grp[0] // 24) + 1
        events.append(dict(
            event_id=f"{asset}-E{n}",
            asset_id=asset, kind=kind, severity=sev,
            start_index=int(ev["timestamp_id"].iloc[0]), end_index=int(ev["timestamp_id"].iloc[-1]),
            start_pos=int(grp[0]), end_pos=int(grp[-1]),
            day=day, start_hour=int(ev["hour"].iloc[0]), end_hour=int(ev["hour"].iloc[-1]),
            duration_h=len(ev),
            channels={CH_NAME[c]: ("high" if v > 0 else "low") for c, v in states.items() if v},
            peak_abs_z=round(float(flagged["max_abs_z"].max()), 1),
            detectors=dict(rules=bool(flagged["rule_flag"].any()), isolation_forest=bool(flagged["if_flag"].any())),
            idle=bool(flagged["idle"].all()),
            hypotheses=ranked[:3],
            explanation=_explain(asset, kind, flagged, day, states, top, sev),
            recommended_checks=top["checks"],
            urgency=top["urgency"],
        ))
    return events


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    frame: pd.DataFrame
    events: List[dict] = field(default_factory=list)
    summary: Dict[str, dict] = field(default_factory=dict)


def run_pipeline(df: pd.DataFrame, cfg: Optional[Config] = None) -> Result:
    cfg = cfg or Config()
    df = validate(df)
    frames, events, summary = [], [], {}
    for asset, g in df.groupby("asset_id", sort=True):
        if len(g) < 24:
            raise ValueError(f"{asset}: need at least 24 hourly rows, got {len(g)}")
        f = detect(featurize(g, cfg), cfg)
        frames.append(f)
        ev = build_events(f, cfg)
        events.extend(ev)
        summary[asset] = dict(
            kind=f["kind"].iloc[0], rows=int(len(f)), rated_mw=round(float(f["rated_mw"].iloc[0]), 2),
            flagged_points=int(f["anomaly"].sum()), events=len(ev),
            worst=max((e["severity"] for e in ev), key=["watch", "warning", "critical"].index, default="healthy"),
            mean_power_mw=round(float(f["power_mw"].mean()), 3),
        )
    frame = pd.concat(frames, ignore_index=True)
    sev_rank = {"critical": 0, "warning": 1, "watch": 2}
    events.sort(key=lambda e: (sev_rank[e["severity"]], -e["peak_abs_z"]))
    return Result(frame=frame, events=events, summary=summary)


SERIES_COLS = ["timestamp_id", "hour", "irradiance_wm2", "wind_speed_ms", "temperature_c", "vibration_mm_s",
               "power_mw", "expected_power_mw", "expected_temp_c", "power_z", "temp_z", "vib_z",
               "if_score", "rule_flag", "if_flag", "anomaly"]


def series_for(res: Result, asset_id: str) -> List[dict]:
    f = res.frame[res.frame["asset_id"] == asset_id]
    out = f[SERIES_COLS].copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(bool)
    return out.round(3).to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Optional LLM polish (off unless ANTHROPIC_API_KEY is set; never required)
# --------------------------------------------------------------------------- #
def llm_polish(event: dict) -> Optional[str]:
    """Rewrite the deterministic explanation as a short shift-handover note.

    The deterministic text is always returned by the API; this is only an
    optional enhancement and every failure falls back silently to None.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import json
    import urllib.request
    payload = {
        "model": os.getenv("LLM_MODEL", "claude-haiku-5-5"), "max_tokens": 300,
        "messages": [{"role": "user", "content":
                      "Rewrite this renewable-asset anomaly finding as a 3-sentence shift-handover note for a "
                      "control-room operator. Use ONLY facts in the text; do not invent numbers or causes.\n\n"
                      + event["explanation"] + "\nNext checks: " + "; ".join(event["recommended_checks"])}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception:
        return None
