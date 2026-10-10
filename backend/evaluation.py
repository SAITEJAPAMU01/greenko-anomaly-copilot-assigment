"""Validation by fault injection.

The bundled CSV has no ground-truth labels, so accuracy cannot be measured on it
directly. Instead we generate realistic telemetry with the same structure
(same physics fitted from the bundled data, same noise level), inject faults of
KNOWN type / size / duration, and score the detectors against those labels.

Run:  python -m backend.evaluation            (writes evaluation/results.json)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .pipeline import Config, run_pipeline

ROOT = Path(__file__).resolve().parents[1]

FAULT_TYPES = ["thermal_vib", "power_loss", "vibration_only", "overtemp_only", "drivetrain"]


def synth_asset(kind: str, asset_id: str, days: int, rng: np.random.Generator, tid0: int) -> pd.DataFrame:
    n = days * 24
    hour = np.tile(np.arange(24), days)
    # slowly varying weather so days are not identical
    day_idx = np.repeat(np.arange(days), 24)
    ambient = 26 + 8 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.9, n) + rng.normal(0, 1.0, days)[day_idx]
    irr = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None) * np.where((hour >= 6) & (hour <= 18), 1, 0)
    irr = irr * np.clip(rng.normal(1, 0.04, days)[day_idx], 0.8, 1.1)
    if kind == "solar":
        wind = np.clip(rng.normal(6, 1.5, n), 3, 10)
        power = 1.9e-3 * irr + rng.normal(0, 0.008, n) * (irr > 50)
    else:
        base = np.clip(rng.normal(6.2, 1.2, days), 4.5, 8)[day_idx]
        wind = np.clip(base + rng.normal(0, 0.9, n), 3, 10)
        power = np.clip(-0.0121 * wind**3 + 0.2 * wind**2 - 0.3 * wind + 0.2, 0.05, 2.7) + rng.normal(0, 0.02, n)
        power = np.clip(power, 0.05, None)
        irr = np.zeros(n)
    temp = ambient + (6 * power if kind == "solar" else 2 * power) + rng.normal(0, 0.5, n)
    vib = np.clip(rng.normal(1.35, 0.11, n), 1.1, 1.6)
    return pd.DataFrame(dict(timestamp_id=np.arange(n) + tid0, asset_id=asset_id, hour=hour,
                             irradiance_wm2=irr.round(1), wind_speed_ms=wind.round(2),
                             temperature_c=temp.round(2), vibration_mm_s=vib.round(2),
                             power_mw=power.round(3)))


def inject(df: pd.DataFrame, kind: str, rng: np.random.Generator, n_faults: int, severity: float) -> pd.DataFrame:
    """severity in (0,1]: 1 = like the bundled-data faults, 0.4 = subtle. Adds `label` and `fault` columns."""
    df = df.copy()
    df["label"], df["fault"] = 0, ""
    n = len(df)
    starts = []
    attempts = 0
    while len(starts) < n_faults and attempts < 500:
        attempts += 1
        s = int(rng.integers(30, n - 6))  # leave baseline-learning room at the start
        if all(abs(s - o) > 12 for o in starts):
            starts.append(s)
    for s in starts:
        dur = int(rng.integers(2, 5))
        sl = slice(s, s + dur)
        ftype = FAULT_TYPES[int(rng.integers(0, len(FAULT_TYPES)))]
        # on a solar asset, a power-loss fault is only observable when the sun is up
        if kind == "solar" and ftype in ("power_loss", "drivetrain"):
            sun = np.flatnonzero(df["irradiance_wm2"].to_numpy() > 300)
            sun = sun[(sun > 30) & (sun < n - 6)]
            s = int(rng.choice(sun)); sl = slice(s, s + dur)
        if ftype == "thermal_vib":
            df.loc[df.index[sl], "temperature_c"] += 18 * severity
            df.loc[df.index[sl], "vibration_mm_s"] += 2.5 * severity
        elif ftype == "power_loss":
            df.loc[df.index[sl], "power_mw"] *= (1 - 0.6 * severity)
        elif ftype == "vibration_only":
            df.loc[df.index[sl], "vibration_mm_s"] += 2.4 * severity
        elif ftype == "overtemp_only":
            df.loc[df.index[sl], "temperature_c"] += 17 * severity
        elif ftype == "drivetrain":
            df.loc[df.index[sl], "temperature_c"] += 18 * severity
            df.loc[df.index[sl], "vibration_mm_s"] += 2.5 * severity
            df.loc[df.index[sl], "power_mw"] *= (1 - 0.45 * severity)
        df.loc[df.index[sl], "label"] = 1
        df.loc[df.index[sl], "fault"] = ftype
    return df


def _prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3), tp=tp, fp=fp, fn=fn)


def _event_recall(df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    """A fault 'event' counts as detected if ANY of its hours is flagged."""
    hit, total = 0, 0
    lab = df["label"].to_numpy()
    i = 0
    while i < len(lab):
        if lab[i] == 1:
            j = i
            while j + 1 < len(lab) and lab[j + 1] == 1:
                j += 1
            total += 1
            hit += int(pred[i:j + 1].any())
            i = j + 1
        else:
            i += 1
    return dict(events=total, detected=hit)


def benchmark(seeds: int = 8, days: int = 30, severity: float = 1.0, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    agg = {m: dict(tp=0, fp=0, fn=0, ev=0, ev_hit=0, clean_hours=0, clean_fp=0)
           for m in ("rules", "isolation_forest", "ensemble")}
    per_fault: Dict[str, List[int]] = {f: [0, 0] for f in FAULT_TYPES}
    diag_ok, diag_n = 0, 0
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        parts = []
        for kind, aid in (("solar", "SOLAR_INV_X"), ("wind", "WTG_X")):
            base = synth_asset(kind, aid, days, rng, tid0=len(parts) * days * 24)
            parts.append(inject(base, kind, rng, n_faults=4, severity=severity))
        data = pd.concat(parts, ignore_index=True)
        res = run_pipeline(data.drop(columns=["label", "fault"]), cfg)
        fr = res.frame.merge(data[["timestamp_id", "asset_id", "label", "fault"]], on=["timestamp_id", "asset_id"])
        for asset, g in fr.groupby("asset_id"):
            g = g.sort_values("timestamp_id").reset_index(drop=True)
            y = g["label"].to_numpy()
            preds = dict(rules=g["rule_flag"].to_numpy(), isolation_forest=g["if_flag"].to_numpy(),
                         ensemble=g["anomaly"].to_numpy())
            for m, p in preds.items():
                p = p.astype(int)
                a = agg[m]
                a["tp"] += int(((y == 1) & (p == 1)).sum()); a["fp"] += int(((y == 0) & (p == 1)).sum())
                a["fn"] += int(((y == 1) & (p == 0)).sum())
                er = _event_recall(g, p.astype(bool)); a["ev"] += er["events"]; a["ev_hit"] += er["detected"]
                a["clean_hours"] += int((y == 0).sum()); a["clean_fp"] += int(((y == 0) & (p == 1)).sum())
            # per-fault-type event recall for the ensemble
            lab = y; i = 0
            while i < len(lab):
                if lab[i] == 1:
                    j = i
                    while j + 1 < len(lab) and lab[j + 1] == 1:
                        j += 1
                    ft = g["fault"].iloc[i]
                    per_fault[ft][1] += 1
                    per_fault[ft][0] += int(preds["ensemble"][i:j + 1].any())
                    # diagnosis check: did the top hypothesis land on the right family?
                    evs = [e for e in res.events if e["asset_id"] == asset
                           and e["start_pos"] <= j and e["end_pos"] >= i]
                    if evs:
                        diag_n += 1
                        top = evs[0]["hypotheses"][0]["id"]
                        diag_ok += int(top in _EXPECTED_TOP[ft])
                    i = j + 1
                else:
                    i += 1
    out = {}
    for m, a in agg.items():
        prec = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0
        rec = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        out[m] = dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                      event_recall=round(a["ev_hit"] / a["ev"], 3) if a["ev"] else 0,
                      false_alarms_per_1000h=round(1000 * a["clean_fp"] / a["clean_hours"], 2))
    return dict(severity=severity, seeds=seeds, days=days, methods=out,
                ensemble_event_recall_by_fault={k: round(v[0] / v[1], 2) if v[1] else None for k, v in per_fault.items()},
                diagnosis_family_accuracy=round(diag_ok / diag_n, 3) if diag_n else None,
                diagnosis_n=diag_n)


# which KB ids count as a correct top diagnosis per injected fault type
_EXPECTED_TOP = {
    "thermal_vib": {"S1", "W1", "W4", "W5"},      # wind variant: heat+vibration, power unaffected -> W1/W4/W5 acceptable
    "power_loss": {"S2", "S3", "W3"},
    "vibration_only": {"S5", "W5"},
    "overtemp_only": {"S4", "W4"},
    "drivetrain": {"W1", "S1", "W2"},
}


def run_all() -> dict:
    return dict(
        note=("Synthetic fault-injection benchmark: labels are known by construction. Scores are "
              "optimistic vs. real plants (clean physics, Gaussian noise); use them to compare methods "
              "and probe sensitivity, not as production accuracy."),
        by_severity=[benchmark(severity=s) for s in (1.0, 0.6, 0.4)],
    )


if __name__ == "__main__":
    res = run_all()
    out = ROOT / "evaluation" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for b in res["by_severity"]:
        print(f"\n== fault severity {b['severity']} ==")
        for m, v in b["methods"].items():
            print(f"{m:17s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} "
                  f"event-recall={v['event_recall']:.2f} false-alarms/1000h={v['false_alarms_per_1000h']}")
        print("ensemble event recall by fault:", b["ensemble_event_recall_by_fault"])
        print("top-1 diagnosis family accuracy:", b["diagnosis_family_accuracy"], "n =", b["diagnosis_n"])
    print(f"\nwrote {out}")
