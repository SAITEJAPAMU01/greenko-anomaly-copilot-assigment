"""FastAPI service: anomaly detection + operator copilot.

Endpoints
---------
GET  /                          dashboard (static HTML)
GET  /api/health                liveness probe (used by Railway)
GET  /api/demo                  short description of the service
GET  /api/overview              assets, severities, thresholds in use
GET  /api/assets/{asset_id}     time series (actual/expected/z-scores/flags) + events for one asset
GET  /api/events                all events, optional ?severity=&asset_id=
GET  /api/events/{event_id}     one event with hypotheses, checks, evidence rows
POST /api/ask                   {"text": "..."} copilot Q&A grounded in detected events
POST /api/run                   alias of /api/ask (kept for the starter contract)
POST /api/analyze               upload a CSV with the same schema -> stateless analysis
GET  /api/evaluation            fault-injection benchmark results (precomputed, see evaluation/)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import Config, Result, llm_polish, run_pipeline, series_for

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "asset_telemetry.csv"
FRONTEND = BASE / "frontend" / "index.html"
EVAL_FILE = BASE / "evaluation" / "results.json"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Renewable Asset Fault & Anomaly Detection Copilot", version="1.0.0")

CFG = Config()
_state: dict = {}


def _result() -> Result:
    """Bundled data is analysed once, lazily, then cached for the process lifetime."""
    if "res" not in _state:
        _state["res"] = run_pipeline(pd.read_csv(DATA), CFG)
    return _state["res"]


class Query(BaseModel):
    text: str = ""


# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    res = _result()
    return {
        "case_study": "Renewable Asset Fault & Anomaly Detection Copilot",
        "status": "running",
        "assets": list(res.summary),
        "events_detected": len(res.events),
        "try": ["/api/overview", "/api/events", "POST /api/ask {\"text\": \"what should I check first?\"}"],
    }


@app.get("/api/overview")
def overview():
    res = _result()
    return {"assets": res.summary, "events": res.events, "config": CFG.__dict__}


@app.get("/api/assets/{asset_id}")
def asset(asset_id: str):
    res = _result()
    if asset_id not in res.summary:
        raise HTTPException(404, f"Unknown asset '{asset_id}'. Known: {list(res.summary)}")
    return {"asset_id": asset_id, "summary": res.summary[asset_id], "series": series_for(res, asset_id),
            "events": [e for e in res.events if e["asset_id"] == asset_id]}


@app.get("/api/events")
def events(severity: Optional[str] = None, asset_id: Optional[str] = None):
    out = _result().events
    if severity:
        out = [e for e in out if e["severity"] == severity]
    if asset_id:
        out = [e for e in out if e["asset_id"] == asset_id]
    return out


@app.get("/api/events/{event_id}")
def event(event_id: str, polish: bool = False):
    res = _result()
    ev = next((e for e in res.events if e["event_id"] == event_id), None)
    if ev is None:
        raise HTTPException(404, f"Unknown event '{event_id}'")
    rows = res.frame[(res.frame["asset_id"] == ev["asset_id"])
                     & (res.frame["timestamp_id"].between(ev["start_index"] - 2, ev["end_index"] + 2))]
    cols = ["timestamp_id", "hour", "power_mw", "expected_power_mw", "temperature_c", "expected_temp_c",
            "vibration_mm_s", "power_z", "temp_z", "vib_z", "anomaly"]
    out = dict(ev)
    out["evidence"] = rows[cols].round(2).to_dict(orient="records")
    if polish:  # optional LLM rewrite; None when no key configured
        out["llm_note"] = llm_polish(ev)
    return out


# ------------------------------ copilot ------------------------------------ #
def answer(text: str, res: Result) -> dict:
    """Grounded Q&A: route the question, then answer ONLY from detected events.

    Deliberately rule-based: an operator-facing answer should never contain a
    number or cause that is not in the pipeline output.
    """
    q = text.lower().strip()
    evs = res.events
    asset_ids = [a for a in res.summary if a.lower() in q]
    if "solar" in q and not asset_ids:
        asset_ids = [a for a, s in res.summary.items() if s["kind"] == "solar"]
    if any(w in q for w in ("wind", "turbine", "wtg")) and not asset_ids:
        asset_ids = [a for a, s in res.summary.items() if s["kind"] == "wind"]
    scoped = [e for e in evs if not asset_ids or e["asset_id"] in asset_ids]

    if not evs:
        return dict(answer="No anomalies were detected in the loaded telemetry. All assets look healthy.",
                    intent="status", events=[])
    if not scoped:
        return dict(answer="No anomalies detected for the asset(s) you asked about.", intent="status", events=[])

    if any(w in q for w in ("check", "next", "do", "action", "fix", "inspect", "recommend", "first")):
        e = scoped[0]
        steps = "\n".join(f"{i}. {c}" for i, c in enumerate(e["recommended_checks"], 1))
        return dict(intent="next_checks", events=[e["event_id"]],
                    answer=f"Highest priority: {e['event_id']} ({e['severity']}, {e['hypotheses'][0]['title']}).\n"
                           f"Recommended checks, in order:\n{steps}")
    if any(w in q for w in ("why", "cause", "reason", "explain", "likely")):
        e = scoped[0]
        return dict(intent="explain", events=[e["event_id"]], answer=e["explanation"])
    # default: status summary
    lines = [f"{len(scoped)} event(s) detected" + (f" for {', '.join(asset_ids)}" if asset_ids else "") + ":"]
    for e in scoped:
        lines.append(f"- {e['event_id']} [{e['severity'].upper()}] Day {e['day']} "
                     f"{e['start_hour']:02d}:00 ({e['duration_h']} h): {e['hypotheses'][0]['title']}")
    lines.append("Ask 'why' for the explanation or 'what should I check' for next steps.")
    return dict(intent="summary", events=[e["event_id"] for e in scoped], answer="\n".join(lines))


@app.post("/api/ask")
def ask(q: Query):
    if not q.text.strip():
        raise HTTPException(400, "Provide a question in 'text'.")
    return {"input": q.text, **answer(q.text, _result())}


@app.post("/api/run")
def run_case(q: Query):
    return ask(q)


# ------------------------------ upload / eval ------------------------------ #
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 5 MB).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
        res = run_pipeline(df, CFG)
    except Exception as exc:  # validation errors are user-facing
        raise HTTPException(422, f"Could not analyse file: {exc}")
    return {"assets": res.summary, "events": res.events,
            "series": {a: series_for(res, a) for a in res.summary}}


@app.get("/api/evaluation")
def evaluation():
    if not EVAL_FILE.exists():
        raise HTTPException(404, "Run `python -m backend.evaluation` to generate evaluation/results.json")
    return json.loads(EVAL_FILE.read_text())
