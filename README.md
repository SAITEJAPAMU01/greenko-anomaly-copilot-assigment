# Renewable Asset Fault & Anomaly Detection Copilot

Flags abnormal behaviour in solar-inverter and wind-turbine telemetry, explains it in operator language, ranks likely causes, and recommends next checks. Python/FastAPI backend, single-file HTML dashboard.
LIVE URL :web-production-c8269.up.railway.app

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload          # open http://localhost:8000
python -m backend.evaluation              # regenerate evaluation/results.json (~25 s)
pip install pytest && pytest -q tests
```

## Deploy to Railway
1. Push this folder to a GitHub repo (or use `railway up` with the CLI).
2. Railway → New Project → Deploy from repo. `railway.json` / `Procfile` supply the start command and the `/api/health` check; `.python-version` pins Python 3.12.
3. Settings → Networking → Generate Domain. No secrets are required.
4. Optional: set `ANTHROPIC_API_KEY` (and `LLM_MODEL`) to enable `GET /api/events/{id}?polish=true`, an LLM rewrite of the explanation as a handover note. Off by default; the deterministic explanation is always returned.

Deployment link: _add after deploying_

## How it works
| Stage | What | Why |
|---|---|---|
| Expected-behaviour model | Solar: power = k·irradiance (median ratio, robust). Wind: monotone (isotonic) power curve vs wind speed, refit after dropping the worst 5 % residuals. Temperature: per-asset median for the same hour of day. | "Normal" depends on operating point (0 MW is fine at night, a fault at noon). Robust fits stop faults in the data from corrupting the baseline. |
| Features | Robust z-scores (median/MAD with noise floors) of power, temperature and vibration residuals; 3-h rolling means; temperature rate of change. | Scale-free, comparable across assets, with persistence context. |
| Detector 1 – rules | `\|z\| ≥ 3.5` per channel, plus absolute limits (vibration ≥ 2.5 mm/s, temperature ≥ 60 °C). | Transparent; an operator can see why it fired. |
| Detector 2 – IsolationForest | On the 7 residual features, per asset; top 3 % scores. | Catches unusual combinations no single rule sees. |
| Ensemble | rules OR (IsolationForest AND some channel \|z\| ≥ 2.5). | The gate stops IF flagging benign noise. |
| Events | Flags ≤ 1 h apart are merged; severity = critical / warning / watch from channel count + hard limits. | Operators think in incidents, not hours. |
| Explainer | The event's signature (power / temperature / vibration each low / normal / high) is matched against a 12-entry fault knowledge base (solar and wind); output is ranked causes, next checks, and a templated sentence using the actual numbers. | Every number in the text comes from the data; nothing is generated freely. |
| Copilot | `/api/ask` routes the question (status / why / what to check) and answers only from detected events. | No hallucinated causes or values. |

## Results on the bundled data
Two events, no other hour flagged:
- **SOLAR_INV_01, day 3, 19:00–21:00 – critical.** Temperature up to 53.5 °C (~19 °C above normal), vibration 4.0 mm/s (2.9× normal) while the inverter was idle → inverter cooling/thermal fault.
- **WTG_07, day 3, 05:00–06:00 – critical.** Temperature 46.9 °C (+19 °C), vibration 3.8 mm/s, power 0.59 MW below the power curve → drivetrain (gearbox/bearing) degradation.

## Evaluation
The CSV has no labels, so accuracy there cannot be measured. `backend/evaluation.py` instead generates 2 assets × 30 days × 8 seeds with the same physics and noise, injects 4 faults per asset (heat+vibration, power loss, vibration only, over-temperature only, drivetrain; 2–4 h) at three severities, and scores the detectors against the known labels (`evaluation/results.json`, also shown in the UI).

| Severity (1.0 = bundled-data size) | Method | Precision | Recall | F1 | False alarms /1000 h |
|---|---|---|---|---|---|
| 1.0 | rules | 0.92 | 0.99 | 0.96 | 1.6 |
| 1.0 | IsolationForest | 0.60 | 1.00 | 0.75 | 12.6 |
| 1.0 | ensemble | 0.85 | 0.99 | 0.92 | 3.2 |
| 0.4 | rules | 0.92 | 0.93 | 0.93 | 1.6 |
| 0.4 | IsolationForest | 0.56 | 0.95 | 0.71 | 13.5 |
| 0.4 | ensemble | 0.80 | 0.95 | 0.87 | 4.3 |

Event-level recall (≥ 1 hour of a fault flagged) was 100 % for rules and ensemble at every severity.

**Honest reading:** on injected faults the plain rules beat the ensemble on precision and F1; IsolationForest alone is noisy. The ensemble is kept because IF is there for multivariate patterns that thresholds miss, which this synthetic benchmark barely contains. With real labelled data, choose by measured results (or drop IF / raise `if_gate_z`). The "top-1 diagnosis" score in the JSON (1.0) is circular - injected fault types and the knowledge base come from the same author - so it is not real diagnostic accuracy.

## Assumptions
- `timestamp_id` is an hourly index per asset; "Day N" = position ÷ 24.
- Asset type is inferred from the id (`WTG*`/`WIND*` → wind, else solar).
- Thresholds (vibration 2.5 / 3.5 mm/s, 60 °C) are placeholders; real limits depend on machine class (e.g. ISO 10816/20816) and OEM manuals.
- The 5-day window gives ≥ 3 samples per hour-of-day for the baseline; shorter data falls back to a rolling median.

## Limitations / failure cases
- Only two real events and no labels: no real-world precision/recall claim is possible.
- Slow degradation (drift over weeks) can be absorbed into the baseline; a long-window or seasonal baseline is needed.
- The hour-of-day temperature baseline ignores weather; a hot day can look anomalous. No ambient-temperature channel exists in the data.
- Solar model ignores temperature derating, clipping and soiling trends; wind model ignores air density, curtailment and wakes.
- The fault knowledge base is generic engineering knowledge written for this exercise, not OEM-validated; it ranks signatures, it does not prove root cause.
- Overlapping faults are merged into one event.

## Production-readiness gaps
Streaming ingestion (SCADA / OPC-UA / historian) instead of CSV; persisted event store with acknowledge/close workflow; per-asset thresholds and curtailment/maintenance flags to suppress known states; labelled outcomes from work orders to tune and track alert precision; retrieval over real O&M manuals (RAG) with citations; authentication, rate limiting, monitoring and drift checks; baseline/model versioning.

## API
`GET /api/health`, `/api/overview`, `/api/assets/{id}`, `/api/events[?severity=&asset_id=]`, `/api/events/{id}[?polish=true]`, `POST /api/ask {"text":…}`, `POST /api/analyze` (CSV upload), `GET /api/evaluation`. Interactive docs at `/docs`.

## Layout
`backend/pipeline.py` (features, detectors, KB, explainer) · `backend/app.py` (API) · `backend/evaluation.py` · `frontend/index.html` · `tests/` · `data/asset_telemetry.csv` · `evaluation/results.json`
