# Store Intelligence API

Production-grade containerized API for the **Purplle Tech Challenge 2026**.

## North Star Metric

**Offline Store Conversion Rate** = Total Unique Customer Purchases / Total Unique Customer Sessions

Staff events (`is_staff=true`) are scrubbed from all analytics.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest --cov=app --cov=pipeline --cov-report=term-missing
docker build -t store-intelligence .
docker run -p 8000:8000 store-intelligence
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/events/ingest` | Batch ingest up to 500 events (207 Multi-Status) |
| GET | `/metrics` | Conversion rate and KPIs |
| GET | `/funnel` | Store conversion funnel |
| GET | `/heatmap` | Zone dwell heatmap |
| GET | `/anomalies` | Active Z-Score anomaly alerts |
| GET | `/health` | Feed lag monitor (STALE_FEED > 10 min) |

## Architecture

- **Agent 1** (`pipeline/detect.py`): Spatial tracking with ground-plane homography
- **Agent 2** (`pipeline/tracker.py`): VLM Critic (gemini-3.5-flash) for low-confidence routing
- **Re-entry Cache**: 5-minute spatial tracking window
- **POS Attribution**: SQL interval window join (5-minute pre-billing window)
