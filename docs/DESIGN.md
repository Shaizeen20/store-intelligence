# Store Intelligence — System Design

## Overview: Decoupled Dual-Engine Multi-Agent Architecture

The Store Intelligence API is built around a **decoupled dual-engine** pattern that separates *real-time perception* from *analytical truth*. Engine One is the **Computer Vision Pipeline** (`pipeline/detect.py`, `pipeline/tracker.py`), which runs at the edge of the store and converts raw camera observations into structured, validated events. Engine Two is the **Analytics API** (`app/metrics.py`, `app/funnel.py`, `app/anomalies.py`), which ingests those events asynchronously, scrubs staff traffic, attributes POS purchases, and computes the North Star Metric: **Offline Store Conversion Rate** (unique customer purchases ÷ unique customer sessions).

These engines communicate only through a stable **Event Schema** (`app/models.py`) and the `POST /events/ingest` contract. The CV pipeline never blocks on database writes; the API never inspects pixels. That boundary keeps each layer independently deployable, horizontally scalable, and testable — the pipeline can be swapped from simulation to live YOLO inference without touching metric math, and the API can replay historical CSV-backed POS data without re-running video.

Within Engine One, **Agent 1 (Spatial Detector)** performs high-throughput triage: bounding-box detections are projected through a ground-plane homography matrix, confidence is adjusted for uniform hints and group proximity, and spatial tracking vectors are emitted. When confidence falls below **0.70**, routing escalates to **Agent 2 (VLM Critic)** — but only for ambiguous frames, not the entire stream. Engine Two applies a complementary triage: staff events are filtered at SQL level, purchases are joined to anonymous visitor IDs via a **5-minute pre-billing interval window**, and rolling Z-Score engines fire alerts like `BILLING_QUEUE_SPIKE` without recomputing video.

Persistent state is deliberately minimal. SQLite (bind-mounted under `data/` in Docker) stores events, POS transactions, and feed-status timestamps. A **5-minute re-entry spatial cache** in the tracker resolves erratic customer paths (ENTRY → ZONE_ENTER → EXIT → ENTRY) into explicit `REENTRY` events locked to the original `visitor_id`, preventing denominator inflation in conversion math. Ingestion uses a **lock-free deduplication cache** plus an **async commit lock** so concurrent 500-event batches cannot corrupt primary keys under spike load.

Container orchestration (`docker-compose.yml`) exposes port 8000, mounts `data/` for CSV and SQLite WAL I/O, and health-checks `/health` every 30 seconds — including `STALE_FEED` detection when the event stream lag exceeds ten minutes.

---

## AI-Assisted Decisions

During design exploration, an LLM-assisted workflow suggested sending **entire video clips frame-by-frame through a remote vision API** — essentially treating Gemini as the primary detector for every bounding box, staff/uniform classification, and group-entry resolution. That approach is architecturally simple but operationally fragile: latency per frame multiplied by dozens of cameras would collapse throughput, API quotas would become the bottleneck, and conversion metrics would lag minutes behind ground truth — unacceptable for anomaly detection on billing queues.

We rejected full-frame API inference as the hot path and instead adopted a **YOLO + ByteTrack-style triage layer** (simulated in `SpatialDetector` with homography-backed world coordinates) that processes detections locally at camera-adjacent speed. Agent 1 owns the 30 FPS class of work: spatial vectors, confidence scoring, and re-entry cache lookups. Only detections that fail the **0.70 confidence gate** are escalated to **Gemini 3.5 Flash** via the `google-genai` SDK in `VLMCriticAgent`.

Critically, we also implemented a **localized heuristic fallback** when `GEMINI_API_KEY` is unset or the remote call fails. Heuristics infer staff flags from uniform scores, promote group entries, and bump confidence within bounded ceilings — preserving pipeline progress without blocking ingestion. This mirrors production best practice: **AI assists judgment on hard cases; rules guarantee liveness.**

Similar reasoning applied to analytics: an LLM suggested computing conversions in application Python loops over raw events. We chose a **SQL interval window join** (`interval_window_join_sql` in `app/database.py`) so POS timestamps from real Brigade Bangalore CSV rows match preceding `zone_enter` / `billing` events inside a five-minute window — deterministic, auditable, and fast under assessment-scale data volumes.

The dual-engine split, confidence-gated VLM routing, and heuristic fallback together optimize for **throughput first, intelligence second** — the architecture survives real load even when cloud AI is unavailable.
