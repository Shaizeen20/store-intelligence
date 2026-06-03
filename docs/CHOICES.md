# Store Intelligence — Engineering Choices

This document records the principal technical decisions behind the Purplle Tech Challenge 2026 Store Intelligence stack, with rationale tied directly to code paths in this repository.

---

## Section 1 — Detection Model Choice

**Decision:** YOLO-class local detection + ByteTrack-style spatial persistence, with asynchronous **Gemini 3.5 Flash** VLM critic routing for sub-threshold frames only.

**Context:** Store floors generate continuous multi-camera video. The perception layer must emit structured events fast enough that billing-queue anomalies and conversion funnels reflect reality within minutes, not batch ETL cycles.

**Why YOLO + ByteTrack (Agent 1):** Object detectors like YOLO provide millisecond-scale bounding boxes suitable for a triage hot path. ByteTrack (or equivalent IoU / Kalman association) maintains `visitor_id` continuity across frames without invoking a remote model. Our `SpatialDetector` in `pipeline/detect.py` implements this role: pixel detections are mapped to floor coordinates via **ground-plane homography**, confidence is penalized for small boxes, uniform hints, and crowd proximity, and vectors are passed downstream as `DetectionVector` records. This keeps GPU/CPU work localized and predictable.

**Why Gemini 3.5 Flash (Agent 2):** Uniform vs. customer ambiguity, group entries, and low-light false positives are semantic problems — poor fits for geometric tracking alone. When confidence drops below **0.70**, `VLMCriticAgent` in `pipeline/tracker.py` sends a compact JSON-oriented prompt to **gemini-3.5-flash** via `google-genai`, requesting `is_staff`, `is_group_entry`, adjusted confidence, and event type. Flash was chosen over heavier models for **lower latency and cost** on sparse critic calls (typically &lt;5% of detections).

**Why heuristic fallback:** Production stores cannot halt tracking during API outages. If `GEMINI_API_KEY` is empty (default in `docker-compose.yml`) or the SDK call fails, `_heuristic_resolve` applies bounded confidence boosts and staff/group hints locally. Throughput is preserved; accuracy degrades gracefully rather than catastrophically.

**Rejected alternative:** Frame-by-frame multimodal API inference for all cameras — rejected due to quota risk, latency multiplication, and tight coupling between video ingress and analytics availability.

---

## Section 2 — Event Schema Design Rationale

**Decision:** Pydantic V2 `EventSchema` with globally unique **`event_id` keys (UUIDv4-generated at emission)**, stable anonymous **`visitor_id` tracking keys**, and explicit **`EventType` ENUM states** including `REENTRY`.

**Why UUIDv4 for `event_id`:** Ingestion accepts up to **500 events per batch** with concurrent POST storms. Primary keys must be globally unique without coordination. The tracker emits `event_id=f"evt_{uuid.uuid4().hex[:16]}"` and ingestion enforces uniqueness at both the **lock-free dedup cache** (`app/dedup.py`) and the SQL unique index on `events.event_id`. UUIDv4 avoids hot-spotting sequential IDs and eliminates cross-store collision during multi-site rollouts.

**Why explicit ENUM states:** Conversion math depends on *what happened*, not free-text labels. `EventType` enumerates `entry`, `group_entry`, `zone_enter`, `dwell`, `billing`, `purchase`, `exit`, and **`reentry`**. Session denominators count distinct `visitor_id` values on **`entry` and `group_entry` only** (`app/metrics.py`) — `reentry` deliberately does not increment sessions. When a customer exits and returns within five minutes, `process_track_event` matches spatial cache coordinates and emits **`REENTRY` locked to the original `visitor_id`**, rather than a fresh `entry` that would inflate unique sessions and depress conversion rate.

**Staff scrubbing:** `is_staff: bool` is first-class schema metadata. Analytics queries filter `is_staff=false` at SQL level so KPIs never commingle employee foot traffic with customer journeys.

**Metadata bag:** Extensible `metadata` captures agent provenance (`agent_1_spatial`, `agent_2_vlm_critic`, `reentry_matched`) for audit trails without breaking strict schema validation on core fields.

**Rejected alternative:** Inferring event semantics solely from `zone_id` transitions — rejected because billing attribution and funnel stages require explicit, testable event labels validated at ingest time.

---

## Section 3 — API Architecture Choice

**Decision:** **FastAPI** async service with **localized in-memory deduplication**, **asyncio commit locks**, HTTP **207 Multi-Status** batch semantics, and a global **503** handler for database faults.

**Why FastAPI:** Native async I/O fits burst ingestion from CV workers; Pydantic V2 validates 500-event payloads in one request; OpenAPI docs support assessment review. Routers are split by concern (`ingestion`, `metrics`, `funnel`, `anomalies`, `health`) for clarity.

**Why in-memory dedup cache:** `LockFreeDedupCache` uses atomic `OrderedDict.setdefault` (CPython-GIL safe) to reject duplicate `event_id` values before SQL touch — critical when ten concurrent identical batches race during stress tests. Duplicates return per-item **409** entries; mixed batches return **207 Multi-Status** with accepted + duplicate counts.

**Why async commit lock (`_ingest_commit_lock`):** Dedup is lock-free, but SQLite commits are not. Concurrent batches that pass dedup could still interleave `session.add()` / `commit()` and trigger `IntegrityError`. The asyncio lock serializes commits while preserving parallel request parsing and dedup checks — blocking data races without serializing the entire HTTP layer.

**Why 207 / 409 / 422 status matrix:** Reviewers and downstream ETL need machine-readable partial success. `_resolve_http_status` maps outcomes: partial accept → **207**, all duplicates → **409**, validation failure → **422**, clean accept → **201**.

**Why global 503 on `OperationalError`:** Metrics and funnel endpoints must never leak stack traces. `app/main.py` catches SQLAlchemy failures and returns structured JSON `{detail, error_type}` — verified in chaos tests with `unittest.mock`.

**Persistence strategy:** Default `DATABASE_URL` points to `./data/store_intelligence.db` (Docker bind mount) with WAL mode for concurrent read/write sanity. Real Brigade POS CSV lives alongside the DB under `data/` for integration tests that prove interval-window joins on genuine timestamps (16:55:36, 19:02:09 IST).

**Rejected alternative:** Synchronous Flask + Redis dedup — viable at scale, but added infrastructure complexity was unnecessary for assessment scope when in-memory dedup plus DB uniqueness constraints already survive concurrent 500×10 ingestion tests.
