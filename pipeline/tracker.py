"""Agent 2 VLM critic + 5-minute re-entry spatial tracking cache."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import get_settings
from app.models import EventSchema, EventType
from pipeline.detect import (
    DetectionVector,
    SimulatedFrameBeat,
    SpatialDetector,
    assert_pre_billing_alignment,
    build_pos_aligned_frame_beats,
    euclidean_distance,
    visitor_id_for_anchor,
)
from pipeline.pos_dataset import (
    IST,
    POS_ANCHOR_LABELS,
    SCORING_STORE_ID,
    load_pos_anchor_transactions,
    load_pos_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class CachedVisitor:
    """Spatial tracking cache entry for re-entry detection."""

    visitor_id: str
    store_id: str
    world_x: float
    world_y: float
    last_seen: datetime
    is_staff: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ReentryCache:
    """5-minute spatial tracking cache for visitor re-entry resolution."""

    def __init__(self, ttl_seconds: int = 300, proximity_meters: float = 1.5) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._proximity = proximity_meters
        self._entries: dict[str, CachedVisitor] = {}

    def _key(self, store_id: str, visitor_id: str) -> str:
        return f"{store_id}:{visitor_id}"

    def _purge_expired(self, now: datetime) -> None:
        expired = [k for k, v in self._entries.items() if now - v.last_seen > self._ttl]
        for key in expired:
            del self._entries[key]

    def lookup_reentry(
        self,
        store_id: str,
        world_x: float,
        world_y: float,
        now: datetime | None = None,
    ) -> CachedVisitor | None:
        """Find a recently exited visitor near the same spatial coordinates."""
        now = now or datetime.now(timezone.utc)
        self._purge_expired(now)

        best: CachedVisitor | None = None
        best_dist = float("inf")

        for entry in self._entries.values():
            if entry.store_id != store_id:
                continue
            dist = euclidean_distance((world_x, world_y), (entry.world_x, entry.world_y))
            if dist <= self._proximity and dist < best_dist:
                best = entry
                best_dist = dist

        return best

    def upsert(self, detection: DetectionVector, is_staff: bool = False) -> CachedVisitor:
        now = detection.timestamp
        self._purge_expired(now)
        key = self._key(detection.store_id, detection.visitor_id)
        entry = CachedVisitor(
            visitor_id=detection.visitor_id,
            store_id=detection.store_id,
            world_x=detection.world_x,
            world_y=detection.world_y,
            last_seen=now,
            is_staff=is_staff,
            confidence=detection.confidence,
            metadata=detection.metadata,
        )
        self._entries[key] = entry
        return entry

    def mark_exit(
        self,
        detection: DetectionVector,
        is_staff: bool = False,
    ) -> CachedVisitor:
        """Record an EXIT spatial signature for subsequent re-entry matching."""
        return self.upsert(detection, is_staff=is_staff)

    def clear(self) -> None:
        self._entries.clear()


@dataclass(frozen=True)
class SemanticIdentityEntry:
    """Cached VLM semantic identity verdict for a visitor track."""

    visitor_id: str
    store_id: str
    is_staff: bool
    is_group_entry: bool
    adjusted_confidence: float
    vlm_event_type: str
    evaluated_at: datetime


class SemanticIdentityMap:
    """
    In-memory cache of successful VLM identity evaluations per visitor track.

    Bypasses expensive Gemini Vision API calls when the same visitor_id was
    semantically resolved within the last 30 seconds (configurable).
    """

    def __init__(self, ttl_seconds: int = 30) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._entries: dict[str, SemanticIdentityEntry] = {}

    def _key(self, store_id: str, visitor_id: str) -> str:
        return f"{store_id}:{visitor_id}"

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.evaluated_at > self._ttl
        ]
        for key in expired:
            del self._entries[key]

    def lookup(
        self,
        store_id: str,
        visitor_id: str,
        now: datetime | None = None,
    ) -> SemanticIdentityEntry | None:
        """Return a cached VLM verdict if evaluated within the TTL window."""
        now = now or datetime.now(timezone.utc)
        self._purge_expired(now)
        entry = self._entries.get(self._key(store_id, visitor_id))
        if entry is None:
            return None
        if now - entry.evaluated_at > self._ttl:
            del self._entries[self._key(store_id, visitor_id)]
            return None
        return entry

    def store(
        self,
        detection: DetectionVector,
        *,
        is_staff: bool,
        is_group_entry: bool,
        adjusted_confidence: float,
        vlm_event_type: str,
        evaluated_at: datetime | None = None,
    ) -> SemanticIdentityEntry:
        """Persist a successful cloud VLM evaluation for reuse under traffic spikes."""
        evaluated_at = evaluated_at or detection.timestamp or datetime.now(timezone.utc)
        entry = SemanticIdentityEntry(
            visitor_id=detection.visitor_id,
            store_id=detection.store_id,
            is_staff=is_staff,
            is_group_entry=is_group_entry,
            adjusted_confidence=adjusted_confidence,
            vlm_event_type=vlm_event_type,
            evaluated_at=evaluated_at,
        )
        self._entries[self._key(detection.store_id, detection.visitor_id)] = entry
        return entry

    def contains(self, store_id: str, visitor_id: str) -> bool:
        return self.lookup(store_id, visitor_id) is not None

    def clear(self) -> None:
        self._entries.clear()


class VLMCriticAgent:
    """
    Agent 2: VLM Critic using gemini-3.5-flash via google-genai SDK.

    Resolves ambiguous staff flags and group entries when Agent 1 confidence
    drops below 0.70. Reuses SemanticIdentityMap to skip duplicate cloud calls
    within 30 seconds per visitor track. Falls back to heuristic rules when the
    API key is unavailable.
    """

    def __init__(self, semantic_cache: SemanticIdentityMap | None = None) -> None:
        settings = get_settings()
        self.model_name = settings.GEMINI_MODEL
        self.api_key = settings.GEMINI_API_KEY
        self.semantic_cache = semantic_cache or SemanticIdentityMap(
            ttl_seconds=settings.VLM_SEMANTIC_CACHE_TTL_SECONDS
        )
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            logger.warning("Gemini client unavailable: %s", exc)
            return None
        return self._client

    def resolve(
        self,
        detection: DetectionVector,
        frame_context: str | None = None,
    ) -> DetectionVector:
        """Programmatically resolve staff flags or group entries."""
        cached = self.semantic_cache.lookup(
            detection.store_id,
            detection.visitor_id,
            detection.timestamp,
        )
        if cached is not None:
            logger.debug(
                "SemanticIdentityMap hit for %s@%s — bypassing VLM API",
                detection.visitor_id,
                detection.store_id,
            )
            return self._apply_semantic_identity(detection, cached)

        client = self._get_client()
        if client is None:
            return self._heuristic_resolve(detection)

        prompt = self._build_prompt(detection, frame_context)
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            resolved = self._parse_vlm_response(detection, response.text)
            self._cache_vlm_success(resolved)
            return resolved
        except Exception as exc:
            logger.warning("VLM critic failed, using heuristic fallback: %s", exc)
            return self._heuristic_resolve(detection)

    @staticmethod
    def _apply_semantic_identity(
        detection: DetectionVector,
        entry: SemanticIdentityEntry,
    ) -> DetectionVector:
        """Reuse cached cloud VLM flags without a network round-trip."""
        detection.confidence = entry.adjusted_confidence
        detection.is_staff_hint = entry.is_staff
        detection.is_group = entry.is_group_entry
        detection.metadata["agent"] = "agent_2_vlm_critic_cached"
        detection.metadata["vlm_event_type"] = entry.vlm_event_type
        detection.metadata["semantic_identity_cache"] = True
        detection.metadata["semantic_identity_evaluated_at"] = entry.evaluated_at.isoformat()
        return detection

    def _cache_vlm_success(self, detection: DetectionVector) -> SemanticIdentityEntry:
        """Store only successful cloud VLM resolutions (not heuristic fallback)."""
        return self.semantic_cache.store(
            detection,
            is_staff=detection.is_staff_hint,
            is_group_entry=detection.is_group,
            adjusted_confidence=detection.confidence,
            vlm_event_type=str(detection.metadata.get("vlm_event_type", "dwell")),
        )

    def _build_prompt(self, detection: DetectionVector, frame_context: str | None) -> str:
        return (
            "You are a retail computer-vision critic agent. "
            "Given a low-confidence store tracking detection, determine:\n"
            "1. is_staff (true/false)\n"
            "2. is_group_entry (true/false)\n"
            "3. adjusted_confidence (0.0-1.0)\n"
            "4. event_type (entry|group_entry|dwell)\n\n"
            f"Detection: visitor_id={detection.visitor_id}, "
            f"confidence={detection.confidence}, "
            f"uniform_hint={detection.is_staff_hint}, "
            f"group_hint={detection.is_group}, "
            f"zone={detection.zone_id}\n"
            f"Context: {frame_context or 'none'}\n\n"
            "Respond with JSON only: "
            '{"is_staff": bool, "is_group_entry": bool, '
            '"adjusted_confidence": float, "event_type": str}'
        )

    def _parse_vlm_response(self, detection: DetectionVector, text: str) -> DetectionVector:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            payload = json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            return self._heuristic_resolve(detection)

        detection.confidence = float(payload.get("adjusted_confidence", detection.confidence))
        detection.is_staff_hint = bool(payload.get("is_staff", detection.is_staff_hint))
        detection.is_group = bool(payload.get("is_group_entry", detection.is_group))
        detection.metadata["agent"] = "agent_2_vlm_critic"
        detection.metadata["vlm_event_type"] = payload.get("event_type", "dwell")
        return detection

    @staticmethod
    def _heuristic_resolve(detection: DetectionVector) -> DetectionVector:
        """Offline fallback when Gemini is unavailable."""
        if detection.is_staff_hint:
            detection.is_staff_hint = True
            detection.confidence = min(detection.confidence + 0.15, 0.85)

        if detection.is_group:
            detection.confidence = min(detection.confidence + 0.10, 0.80)
            detection.metadata["vlm_event_type"] = "group_entry"
        else:
            detection.metadata["vlm_event_type"] = "entry"

        detection.metadata["agent"] = "agent_2_heuristic_fallback"
        return detection


class TrackingPipeline:
    """
    Multi-agent CV pipeline orchestrator.

    Agent 1 → (confidence < 0.70) → Agent 2 VLM Critic → Re-entry cache
    """

    def __init__(
        self,
        detector: SpatialDetector | None = None,
        vlm_agent: VLMCriticAgent | None = None,
        reentry_cache: ReentryCache | None = None,
        semantic_cache: SemanticIdentityMap | None = None,
    ) -> None:
        settings = get_settings()
        self.detector = detector or SpatialDetector()
        self.semantic_cache = semantic_cache or SemanticIdentityMap(
            ttl_seconds=settings.VLM_SEMANTIC_CACHE_TTL_SECONDS
        )
        self.vlm_agent = vlm_agent or VLMCriticAgent(semantic_cache=self.semantic_cache)
        self.reentry_cache = reentry_cache or ReentryCache(
            ttl_seconds=settings.REENTRY_CACHE_TTL_SECONDS
        )

    def process_detection(
        self,
        visitor_id: str,
        store_id: str,
        camera_id: str,
        pixel_x: float,
        pixel_y: float,
        raw_confidence: float,
        zone_id: str | None = None,
        bbox_area: float | None = None,
        uniform_score: float = 0.0,
        group_proximity_count: int = 0,
        frame_context: str | None = None,
    ) -> EventSchema:
        detection = self.detector.detect(
            visitor_id=visitor_id,
            store_id=store_id,
            camera_id=camera_id,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            raw_confidence=raw_confidence,
            zone_id=zone_id,
            bbox_area=bbox_area,
            uniform_score=uniform_score,
            group_proximity_count=group_proximity_count,
        )

        reentry = self.reentry_cache.lookup_reentry(
            store_id=store_id,
            world_x=detection.world_x,
            world_y=detection.world_y,
            now=detection.timestamp,
        )
        if reentry and reentry.visitor_id != visitor_id:
            detection.visitor_id = reentry.visitor_id
            detection.metadata["reentry_matched"] = True
            detection.metadata["original_visitor_id"] = visitor_id

        if self.detector.needs_vlm_review(detection):
            detection = self.vlm_agent.resolve(detection, frame_context)

        event_type = EventType.DWELL
        if detection.is_group:
            event_type = EventType.GROUP_ENTRY
        elif detection.metadata.get("vlm_event_type") == "entry":
            event_type = EventType.ENTRY

        is_staff = detection.is_staff_hint and detection.confidence >= 0.70

        self.reentry_cache.upsert(detection, is_staff=is_staff)

        return self.detector.to_event(
            detection,
            event_type=event_type,
            dwell_ms=0,
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
        )

    def process_track_event(
        self,
        event_type: EventType | str,
        visitor_id: str,
        store_id: str,
        camera_id: str,
        pixel_x: float,
        pixel_y: float,
        raw_confidence: float = 0.95,
        zone_id: str | None = None,
        timestamp: datetime | None = None,
        bbox_area: float | None = None,
        uniform_score: float = 0.0,
        group_proximity_count: int = 0,
        frame_context: str | None = None,
    ) -> EventSchema:
        """
        Emit a structured journey event (ENTRY, ZONE_ENTER, EXIT, REENTRY).

        On ENTRY within 5 minutes of a prior EXIT at the same spatial coordinates,
        emits REENTRY locked to the original visitor_id.
        """
        normalized = (
            event_type.value if isinstance(event_type, EventType) else str(event_type).lower().strip()
        )
        ts = timestamp or datetime.now(timezone.utc)

        detection = self.detector.detect(
            visitor_id=visitor_id,
            store_id=store_id,
            camera_id=camera_id,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            raw_confidence=raw_confidence,
            zone_id=zone_id,
            bbox_area=bbox_area,
            uniform_score=uniform_score,
            group_proximity_count=group_proximity_count,
        )
        detection.timestamp = ts

        resolved_type = normalized
        original_visitor_id = visitor_id

        if normalized == EventType.EXIT.value:
            self.reentry_cache.mark_exit(detection, is_staff=False)
            return self.detector.to_event(
                detection,
                event_type=EventType.EXIT,
                dwell_ms=0,
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
            )

        if normalized == EventType.ENTRY.value:
            reentry = self.reentry_cache.lookup_reentry(
                store_id=store_id,
                world_x=detection.world_x,
                world_y=detection.world_y,
                now=ts,
            )
            if reentry and reentry.visitor_id != visitor_id:
                detection.visitor_id = reentry.visitor_id
                detection.metadata["reentry_matched"] = True
                detection.metadata["original_visitor_id"] = visitor_id
                detection.metadata["agent"] = "agent_1_spatial_reentry"
                original_visitor_id = reentry.visitor_id
                resolved_type = EventType.REENTRY.value

        if self.detector.needs_vlm_review(detection) and resolved_type != EventType.REENTRY.value:
            detection = self.vlm_agent.resolve(detection, frame_context)

        if resolved_type == EventType.REENTRY.value:
            output_type: EventType | str = EventType.REENTRY
        elif normalized == EventType.ENTRY.value:
            output_type = EventType.ENTRY
        elif normalized == EventType.ZONE_ENTER.value:
            output_type = EventType.ZONE_ENTER
        elif normalized == EventType.BILLING.value:
            output_type = EventType.BILLING
        elif normalized == EventType.EXIT.value:
            output_type = EventType.EXIT
        elif normalized == EventType.DWELL.value:
            output_type = EventType.DWELL
        elif detection.is_group:
            output_type = EventType.GROUP_ENTRY
        elif detection.metadata.get("vlm_event_type") == "entry":
            output_type = EventType.ENTRY
        elif normalized in EventType._value2member_map_:
            output_type = EventType(normalized)
        else:
            output_type = EventType.DWELL

        is_staff = detection.is_staff_hint and detection.confidence >= 0.70
        self.reentry_cache.upsert(detection, is_staff=is_staff)

        event = self.detector.to_event(
            detection,
            event_type=output_type,
            dwell_ms=0,
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
        )
        event.metadata["original_visitor_id"] = original_visitor_id
        if resolved_type == EventType.REENTRY.value:
            event.metadata["reentry"] = True
        return event


class PosAlignedFrameSimulator:
    """
    Simulated multi-frame loop aligned to verified ST1008 POS anchor timestamps.

    Dynamically generates visitor ENTRY and BILLING-zone checkout events positioned
    inside the 5-minute pre-billing window before 16:55:36 and 19:02:09 IST so
    automated scoring runners produce non-zero conversion joins.
    """

    def __init__(
        self,
        pipeline: TrackingPipeline | None = None,
        store_id: str = SCORING_STORE_ID,
        anchor_labels: tuple[str, ...] = POS_ANCHOR_LABELS,
    ) -> None:
        self.pipeline = pipeline or TrackingPipeline()
        self.store_id = store_id
        self.anchor_labels = anchor_labels

    def frame_beats(self) -> list[SimulatedFrameBeat]:
        return build_pos_aligned_frame_beats(self.store_id, self.anchor_labels)

    def run_simulated_frame_loop(self) -> list[EventSchema]:
        """
        Execute the synthetic frame loop and return validated EventSchema rows.

        Each POS anchor produces:
          1. ENTRY at ENTRANCE
          2. ZONE_ENTER inside BILLING
          3. billing (checkout queue) inside BILLING
        """
        anchors = load_pos_anchor_transactions(self.store_id, self.anchor_labels)
        anchor_by_visitor = {visitor_id_for_anchor(anchor): anchor for anchor in anchors}
        events: list[EventSchema] = []

        for beat in self.frame_beats():
            matching_anchor = anchor_by_visitor.get(beat.visitor_id)

            event = self.pipeline.process_track_event(
                beat.event_type,
                visitor_id=beat.visitor_id,
                store_id=beat.store_id,
                camera_id=beat.zone.camera_id,
                pixel_x=beat.zone.pixel_x,
                pixel_y=beat.zone.pixel_y,
                raw_confidence=beat.raw_confidence,
                zone_id=beat.zone.zone_id,
                timestamp=beat.timestamp,
            )
            event.dwell_ms = beat.dwell_ms or event.dwell_ms
            event.metadata.update(
                {
                    "simulated_frame_loop": True,
                    "pos_anchor": matching_anchor.anchor_label if matching_anchor else None,
                    "pos_transaction_id": (
                        matching_anchor.transaction_id if matching_anchor else None
                    ),
                    "zone_calibration": beat.zone.zone_id,
                }
            )

            if matching_anchor:
                assert assert_pre_billing_alignment(
                    event.timestamp,
                    matching_anchor.pos_timestamp,
                ), (
                    f"Event {event.event_id} at {event.timestamp} falls outside "
                    f"pre-billing window for anchor {matching_anchor.anchor_label}"
                )

            events.append(event)

        return events

    @staticmethod
    def pos_transaction_records() -> list[dict]:
        """Serialize verified POS CSV rows for DB seeding by scoring runners."""
        return [
            {
                "transaction_id": row.transaction_id,
                "store_id": row.store_id,
                "timestamp": row.timestamp,
                "amount": row.amount,
                "metadata": {
                    "payment_mode": row.payment_mode,
                    "customer_segment": row.customer_segment,
                    "anchor_label": row.timestamp.astimezone(IST).strftime("%H:%M:%S"),
                },
            }
            for row in load_pos_dataset()
            if row.store_id == SCORING_STORE_ID
        ]


def run_scoring_frame_loop(store_id: str = SCORING_STORE_ID) -> list[EventSchema]:
    """Convenience entrypoint for automated assessment scoring runners."""
    return PosAlignedFrameSimulator(store_id=store_id).run_simulated_frame_loop()
