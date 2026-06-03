"""Agent 1: Spatial detection and confidence scoring for store CV pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.config import get_settings
from app.models import EventSchema, EventType
from pipeline.pos_dataset import (
    POS_ANCHOR_LABELS,
    PRE_BILLING_WINDOW_SECONDS,
    PosAnchorTransaction,
    SCORING_STORE_ID,
    load_pos_anchor_transactions,
    pre_billing_event_schedule,
)


@dataclass
class DetectionVector:
    """Spatial tracking vector from Agent 1."""

    visitor_id: str
    store_id: str
    camera_id: str
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    confidence: float
    is_staff_hint: bool = False
    is_group: bool = False
    zone_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HomographyMatrix:
    """Ground-plane homography mapping pixel coordinates to store floor plan."""

    matrix: np.ndarray

    @classmethod
    def from_calibration(
        cls,
        src_points: list[tuple[float, float]],
        dst_points: list[tuple[float, float]],
    ) -> HomographyMatrix:
        """Build homography from at least 4 calibration point pairs."""
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("Homography requires at least 4 point pairs")

        src = np.array(src_points[:4], dtype=np.float64)
        dst = np.array(dst_points[:4], dtype=np.float64)

        n = 4
        a_matrix = np.zeros((2 * n, 8), dtype=np.float64)
        b_vector = np.zeros((2 * n, 1), dtype=np.float64)

        for i in range(n):
            x, y = src[i]
            u, v = dst[i]
            a_matrix[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y]
            b_vector[2 * i, 0] = -u
            a_matrix[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y]
            b_vector[2 * i + 1, 0] = -v

        h8, _, _, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        homography = np.eye(3, dtype=np.float64)
        homography.flat[:8] = h8.flat
        return cls(matrix=homography)

    @classmethod
    def identity(cls) -> HomographyMatrix:
        return cls(matrix=np.eye(3, dtype=np.float64))

    def transform(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Project pixel coordinates onto the ground plane."""
        point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
        projected = self.matrix @ point
        if abs(projected[2]) < 1e-9:
            return pixel_x, pixel_y
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])


DEFAULT_HOMOGRAPHY = HomographyMatrix.from_calibration(
    src_points=[(0, 0), (640, 0), (640, 480), (0, 480)],
    dst_points=[(0, 0), (10, 0), (10, 8), (0, 8)],
)


class SpatialDetector:
    """
    Agent 1: Processes raw detections into spatial tracking vectors with
    confidence scores. Low-confidence detections are flagged for Agent 2 review.
    """

    def __init__(
        self,
        homography: HomographyMatrix | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.homography = homography or DEFAULT_HOMOGRAPHY
        self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD

    def detect(
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
    ) -> DetectionVector:
        """Run spatial detection and compute adjusted confidence."""
        world_x, world_y = self.homography.transform(pixel_x, pixel_y)

        confidence = self._adjust_confidence(
            raw_confidence=raw_confidence,
            bbox_area=bbox_area,
            uniform_score=uniform_score,
            group_proximity_count=group_proximity_count,
        )

        is_staff_hint = uniform_score > 0.6
        is_group = group_proximity_count >= 3

        return DetectionVector(
            visitor_id=visitor_id,
            store_id=store_id,
            camera_id=camera_id,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            world_x=world_x,
            world_y=world_y,
            confidence=confidence,
            is_staff_hint=is_staff_hint,
            is_group=is_group,
            zone_id=zone_id,
            metadata={
                "agent": "agent_1_spatial",
                "raw_confidence": raw_confidence,
                "uniform_score": uniform_score,
                "group_proximity_count": group_proximity_count,
            },
        )

    def needs_vlm_review(self, detection: DetectionVector) -> bool:
        """Route to Agent 2 when confidence drops below threshold."""
        return detection.confidence < self.confidence_threshold

    def to_event(
        self,
        detection: DetectionVector,
        event_type: EventType | str = EventType.DWELL,
        dwell_ms: int | None = None,
        event_id: str | None = None,
    ) -> EventSchema:
        """Convert a validated detection vector into an EventSchema."""
        ts = detection.timestamp
        eid = event_id or f"evt_{detection.store_id}_{detection.visitor_id}_{int(ts.timestamp() * 1000)}"

        return EventSchema(
            event_id=eid,
            store_id=detection.store_id,
            camera_id=detection.camera_id,
            visitor_id=detection.visitor_id,
            event_type=event_type,
            timestamp=ts,
            zone_id=detection.zone_id,
            dwell_ms=dwell_ms,
            is_staff=detection.is_staff_hint,
            confidence=detection.confidence,
            metadata=detection.metadata,
        )

    @staticmethod
    def _adjust_confidence(
        raw_confidence: float,
        bbox_area: float | None,
        uniform_score: float,
        group_proximity_count: int,
    ) -> float:
        confidence = max(0.0, min(1.0, raw_confidence))

        if bbox_area is not None:
            if bbox_area < 500:
                confidence *= 0.85
            elif bbox_area > 50_000:
                confidence *= 0.90

        if uniform_score > 0.5:
            confidence *= 0.75

        if group_proximity_count >= 3:
            confidence *= 0.80

        return round(max(0.0, min(1.0, confidence)), 4)


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Verified POS dataset alignment (ST1008 scoring anchors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneCalibration:
    """Camera pixel anchor for a store zone used by the simulated frame loop."""

    zone_id: str
    pixel_x: float
    pixel_y: float
    camera_id: str


ZONE_ENTRANCE = ZoneCalibration("ENTRANCE", 320.0, 400.0, "cam_entrance_main")
ZONE_BILLING = ZoneCalibration("BILLING", 520.0, 280.0, "cam_billing_a")
ZONE_SKINCARE = ZoneCalibration("SKINCARE", 400.0, 300.0, "cam_skincare_a")

STORE_ZONE_MAP: dict[str, ZoneCalibration] = {
    "ENTRANCE": ZONE_ENTRANCE,
    "BILLING": ZONE_BILLING,
    "SKINCARE": ZONE_SKINCARE,
}


@dataclass(frozen=True)
class SimulatedFrameBeat:
    """Single synthetic frame observation mapped to a future EventSchema."""

    visitor_id: str
    store_id: str
    zone: ZoneCalibration
    event_type: EventType
    timestamp: datetime
    raw_confidence: float = 0.94
    dwell_ms: int = 0


def visitor_id_for_anchor(anchor: PosAnchorTransaction) -> str:
    """Deterministic visitor_id keyed to a verified POS anchor timestamp."""
    label = anchor.anchor_label.replace(":", "")
    return f"vis_{anchor.store_id}_{label}"


def build_pos_aligned_frame_beats(
    store_id: str = SCORING_STORE_ID,
    anchor_labels: tuple[str, ...] = POS_ANCHOR_LABELS,
) -> list[SimulatedFrameBeat]:
    """
    Build ordered frame beats for verified POS anchors (16:55:36, 19:02:09 IST).

    Each anchor emits ENTRY → BILLING zone_enter → billing checkout inside the
    5-minute pre-billing interval preceding the genuine POS timestamp.
    """
    beats: list[SimulatedFrameBeat] = []
    for anchor in load_pos_anchor_transactions(store_id, anchor_labels):
        visitor_id = visitor_id_for_anchor(anchor)
        schedule = pre_billing_event_schedule(anchor.pos_timestamp)

        beats.extend(
            [
                SimulatedFrameBeat(
                    visitor_id=visitor_id,
                    store_id=store_id,
                    zone=ZONE_ENTRANCE,
                    event_type=EventType.ENTRY,
                    timestamp=schedule["entry"],
                    raw_confidence=0.96,
                ),
                SimulatedFrameBeat(
                    visitor_id=visitor_id,
                    store_id=store_id,
                    zone=ZONE_BILLING,
                    event_type=EventType.ZONE_ENTER,
                    timestamp=schedule["billing_zone"],
                    raw_confidence=0.93,
                    dwell_ms=120_000,
                ),
                SimulatedFrameBeat(
                    visitor_id=visitor_id,
                    store_id=store_id,
                    zone=ZONE_BILLING,
                    event_type=EventType.BILLING,
                    timestamp=schedule["checkout"],
                    raw_confidence=0.95,
                    dwell_ms=60_000,
                ),
            ]
        )
    return beats


def assert_pre_billing_alignment(
    event_timestamp: datetime,
    pos_timestamp: datetime,
    *,
    window_seconds: int = PRE_BILLING_WINDOW_SECONDS,
) -> bool:
    """Return True when event_timestamp lies in [pos_ts - window, pos_ts)."""
    delta = pos_timestamp - event_timestamp
    return timedelta(seconds=0) <= delta <= timedelta(seconds=window_seconds)
