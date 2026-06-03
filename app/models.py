"""Pydantic V2 event schemas matching the Purplle Store Intelligence spec."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    """Canonical event types emitted by the CV pipeline."""

    ENTRY = "entry"
    EXIT = "exit"
    DWELL = "dwell"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    PURCHASE = "purchase"
    BILLING = "billing"
    GROUP_ENTRY = "group_entry"
    REENTRY = "reentry"


class EventSchema(BaseModel):
    """Single store-intelligence event as defined by the Purplle spec."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "event_id": "evt_001",
                "store_id": "store_mumbai_01",
                "camera_id": "cam_entrance_a",
                "visitor_id": "vis_abc123",
                "event_type": "entry",
                "timestamp": "2026-05-22T10:00:00Z",
                "zone_id": "entrance",
                "dwell_ms": 0,
                "is_staff": False,
                "confidence": 0.92,
                "metadata": {"source": "agent_1"},
            }
        },
    )

    event_id: str = Field(..., min_length=1, description="Globally unique event identifier")
    store_id: str = Field(..., min_length=1, description="Physical store identifier")
    camera_id: str = Field(..., min_length=1, description="Camera that captured the event")
    visitor_id: str = Field(..., min_length=1, description="Anonymous visitor tracking ID")
    event_type: EventType | str = Field(..., description="Event classification")
    timestamp: datetime = Field(..., description="UTC event timestamp")
    zone_id: str | None = Field(default=None, description="Store zone identifier")
    dwell_ms: int | None = Field(default=None, ge=0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(default=False, description="Whether visitor is store staff")
    confidence: float = Field(..., description="Detection confidence score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata bag")

    @field_validator("event_type", mode="before")
    @classmethod
    def normalize_event_type(cls, value: str | EventType) -> str:
        if isinstance(value, EventType):
            return value.value
        return str(value).lower().strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class IngestBatchRequest(BaseModel):
    """Batch ingestion payload (up to 500 events)."""

    events: list[EventSchema] = Field(..., min_length=1, max_length=500)


class IngestResultItem(BaseModel):
    """Per-event ingestion outcome."""

    event_id: str
    status: int
    detail: str | None = None


class IngestBatchResponse(BaseModel):
    """207 Multi-Status response body."""

    accepted: int
    rejected: int
    duplicates: int
    results: list[IngestResultItem]


class POSTransaction(BaseModel):
    """Anonymous POS billing record for conversion attribution."""

    transaction_id: str
    store_id: str
    timestamp: datetime
    amount: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """North Star Metric and supporting KPIs with statistical calibration."""

    store_id: str
    conversion_rate: float
    unique_customer_sessions: int
    unique_customer_purchases: int
    total_events: int
    staff_events_scrubbed: int
    # --- PART B MANDATE ---
    data_confidence: bool = Field(
        ..., 
        description="False if unique_customer_sessions < 20, enforcing calibration"
    )
    # ----------------------
    window_start: datetime | None = None
    window_end: datetime | None = None


class FunnelStage(BaseModel):
    """Single funnel stage count."""

    stage: str
    unique_visitors: int
    drop_off_rate: float | None = None


class FunnelResponse(BaseModel):
    """Store funnel breakdown."""

    store_id: str
    stages: list[FunnelStage]
    overall_conversion_rate: float


class HeatmapCell(BaseModel):
    """Aggregated dwell/intensity for a zone."""

    zone_id: str
    visitor_count: int
    total_dwell_ms: int
    intensity: float


class HeatmapResponse(BaseModel):
    """Zone-level heatmap data."""

    store_id: str
    cells: list[HeatmapCell]


class AnomalyAlert(BaseModel):
    """Active anomaly alert."""

    alert_type: str
    store_id: str
    zone_id: str | None = None
    z_score: float
    current_value: float
    baseline_mean: float
    baseline_std: float
    triggered_at: datetime
    message: str


class AnomaliesResponse(BaseModel):
    """Active anomalies for a store."""

    store_id: str
    alerts: list[AnomalyAlert]


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    database: str
    last_event_at: datetime | None
    feed_lag_seconds: float | None
    alerts: list[str]
