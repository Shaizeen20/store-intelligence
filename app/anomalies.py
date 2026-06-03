"""Active anomaly detection engine with rolling Z-Score baselines."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from statistics import mean, pstdev

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import EventRecord, get_session
from app.models import AnomaliesResponse, AnomalyAlert

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/anomalies", tags=["anomalies"])

ALERT_TYPES = {
    "billing": "BILLING_QUEUE_SPIKE",
    "dwell": "DWELL_ANOMALY",
    "entry": "TRAFFIC_SPIKE",
    "zone_enter": "ZONE_CONGESTION",
}


class RollingZScoreEngine:
    """Rolling Z-Score calculator against a moving historical baseline window."""

    def __init__(self, window_size: int = 60, z_threshold: float = 2.5) -> None:
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._history: dict[str, deque[float]] = {}

    def _key(self, store_id: str, metric: str, zone_id: str | None = None) -> str:
        suffix = zone_id or "all"
        return f"{store_id}:{metric}:{suffix}"

    def update(self, store_id: str, metric: str, value: float, zone_id: str | None = None) -> None:
        key = self._key(store_id, metric, zone_id)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.window_size)
        self._history[key].append(value)

    def evaluate(
        self,
        store_id: str,
        metric: str,
        current_value: float,
        zone_id: str | None = None,
    ) -> AnomalyAlert | None:
        key = self._key(store_id, metric, zone_id)
        history = self._history.get(key, deque())

        if len(history) < 3:
            self.update(store_id, metric, current_value, zone_id)
            return None

        baseline = list(history)
        baseline_mean = mean(baseline)
        baseline_std = pstdev(baseline) if len(baseline) > 1 else 0.0

        if baseline_std == 0:
            z_score = 0.0 if current_value == baseline_mean else float("inf")
        else:
            z_score = (current_value - baseline_mean) / baseline_std

        self.update(store_id, metric, current_value, zone_id)

        if abs(z_score) < self.z_threshold:
            return None

        alert_type = ALERT_TYPES.get(metric, f"{metric.upper()}_ANOMALY")
        now = datetime.now(timezone.utc)

        return AnomalyAlert(
            alert_type=alert_type,
            store_id=store_id,
            zone_id=zone_id,
            z_score=round(z_score, 4),
            current_value=current_value,
            baseline_mean=round(baseline_mean, 4),
            baseline_std=round(baseline_std, 4),
            triggered_at=now,
            message=(
                f"{alert_type}: {metric} value {current_value:.2f} deviates "
                f"{z_score:.2f}σ from baseline mean {baseline_mean:.2f}"
            ),
        )


zscore_engine = RollingZScoreEngine()


async def _count_recent_events(
    session: AsyncSession,
    store_id: str,
    event_type: str,
    zone_id: str | None = None,
    minutes: int = 5,
) -> int:
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    filters = [
        EventRecord.store_id == store_id,
        EventRecord.is_staff.is_(False),
        EventRecord.event_type == event_type,
        EventRecord.timestamp >= cutoff,
    ]
    if zone_id:
        filters.append(EventRecord.zone_id == zone_id)

    result = await session.execute(
        select(func.count()).select_from(EventRecord).where(*filters)
    )
    return result.scalar() or 0


async def scan_anomalies(
    session: AsyncSession,
    store_id: str,
) -> list[AnomalyAlert]:
    """Scan all monitored metrics for active anomalies."""
    settings = get_settings()
    engine = RollingZScoreEngine(
        window_size=settings.ANOMALY_BASELINE_WINDOW,
        z_threshold=settings.ANOMALY_ZSCORE_THRESHOLD,
    )

    alerts: list[AnomalyAlert] = []

    for event_type in ("billing", "entry", "dwell"):
        count = await _count_recent_events(session, store_id, event_type)
        alert = engine.evaluate(store_id, event_type, float(count))
        if alert:
            alerts.append(alert)

    zone_result = await session.execute(
        select(EventRecord.zone_id)
        .where(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.is_staff.is_(False),
        )
        .distinct()
    )
    zones = [row[0] for row in zone_result.all() if row[0]]

    for zone in zones:
        count = await _count_recent_events(session, store_id, "zone_enter", zone_id=zone)
        alert = engine.evaluate(store_id, "zone_enter", float(count), zone_id=zone)
        if alert:
            alerts.append(alert)

    return alerts


@router.get("", response_model=AnomaliesResponse)
async def get_anomalies(
    store_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
) -> AnomaliesResponse:
    """Return active anomaly alerts for a store."""
    alerts = await scan_anomalies(session, store_id)
    return AnomaliesResponse(store_id=store_id, alerts=alerts)


async def generate_llm_anomaly_verdict(anomaly_type: str, store_id: str, details: dict) -> str:
    import os
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"⚠️ [LLM Offline] High-frequency {anomaly_type} detected at store {store_id}. Please mount GEMINI_API_KEY to activate autonomous AI diagnostics."
    try:
        # Initialize modern GenAI client link
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an autonomous AI Judge auditing automated video tracking telemetry records for an elite retail store.
        Analyze the following tracking fragmentation profile and write a highly professional, 2-sentence executive operational verdict.
        
        STORE TARGET: {store_id}
        ANOMALY SIGNATURE: {anomaly_type}
        TELEMETRY METADATA: {details}
        
        Your report must declare:
        1. The exact mathematical impact on the Conversion Rate KPI.
        2. Explicit operational directives for ground staff to resolve the physical camera/zone anomaly immediately.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI Evaluation Exception: Unable to parse log context via Gemini. Details: {str(e)}"
