"""Health monitoring with feed lag and STALE_FEED detection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import FeedStatusRecord, get_session
from app.models import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    """
    Service health monitor tracking data feed update lags.

    Fires STALE_FEED alert when the most recent event lag exceeds 10 minutes.
    """
    settings = get_settings()
    alerts: list[str] = []
    db_status = "ok"
    last_event_at: datetime | None = None
    feed_lag_seconds: float | None = None

    try:
        result = await session.execute(
            select(FeedStatusRecord)
            .order_by(FeedStatusRecord.last_event_at.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        if record and record.last_event_at:
            last_event_at = record.last_event_at
            if last_event_at.tzinfo is None:
                last_event_at = last_event_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            feed_lag_seconds = (now - last_event_at).total_seconds()
            if feed_lag_seconds > settings.STALE_FEED_THRESHOLD_SECONDS:
                alerts.append("STALE_FEED")
    except Exception as exc:
        logger.error("Health check database query failed: %s", exc)
        db_status = "error"
        alerts.append("DATABASE_ERROR")

    overall = "healthy" if not alerts else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        last_event_at=last_event_at,
        feed_lag_seconds=feed_lag_seconds,
        alerts=alerts,
    )
