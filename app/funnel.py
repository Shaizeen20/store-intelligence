"""Funnel and heatmap calculation engines."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import EventRecord, get_session
from app.metrics import compute_conversion_metrics
from app.models import FunnelResponse, FunnelStage, HeatmapCell, HeatmapResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["funnel"])

FUNNEL_STAGES = [
    ("entry", ["entry", "group_entry"]),
    ("browse", ["zone_enter", "dwell"]),
    ("consideration", ["zone_enter"]),
    ("billing", ["billing"]),
    ("purchase", ["purchase"]),
]


async def _distinct_visitors(
    session: AsyncSession,
    store_id: str,
    event_types: list[str],
    zone_id: str | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> int:
    filters = [
        EventRecord.store_id == store_id,
        EventRecord.is_staff.is_(False),
        EventRecord.event_type.in_(event_types),
    ]
    if zone_id:
        filters.append(EventRecord.zone_id == zone_id)
    if window_start:
        filters.append(EventRecord.timestamp >= window_start)
    if window_end:
        filters.append(EventRecord.timestamp <= window_end)

    result = await session.execute(
        select(func.count(func.distinct(EventRecord.visitor_id)))
        .select_from(EventRecord)
        .where(*filters)
    )
    return result.scalar() or 0


async def compute_store_funnel(
    session: AsyncSession,
    store_id: str,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> FunnelResponse:
    """Build store conversion funnel with staff events scrubbed."""
    stages: list[FunnelStage] = []
    prev_count: int | None = None

    for stage_name, event_types in FUNNEL_STAGES:
        count = await _distinct_visitors(
            session, store_id, event_types, window_start=window_start, window_end=window_end
        )
        drop_off: float | None = None
        if prev_count is not None and prev_count > 0:
            drop_off = round(1.0 - (count / prev_count), 4)
        stages.append(FunnelStage(stage=stage_name, unique_visitors=count, drop_off_rate=drop_off))
        prev_count = count

    metrics = await compute_conversion_metrics(session, store_id, window_start, window_end)

    return FunnelResponse(
        store_id=store_id,
        stages=stages,
        overall_conversion_rate=metrics.conversion_rate,
    )


@router.get("/funnel", response_model=FunnelResponse)
async def get_funnel(
    store_id: str = Query(...),
    window_start: datetime | None = Query(None),
    window_end: datetime | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> FunnelResponse:
    """
    Store conversion funnel with staff events scrubbed.

    Stages: entry → browse → consideration → billing → purchase
    """
    return await compute_store_funnel(session, store_id, window_start, window_end)


@router.get("/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store_id: str = Query(...),
    window_start: datetime | None = Query(None),
    window_end: datetime | None = Query(None),
    session: AsyncSession = Depends(get_session),
) -> HeatmapResponse:
    """
    Zone-level heatmap aggregating dwell time and visitor counts.

    Staff traffic is excluded from intensity calculations.
    """
    filters = [
        EventRecord.store_id == store_id,
        EventRecord.is_staff.is_(False),
        EventRecord.zone_id.isnot(None),
    ]
    if window_start:
        filters.append(EventRecord.timestamp >= window_start)
    if window_end:
        filters.append(EventRecord.timestamp <= window_end)

    result = await session.execute(
        select(
            EventRecord.zone_id,
            func.count(func.distinct(EventRecord.visitor_id)).label("visitor_count"),
            func.coalesce(func.sum(EventRecord.dwell_ms), 0).label("total_dwell_ms"),
        )
        .where(*filters)
        .group_by(EventRecord.zone_id)
    )
    rows = result.all()

    if not rows:
        return HeatmapResponse(store_id=store_id, cells=[])

    max_dwell = max(row.total_dwell_ms for row in rows) or 1

    cells = [
        HeatmapCell(
            zone_id=row.zone_id,
            visitor_count=row.visitor_count,
            total_dwell_ms=int(row.total_dwell_ms),
            intensity=round(row.total_dwell_ms / max_dwell, 4),
        )
        for row in rows
    ]

    return HeatmapResponse(store_id=store_id, cells=cells)
