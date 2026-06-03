"""Store-scoped REST routes for dashboard and Part E bonus consumers."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.funnel import compute_store_funnel
from app.metrics import compute_conversion_metrics
from app.models import FunnelResponse, MetricsResponse

router = APIRouter(prefix="/stores", tags=["stores"])


@router.get("/{store_id}/metrics", response_model=MetricsResponse)
async def get_store_metrics(
    store_id: str,
    session: AsyncSession = Depends(get_session),
) -> MetricsResponse:
    """Return North Star conversion metrics for a specific store."""
    return await compute_conversion_metrics(session, store_id)


@router.get("/{store_id}/funnel", response_model=FunnelResponse)
async def get_store_funnel(
    store_id: str,
    session: AsyncSession = Depends(get_session),
) -> FunnelResponse:
    """Return conversion funnel breakdown for a specific store."""
    return await compute_store_funnel(session, store_id)
