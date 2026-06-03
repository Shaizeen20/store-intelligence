# PROMPT: Harden POST /events/ingest for concurrent 500-event batches with lock-free
# deduplication, IntegrityError safety, serialized commits, and strict HTTP 207
# Multi-Status responses under partial success and duplicate-heavy chaos loads.
#
# CHANGES MADE: Added asyncio commit lock, SQLAlchemy IntegrityError handling,
# duplicate DB-row suppression with dedup release, and refined status-code logic
# for concurrent partial-success (207) vs all-duplicate (409) vs full accept (201).

"""Event ingestion router with batch processing and 207 Multi-Status."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import EventRecord, FeedStatusRecord, get_session
from app.dedup import dedup_cache
from app.models import EventSchema, IngestBatchRequest, IngestBatchResponse, IngestResultItem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["ingestion"])

_ingest_commit_lock = asyncio.Lock()


async def _upsert_feed_status(session: AsyncSession, store_id: str, ts: datetime) -> None:
    from sqlalchemy import select

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    result = await session.execute(
        select(FeedStatusRecord).where(FeedStatusRecord.store_id == store_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        session.add(FeedStatusRecord(store_id=store_id, last_event_at=ts))
    elif record.last_event_at is None or ts > (
        record.last_event_at.replace(tzinfo=timezone.utc)
        if record.last_event_at.tzinfo is None
        else record.last_event_at.astimezone(timezone.utc)
    ):
        record.last_event_at = ts


def _serialize_event(event: EventSchema) -> EventRecord:
    return EventRecord(
        event_id=event.event_id,
        store_id=event.store_id,
        camera_id=event.camera_id,
        visitor_id=event.visitor_id,
        event_type=str(event.event_type),
        timestamp=event.timestamp,
        zone_id=event.zone_id,
        dwell_ms=event.dwell_ms,
        is_staff=event.is_staff,
        confidence=event.confidence,
        metadata_json=json.dumps(event.metadata),
    )


def _resolve_http_status(accepted: int, rejected: int, duplicates: int, total: int) -> int:
    if rejected > 0 and accepted > 0:
        return status.HTTP_207_MULTI_STATUS
    if duplicates > 0 and accepted > 0:
        return status.HTTP_207_MULTI_STATUS
    if rejected > 0 and accepted == 0:
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    if duplicates > 0 and accepted == 0 and duplicates == total:
        return status.HTTP_409_CONFLICT
    if duplicates > 0:
        return status.HTTP_207_MULTI_STATUS
    return status.HTTP_201_CREATED


@router.post(
    "/ingest",
    response_model=IngestBatchResponse,
    status_code=status.HTTP_207_MULTI_STATUS,
    summary="Ingest up to 500 store intelligence events",
)
async def ingest_events(
    payload: IngestBatchRequest,
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    """
    Accept batches of up to 500 events with lock-free deduplication.

    Returns HTTP 207 Multi-Status when partial success occurs (some accepted,
    some rejected/duplicated).
    """
    settings = get_settings()
    if len(payload.events) > settings.INGEST_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch exceeds maximum size of {settings.INGEST_BATCH_SIZE}",
        )

    results: list[IngestResultItem] = []
    accepted = 0
    rejected = 0
    duplicates = 0
    latest_by_store: dict[str, datetime] = {}
    pending_records: list[EventRecord] = []

    for event in payload.events:
        try:
            if not dedup_cache.try_claim(event.event_id):
                duplicates += 1
                results.append(
                    IngestResultItem(
                        event_id=event.event_id,
                        status=409,
                        detail="Duplicate event_id",
                    )
                )
                continue

            record = _serialize_event(event)
            pending_records.append(record)

            store_ts = latest_by_store.get(event.store_id)
            if store_ts is None or event.timestamp > store_ts:
                latest_by_store[event.store_id] = event.timestamp

            accepted += 1
            results.append(
                IngestResultItem(event_id=event.event_id, status=201, detail="Accepted")
            )
        except Exception as exc:
            rejected += 1
            dedup_cache.release(event.event_id)
            logger.exception("Failed to ingest event %s", event.event_id)
            results.append(
                IngestResultItem(
                    event_id=event.event_id,
                    status=422,
                    detail="Event validation failed",
                )
            )

    if accepted > 0:
        try:
            async with _ingest_commit_lock:
                for record in pending_records:
                    session.add(record)
                for store_id, ts in latest_by_store.items():
                    await _upsert_feed_status(session, store_id, ts)
                await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            logger.warning("IntegrityError during batch commit: %s", exc)
            accepted = 0
            duplicates = 0
            rejected = 0
            results = []
            seen: set[str] = set()
            for event in payload.events:
                if event.event_id in seen or dedup_cache.contains(event.event_id):
                    duplicates += 1
                    results.append(
                        IngestResultItem(
                            event_id=event.event_id,
                            status=409,
                            detail="Duplicate event_id",
                        )
                    )
                else:
                    rejected += 1
                    dedup_cache.release(event.event_id)
                    results.append(
                        IngestResultItem(
                            event_id=event.event_id,
                            status=422,
                            detail="Database integrity conflict under concurrent load",
                        )
                    )
                seen.add(event.event_id)
        except (OperationalError, SQLAlchemyError):
            await session.rollback()
            for event in payload.events:
                dedup_cache.release(event.event_id)
            raise
    else:
        await session.rollback()

    response_body = IngestBatchResponse(
        accepted=accepted,
        rejected=rejected,
        duplicates=duplicates,
        results=results,
    )

    http_status = _resolve_http_status(
        accepted, rejected, duplicates, len(payload.events)
    )

    return JSONResponse(
        status_code=http_status,
        content=response_body.model_dump(),
    )
