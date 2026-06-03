"""Async SQLAlchemy database layer with interval-window join support."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    event,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    camera_id: Mapped[str] = mapped_column(String(64))
    visitor_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dwell_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_zone", "store_id", "zone_id"),
    )


class POSTransactionRecord(Base):
    __tablename__ = "pos_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    amount: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class FeedStatusRecord(Base):
    __tablename__ = "feed_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


settings = get_settings()
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def interval_window_join_sql(
    session: AsyncSession,
    store_id: str,
    window_seconds: int = 300,
) -> list[dict]:
    """
    SQL-level Interval Window Join: match anonymous POS transactions to the
    nearest preceding customer visitor_id within a pre-billing window.

    For each POS transaction, find the most recent non-staff zone/billing
    event for the same store whose timestamp falls within [txn_ts - window, txn_ts).
    """
    query = text(
        """
        WITH customer_events AS (
            SELECT
                visitor_id,
                store_id,
                timestamp,
                zone_id,
                event_type
            FROM events
            WHERE store_id = :store_id
              AND is_staff = 0
              AND event_type IN ('zone_enter', 'dwell', 'billing', 'entry')
        ),
        attributed AS (
            SELECT
                p.transaction_id,
                p.store_id,
                p.timestamp AS txn_timestamp,
                p.amount,
                ce.visitor_id,
                ce.timestamp AS event_timestamp,
                ce.zone_id,
                ROW_NUMBER() OVER (
                    PARTITION BY p.transaction_id
                    ORDER BY ce.timestamp DESC
                ) AS rn
            FROM pos_transactions p
            LEFT JOIN customer_events ce
                ON ce.store_id = p.store_id
               AND ce.timestamp <= p.timestamp
               AND ce.timestamp >= datetime(p.timestamp, :neg_window)
            WHERE p.store_id = :store_id
        )
        SELECT
            transaction_id,
            store_id,
            txn_timestamp,
            amount,
            visitor_id,
            event_timestamp,
            zone_id
        FROM attributed
        WHERE rn = 1 AND visitor_id IS NOT NULL
        ORDER BY txn_timestamp
        """
    )
    result = await session.execute(
        query,
        {
            "store_id": store_id,
            "neg_window": f"-{window_seconds} seconds",
        },
    )
    rows = result.mappings().all()
    return [dict(row) for row in rows]
