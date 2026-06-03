"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_session
from app.dedup import dedup_cache
from app.main import app
from tests.fixtures.brigade_loader import (
    BRIGADE_STORE_ID,
    POSTransactionRow,
    load_brigade_pos_dataset,
)


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def brigade_pos_dataset() -> list[POSTransactionRow]:
    """Production POS fixture — Brigade Bangalore 10 April 2026 CSV."""
    return load_brigade_pos_dataset()


@pytest.fixture(scope="session")
def brigade_pos_anchor_timestamps(
    brigade_pos_dataset: list[POSTransactionRow],
) -> dict[str, datetime]:
    """Anchor POS timestamps referenced by the assessment framework (IST)."""
    from datetime import timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    return {
        row.timestamp.astimezone(ist).strftime("%H:%M:%S"): row.timestamp
        for row in brigade_pos_dataset
    }


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    dedup_cache.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    dedup_cache.clear()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
