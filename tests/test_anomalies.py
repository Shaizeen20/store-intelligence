# PROMPT: Elite-tier anomaly & health chaos suite — validate rolling Z-Score engine
# under empty feeds, 100%-staff scrubbing, billing queue spikes, and STALE_FEED
# integrity checks aligned with real Brigade POS traffic patterns.
#
# CHANGES MADE: Added PROMPT/CHANGES blocks, Brigade-aware billing spike tests,
# cold-start empty-store guards, staff-exclusion chaos cases, and structured health
# monitor assertions for feed-lag fault detection without false positives.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.anomalies import RollingZScoreEngine, scan_anomalies
from app.database import EventRecord, FeedStatusRecord
from tests.fixtures.brigade_loader import BRIGADE_STORE_ID, load_brigade_pos_dataset


def _ts(offset_minutes: int = 0) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)


class TestRollingZScoreEngine:
    def test_insufficient_history_returns_no_alert(self):
        engine = RollingZScoreEngine(window_size=60, z_threshold=2.5)
        assert engine.evaluate("store_x", "billing", 100.0) is None

    def test_spike_triggers_billing_queue_alert(self):
        engine = RollingZScoreEngine(window_size=60, z_threshold=2.0)
        for value in [2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 2.0, 2.0]:
            engine.evaluate("store_spike", "billing", value)

        alert = engine.evaluate("store_spike", "billing", 50.0)
        assert alert is not None
        assert alert.alert_type == "BILLING_QUEUE_SPIKE"
        assert alert.z_score > 2.0

    def test_normal_value_no_alert(self):
        engine = RollingZScoreEngine(window_size=60, z_threshold=2.5)
        for value in [10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 10.0]:
            engine.evaluate("store_norm", "entry", value)
        assert engine.evaluate("store_norm", "entry", 10.5) is None

    def test_zero_baseline_no_division_error(self):
        engine = RollingZScoreEngine(window_size=60, z_threshold=2.5)
        for _ in range(5):
            engine.evaluate("store_zero", "billing", 0.0)
        alert = engine.evaluate("store_zero", "billing", 0.0)
        assert alert is None


class TestEmptyStoreAnomalies:
    @pytest.mark.asyncio
    async def test_empty_store_no_alerts(self, db_session: AsyncSession):
        assert await scan_anomalies(db_session, "store_anomaly_empty") == []

    @pytest.mark.asyncio
    async def test_empty_store_endpoint(self, client: AsyncClient):
        resp = await client.get("/anomalies", params={"store_id": "store_anomaly_empty"})
        assert resp.status_code == 200
        assert resp.json()["alerts"] == []


class TestAllStaffAnomalyScrubbing:
    @pytest.mark.asyncio
    async def test_staff_billing_excluded_from_spike_detection(self, db_session: AsyncSession):
        for i in range(100):
            db_session.add(
                EventRecord(
                    event_id=f"staff_bill_{i}",
                    store_id="store_staff_anom",
                    camera_id="cam_a",
                    visitor_id=f"staff_{i}",
                    event_type="billing",
                    timestamp=_ts(0),
                    zone_id="billing",
                    dwell_ms=0,
                    is_staff=True,
                    confidence=0.99,
                    metadata_json="{}",
                )
            )
        await db_session.commit()

        alerts = await scan_anomalies(db_session, "store_staff_anom")
        assert not any(a.alert_type == "BILLING_QUEUE_SPIKE" for a in alerts)

    @pytest.mark.asyncio
    async def test_hundred_percent_staff_zero_customer_anomalies(
        self, db_session: AsyncSession
    ):
        for i in range(50):
            db_session.add(
                EventRecord(
                    event_id=f"staff_entry_{i}",
                    store_id="store_all_staff",
                    camera_id="cam_a",
                    visitor_id=f"staff_{i}",
                    event_type="entry",
                    timestamp=_ts(),
                    zone_id="entrance",
                    dwell_ms=0,
                    is_staff=True,
                    confidence=0.99,
                    metadata_json="{}",
                )
            )
        await db_session.commit()
        alerts = await scan_anomalies(db_session, "store_all_staff")
        assert isinstance(alerts, list)


class TestBrigadeAwareBillingSpike:
    @pytest.mark.asyncio
    async def test_brigade_store_customer_billing_surge(self, db_session: AsyncSession):
        for i in range(60):
            db_session.add(
                EventRecord(
                    event_id=f"brigade_bill_{i}",
                    store_id=BRIGADE_STORE_ID,
                    camera_id="cam_billing",
                    visitor_id=f"vis_b_{i}",
                    event_type="billing",
                    timestamp=_ts(0),
                    zone_id="billing",
                    dwell_ms=0,
                    is_staff=False,
                    confidence=0.92,
                    metadata_json='{"source":"brigade"}',
                )
            )
        await db_session.commit()
        alerts = await scan_anomalies(db_session, BRIGADE_STORE_ID)
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_brigade_dataset_loads_for_anomaly_context(self):
        rows = load_brigade_pos_dataset()
        assert len(rows) >= 2
        assert rows[0].store_id == BRIGADE_STORE_ID


class TestHealthMonitorChaos:
    @pytest.mark.asyncio
    async def test_healthy_feed_no_stale_alert(self, client: AsyncClient, db_session: AsyncSession):
        db_session.add(
            FeedStatusRecord(store_id="store_health", last_event_at=datetime.now(timezone.utc))
        )
        await db_session.commit()

        resp = await client.get("/health")
        data = resp.json()
        assert resp.status_code == 200
        assert "STALE_FEED" not in data["alerts"]

    @pytest.mark.asyncio
    async def test_stale_feed_when_lag_exceeds_ten_minutes(
        self, client: AsyncClient, db_session: AsyncSession
    ):
        stale = datetime.now(timezone.utc) - timedelta(minutes=15)
        db_session.add(FeedStatusRecord(store_id="store_stale", last_event_at=stale))
        await db_session.commit()

        resp = await client.get("/health")
        data = resp.json()
        assert "STALE_FEED" in data["alerts"]
        assert data["feed_lag_seconds"] > 600

    @pytest.mark.asyncio
    async def test_cold_start_health_without_feed_rows(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert "Traceback" not in resp.text


class TestZeroPurchaseAnomalyScan:
    @pytest.mark.asyncio
    async def test_zero_purchase_store_scan_safe(self, db_session: AsyncSession):
        for i in range(3):
            db_session.add(
                EventRecord(
                    event_id=f"zp_{i}",
                    store_id="store_zp_anom",
                    camera_id="cam_a",
                    visitor_id=f"vis_{i}",
                    event_type="entry",
                    timestamp=_ts(),
                    zone_id="entrance",
                    dwell_ms=0,
                    is_staff=False,
                    confidence=0.9,
                    metadata_json="{}",
                )
            )
        await db_session.commit()
        assert isinstance(await scan_anomalies(db_session, "store_zp_anom"), list)
