"""Production POS dataset loader for Brigade Bangalore integration tests."""

from __future__ import annotations

from pipeline.pos_dataset import (
    BRIGADE_CSV_NAME,
    IST,
    POS_ANCHOR_LABELS,
    POSTransactionRow,
    PosAnchorTransaction,
    SCORING_STORE_ID,
    anchor_label_for_timestamp,
    load_pos_anchor_transactions,
    load_pos_dataset,
    pre_billing_event_schedule,
)

# Backward-compatible alias used by existing tests.
BRIGADE_STORE_ID = SCORING_STORE_ID


def brigade_csv_path():
    from pipeline.pos_dataset import pos_csv_path

    return pos_csv_path()


def load_brigade_pos_dataset() -> list[POSTransactionRow]:
    return load_pos_dataset()


def video_events_for_transaction(
    txn: POSTransactionRow,
    visitor_id: str,
    *,
    pre_billing_minutes: int = 4,
) -> list[dict]:
    """Build mock CV events aligned to a real POS timestamp."""
    from datetime import timedelta

    schedule = pre_billing_event_schedule(txn.timestamp)
    return [
        {
            "event_id": f"vid_{txn.transaction_id}_entry",
            "store_id": txn.store_id,
            "camera_id": "cam_entrance_main",
            "visitor_id": visitor_id,
            "event_type": "entry",
            "timestamp": schedule["entry"],
            "zone_id": "ENTRANCE",
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.94,
            "metadata": {"source": "brigade_production_sync", "txn_id": txn.transaction_id},
        },
        {
            "event_id": f"vid_{txn.transaction_id}_zone",
            "store_id": txn.store_id,
            "camera_id": "cam_billing_a",
            "visitor_id": visitor_id,
            "event_type": "zone_enter",
            "timestamp": schedule["billing_zone"],
            "zone_id": "BILLING",
            "dwell_ms": 180000,
            "is_staff": False,
            "confidence": 0.91,
            "metadata": {"source": "brigade_production_sync", "txn_id": txn.transaction_id},
        },
    ]
