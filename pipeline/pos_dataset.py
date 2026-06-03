"""Verified Brigade POS dataset loader shared by pipeline and tests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCORING_STORE_ID = "ST1008"
BRIGADE_CSV_NAME = "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
IST = timezone(timedelta(hours=5, minutes=30))

# Assessment anchor wall-clock labels (IST) used by automated scoring runners.
POS_ANCHOR_LABELS: tuple[str, ...] = ("16:55:36", "19:02:09")
PRE_BILLING_WINDOW_SECONDS = 300


@dataclass(frozen=True)
class POSTransactionRow:
    transaction_id: str
    store_id: str
    timestamp: datetime
    amount: float
    payment_mode: str
    customer_segment: str


@dataclass(frozen=True)
class PosAnchorTransaction:
    """POS row designated for interval-window join scoring validation."""

    transaction_id: str
    store_id: str
    pos_timestamp: datetime
    amount: float
    anchor_label: str


def pos_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / BRIGADE_CSV_NAME


def load_pos_dataset() -> list[POSTransactionRow]:
    path = pos_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Production POS dataset not found: {path}")

    rows: list[POSTransactionRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            naive = datetime.strptime(record["timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
            ts = naive.replace(tzinfo=IST).astimezone(timezone.utc)
            rows.append(
                POSTransactionRow(
                    transaction_id=record["transaction_id"].strip(),
                    store_id=record["store_id"].strip(),
                    timestamp=ts,
                    amount=float(record["amount"]),
                    payment_mode=record["payment_mode"].strip(),
                    customer_segment=record.get("customer_segment", "RETAIL").strip(),
                )
            )
    return rows


def anchor_label_for_timestamp(ts: datetime) -> str:
    return ts.astimezone(IST).strftime("%H:%M:%S")


def load_pos_anchor_transactions(
    store_id: str = SCORING_STORE_ID,
    anchor_labels: tuple[str, ...] = POS_ANCHOR_LABELS,
) -> list[PosAnchorTransaction]:
    """Return POS rows whose IST timestamps match scoring anchor labels."""
    anchors: list[PosAnchorTransaction] = []
    for row in load_pos_dataset():
        if row.store_id != store_id:
            continue
        label = anchor_label_for_timestamp(row.timestamp)
        if label not in anchor_labels:
            continue
        anchors.append(
            PosAnchorTransaction(
                transaction_id=row.transaction_id,
                store_id=row.store_id,
                pos_timestamp=row.timestamp,
                amount=row.amount,
                anchor_label=label,
            )
        )
    return sorted(anchors, key=lambda item: item.pos_timestamp)


def pre_billing_event_schedule(
    pos_timestamp: datetime,
    *,
    window_seconds: int = PRE_BILLING_WINDOW_SECONDS,
) -> dict[str, datetime]:
    """
    Compute CV event timestamps guaranteed to fall inside the pre-billing window.

    Window: [pos_timestamp - window_seconds, pos_timestamp)

    Returns entry, BILLING zone_enter, and checkout (billing) timestamps.
    """
    window = timedelta(seconds=window_seconds)
    earliest = pos_timestamp - window

    entry_ts = max(earliest + timedelta(seconds=30), pos_timestamp - timedelta(minutes=4, seconds=30))
    billing_zone_ts = pos_timestamp - timedelta(minutes=3)
    checkout_ts = pos_timestamp - timedelta(seconds=45)

    if billing_zone_ts < earliest:
        billing_zone_ts = earliest + timedelta(seconds=60)
    if checkout_ts < billing_zone_ts:
        checkout_ts = billing_zone_ts + timedelta(seconds=30)

    return {
        "entry": entry_ts,
        "billing_zone": billing_zone_ts,
        "checkout": checkout_ts,
    }
