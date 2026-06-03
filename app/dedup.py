"""Lock-free in-memory deduplication layer for event ingestion."""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone


class LockFreeDedupCache:
    """
    Thread-safe deduplication cache using atomic dict operations (CPython GIL).

    Uses OrderedDict with a bounded size and TTL eviction. setdefault is used
    as an atomic check-and-set for duplicate detection without explicit locks.
    """

    def __init__(self, max_size: int = 100_000, ttl_seconds: int = 86400) -> None:
        self._cache: OrderedDict[str, datetime] = OrderedDict()
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._evict_lock = threading.Lock()

    def _evict_expired(self, now: datetime) -> None:
        expired_keys = [
            key for key, ts in self._cache.items() if now - ts > self._ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)

    def _evict_overflow(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def try_claim(self, event_id: str) -> bool:
        """
        Attempt to claim an event_id for processing.

        Returns True if this is the first time seeing the event_id (not a duplicate).
        Returns False if the event_id was already processed.
        """
        now = datetime.now(timezone.utc)
        existing = self._cache.setdefault(event_id, now)
        if existing is not now:
            return False

        with self._evict_lock:
            self._evict_expired(now)
            self._evict_overflow()
        return True

    def contains(self, event_id: str) -> bool:
        return event_id in self._cache

    def size(self) -> int:
        return len(self._cache)

    def release(self, event_id: str) -> None:
        """Release a claimed event_id so it can be retried after a failed write."""
        self._cache.pop(event_id, None)

    def clear(self) -> None:
        self._cache.clear()


dedup_cache = LockFreeDedupCache()
