"""Application configuration."""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Runtime settings loaded from environment variables."""

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./store_intelligence.db",
    )
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    REENTRY_CACHE_TTL_SECONDS: int = int(os.getenv("REENTRY_CACHE_TTL_SECONDS", "300"))
    BILLING_WINDOW_SECONDS: int = int(os.getenv("BILLING_WINDOW_SECONDS", "300"))
    INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "500"))
    STALE_FEED_THRESHOLD_SECONDS: int = int(
        os.getenv("STALE_FEED_THRESHOLD_SECONDS", "600")
    )
    ANOMALY_ZSCORE_THRESHOLD: float = float(
        os.getenv("ANOMALY_ZSCORE_THRESHOLD", "2.5")
    )
    ANOMALY_BASELINE_WINDOW: int = int(os.getenv("ANOMALY_BASELINE_WINDOW", "60"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    VLM_SEMANTIC_CACHE_TTL_SECONDS: int = int(
        os.getenv("VLM_SEMANTIC_CACHE_TTL_SECONDS", "30")
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
