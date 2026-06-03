"""FastAPI application entrypoint for Store Intelligence API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.anomalies import router as anomalies_router
from app.config import get_settings
from app.database import init_db
from app.funnel import router as funnel_router
from app.health import router as health_router
from app.ingestion import router as ingestion_router
from app.logging_config import setup_logging
from app.metrics import router as metrics_router
from app.stores import router as stores_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting Store Intelligence API")
    await init_db()
    yield
    logger.info("Shutting down Store Intelligence API")


app = FastAPI(
    title="Store Intelligence API",
    description=(
        "Production-grade containerized API for Purplle Tech Challenge 2026. "
        "Optimizes Offline Store Conversion Rate via multi-agent CV pipeline "
        "and real-time analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingestion_router)
app.include_router(stores_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(anomalies_router)
app.include_router(health_router)


@app.exception_handler(OperationalError)
@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global 503 handler for database connectivity drops."""
    logger.error(
        "Database error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable — database connection failed",
            "error_type": type(exc).__name__,
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "north_star": "Offline Store Conversion Rate",
    }
