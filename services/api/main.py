"""
services/api/main.py

Main FastAPI application entry point for Valora AI — a professional
AI-powered voice generation platform.

This module wires together the TTS model, inference service, and
preprocessing pipeline (services/tts/*) into a production-ready FastAPI
application, including lifespan-managed startup/shutdown, request
logging, security middleware, global exception handling, and health
diagnostics endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from services.tts.config import settings
from services.tts.inference import TTSInferenceError, get_tts_service
from services.tts.utils import get_gpu_info, get_logger

logger = get_logger("valora.api.main")

APP_TITLE = "Valora AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Professional AI-powered voice generation platform."

ALLOWED_HOSTS = ["*"]
ALLOWED_ORIGINS = ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown lifecycle.

    On startup: loads and warms up the TTS model, verifies CUDA
    availability, and logs GPU diagnostics. On shutdown: releases GPU
    memory and gracefully shuts down the inference service's thread
    pool and underlying model.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded back to the running application.
    """
    logger.info("Starting %s v%s (env=%s)...", APP_TITLE, APP_VERSION, settings.ENV)

    try:
        gpu_info = get_gpu_info()
        if gpu_info.get("cuda_available"):
            logger.info("CUDA is available. GPU info: %s", gpu_info)
        else:
            logger.warning("CUDA is not available; running on CPU.")

        service = get_tts_service()
        service.warm_up()
        app.state.tts_service = service

        logger.info("%s startup complete. Health: %s", APP_TITLE, service.health_check())
    except Exception as exc:
        logger.exception("Fatal error during application startup.")
        raise

    yield

    logger.info("Shutting down %s...", APP_TITLE)
    try:
        service = getattr(app.state, "tts_service", None)
        if service is not None:
            service.shutdown()
    except Exception:
        logger.exception("Error occurred while shutting down TTS inference service.")
    finally:
        logger.info("%s shutdown complete.", APP_TITLE)


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)


# ==================================================================== #
# Middleware
# ==================================================================== #

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def request_logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Log request/response metadata and attach security headers and a
    unique request ID to every response.

    Args:
        request: The incoming HTTP request.
        call_next: The next handler in the middleware chain.

    Returns:
        Response: The outgoing HTTP response, annotated with a request
            ID and standard security headers.
    """
    request_id = uuid.uuid4().hex
    start_time = time.monotonic()

    logger.info(
        "Request started | id=%s | method=%s | path=%s | client=%s",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.exception(
            "Request failed | id=%s | method=%s | path=%s | elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.monotonic() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"

    logger.info(
        "Request completed | id=%s | method=%s | path=%s | status=%d | elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response


# ==================================================================== #
# Global exception handlers
# ==================================================================== #


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors with a structured JSON response.

    Args:
        request: The incoming HTTP request that failed validation.
        exc: The raised validation error.

    Returns:
        JSONResponse: A 422 response describing the validation failure.
    """
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handle standard HTTP exceptions with a structured JSON response.

    Args:
        request: The incoming HTTP request.
        exc: The raised HTTP exception.

    Returns:
        JSONResponse: A response mirroring the exception's status code.
    """
    logger.warning("HTTP exception on %s: %s", request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(TTSInferenceError)
async def tts_inference_exception_handler(
    request: Request, exc: TTSInferenceError
) -> JSONResponse:
    """
    Handle domain-specific TTS inference errors with a structured JSON
    response.

    Args:
        request: The incoming HTTP request.
        exc: The raised TTS inference error.

    Returns:
        JSONResponse: A 500 response describing the synthesis failure.
    """
    logger.error("TTS inference error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "tts_inference_error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception, ensuring clients
    always receive a well-formed JSON error response.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception instance.

    Returns:
        JSONResponse: A 500 response with a generic error message.
    """
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path),
        },
    )


# ==================================================================== #
# Core endpoints
# ==================================================================== #


@app.get("/", tags=["General"])
async def read_root() -> Dict[str, str]:
    """
    Root endpoint reporting basic service identity and status.

    Returns:
        Dict[str, str]: A JSON object with the service name and status.
    """
    return {"name": APP_TITLE, "status": "running"}


@app.get("/health", tags=["General"])
async def health_check() -> Dict[str, Any]:
    """
    Report the operational health of the TTS model and inference
    service, including device and GPU diagnostics.

    Returns:
        Dict[str, Any]: Structured health information.

    Raises:
        HTTPException: Implicitly via the global handler if the health
            check itself fails unexpectedly.
    """
    service = get_tts_service()
    model_health = service.health_check()
    gpu_info = get_gpu_info()

    is_healthy = bool(model_health.get("is_loaded", False))

    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "environment": settings.ENV,
        "model": model_health,
        "gpu": gpu_info,
    }


@app.get("/version", tags=["General"])
async def get_version() -> Dict[str, str]:
    """
    Report the application and configured TTS model version details.

    Returns:
        Dict[str, str]: Version metadata for the service and model.
    """
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "environment": settings.ENV,
        "model_name": settings.TTS_MODEL_NAME,
    }


# ==================================================================== #
# Routers
# ==================================================================== #

from services.api.routes import tts as tts_routes  # noqa: E402
from services.api.routes import stt as stt_routes  # noqa: E402

app.include_router(tts_routes.router)
app.include_router(stt_routes.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )