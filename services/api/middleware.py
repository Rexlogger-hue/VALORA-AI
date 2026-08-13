"""
services/api/middleware.py

Centralized middleware registration for the Valora AI API.

Provides request logging, response timing, request ID generation,
security headers, GZip compression, trusted host validation, and CORS
configuration, all wired together via a single `register_middlewares`
entry point for use in `services/api/main.py`.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from services.tts.utils import get_logger

logger = get_logger("valora.api.middleware")

DEFAULT_ALLOWED_ORIGINS: List[str] = ["*"]
DEFAULT_ALLOWED_HOSTS: List[str] = ["*"]
DEFAULT_GZIP_MIN_SIZE: int = 1024


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique request ID to every incoming
    request, logs request/response lifecycle events, measures response
    timing, and attaches standard security headers to every response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process an incoming request: generate a request ID, time the
        handler execution, log the outcome, and attach security headers.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the middleware chain.

        Returns:
            Response: The outgoing HTTP response, annotated with a
                request ID, timing header, and security headers.
        """
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        start_time = time.monotonic()

        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "Request started | id=%s | method=%s | path=%s | client=%s",
            request_id,
            request.method,
            request.url.path,
            client_host,
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

        self._apply_security_headers(response)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "Request completed | id=%s | method=%s | path=%s | status=%d | elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response

    @staticmethod
    def _apply_security_headers(response: Response) -> None:
        """
        Attach standard security headers to an outgoing response.

        Args:
            response: The response to annotate in place.
        """
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )


def _register_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
) -> None:
    """
    Register CORS middleware on the application.

    Args:
        app: The FastAPI application instance.
        allowed_origins: List of allowed origin URLs. Defaults to
            `DEFAULT_ALLOWED_ORIGINS` ("*") if not provided.
    """
    origins = allowed_origins if allowed_origins is not None else DEFAULT_ALLOWED_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-Audio-Duration-Seconds",
                         "X-Sample-Rate", "X-Latency-Seconds", "X-Voice-Preset"],
    )
    logger.info("CORS middleware registered (origins=%s).", origins)


def _register_trusted_hosts(
    app: FastAPI,
    allowed_hosts: Optional[List[str]] = None,
) -> None:
    """
    Register trusted host validation middleware on the application.

    Args:
        app: The FastAPI application instance.
        allowed_hosts: List of allowed host headers. Defaults to
            `DEFAULT_ALLOWED_HOSTS` ("*") if not provided.
    """
    hosts = allowed_hosts if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    logger.info("TrustedHost middleware registered (hosts=%s).", hosts)


def _register_gzip(app: FastAPI, minimum_size: int = DEFAULT_GZIP_MIN_SIZE) -> None:
    """
    Register GZip compression middleware on the application.

    Args:
        app: The FastAPI application instance.
        minimum_size: Minimum response size, in bytes, before GZip
            compression is applied.
    """
    app.add_middleware(GZipMiddleware, minimum_size=minimum_size)
    logger.info("GZip middleware registered (minimum_size=%d bytes).", minimum_size)


def _register_request_context(app: FastAPI) -> None:
    """
    Register the request logging, timing, request ID, and security
    header middleware on the application.

    Args:
        app: The FastAPI application instance.
    """
    app.add_middleware(RequestContextMiddleware)
    logger.info("RequestContext middleware registered (logging, timing, security headers).")


def register_middlewares(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allowed_hosts: Optional[List[str]] = None,
    gzip_minimum_size: int = DEFAULT_GZIP_MIN_SIZE,
) -> None:
    """
    Register all Valora AI middleware on the given FastAPI application,
    in the correct order (outermost first): security/logging context,
    trusted hosts, CORS, then GZip compression.

    Args:
        app: The FastAPI application instance to configure.
        allowed_origins: Optional list of allowed CORS origins.
            Defaults to allow all origins.
        allowed_hosts: Optional list of allowed host headers for
            trusted host validation. Defaults to allow all hosts.
        gzip_minimum_size: Minimum response size, in bytes, before
            GZip compression is applied.
    """
    _register_request_context(app)
    _register_trusted_hosts(app, allowed_hosts)
    _register_cors(app, allowed_origins)
    _register_gzip(app, gzip_minimum_size)

    logger.info("All Valora AI middleware registered successfully.")


__all__ = [
    "RequestContextMiddleware",
    "register_middlewares",
]