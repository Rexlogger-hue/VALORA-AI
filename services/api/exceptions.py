"""
services/api/exceptions.py

Custom exception types and centralized exception handler registration
for the Valora AI API.

Defines a small hierarchy of API-level exceptions (`APIException` and
its subclasses) and a `register_exception_handlers` function that wires
them, along with FastAPI/Starlette built-ins, into structured JSON
error responses on a given `FastAPI` application instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("valora.api.exceptions")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class APIException(Exception):
    """
    Base exception for all Valora AI API-level errors.

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code to return for this error.
        error_code: Machine-readable error category identifier.
        extra: Optional additional structured context for the response.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "api_error"

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the API exception.

        Args:
            message: Human-readable error message.
            status_code: Optional override for the HTTP status code.
            error_code: Optional override for the machine-readable
                error category identifier.
            extra: Optional additional structured context to include
                in the JSON error response.
        """
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        self.extra = extra or {}


class ValidationException(APIException):
    """Raised when request input fails domain-level validation."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "validation_error"


class ModelLoadException(APIException):
    """Raised when the TTS model fails to load or is unavailable."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_load_error"


class InferenceException(APIException):
    """Raised when speech synthesis/inference fails."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "inference_error"


def _build_error_body(
    error_code: str, detail: Any, path: str, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a structured JSON error response body.

    Args:
        error_code: Machine-readable error category identifier.
        detail: Human-readable or structured error detail.
        path: The request path that produced the error.
        extra: Optional additional structured context to merge in.

    Returns:
        Dict[str, Any]: The structured error response body.
    """
    body: Dict[str, Any] = {
        "error": error_code,
        "detail": detail,
        "path": path,
    }
    if extra:
        body.update(extra)
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all custom and built-in exception handlers on the given
    FastAPI application instance.

    Args:
        app: The FastAPI application to register handlers on.
    """

    @app.exception_handler(ValidationException)
    async def validation_exception_handler(
        request: Request, exc: ValidationException
    ) -> JSONResponse:
        """
        Handle domain-level validation failures.

        Args:
            request: The incoming HTTP request.
            exc: The raised validation exception.

        Returns:
            JSONResponse: A structured JSON error response.
        """
        logger.warning("ValidationException on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body(exc.error_code, exc.message, str(request.url.path), exc.extra),
        )

    @app.exception_handler(ModelLoadException)
    async def model_load_exception_handler(
        request: Request, exc: ModelLoadException
    ) -> JSONResponse:
        """
        Handle TTS model load/availability failures.

        Args:
            request: The incoming HTTP request.
            exc: The raised model load exception.

        Returns:
            JSONResponse: A structured JSON error response.
        """
        logger.error("ModelLoadException on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body(exc.error_code, exc.message, str(request.url.path), exc.extra),
        )

    @app.exception_handler(InferenceException)
    async def inference_exception_handler(
        request: Request, exc: InferenceException
    ) -> JSONResponse:
        """
        Handle speech synthesis/inference failures.

        Args:
            request: The incoming HTTP request.
            exc: The raised inference exception.

        Returns:
            JSONResponse: A structured JSON error response.
        """
        logger.error("InferenceException on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body(exc.error_code, exc.message, str(request.url.path), exc.extra),
        )

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        """
        Handle any generic/unclassified API-level exception.

        Args:
            request: The incoming HTTP request.
            exc: The raised API exception.

        Returns:
            JSONResponse: A structured JSON error response.
        """
        logger.error("APIException on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body(exc.error_code, exc.message, str(request.url.path), exc.extra),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle FastAPI/Pydantic request validation errors.

        Args:
            request: The incoming HTTP request that failed validation.
            exc: The raised validation error.

        Returns:
            JSONResponse: A structured 422 JSON error response.
        """
        logger.warning("RequestValidationError on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_build_error_body("request_validation_error", exc.errors(), str(request.url.path)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Handle standard Starlette/FastAPI HTTP exceptions.

        Args:
            request: The incoming HTTP request.
            exc: The raised HTTP exception.

        Returns:
            JSONResponse: A structured JSON error response mirroring
                the exception's status code.
        """
        logger.warning("HTTPException on %s: %s", request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_error_body("http_error", exc.detail, str(request.url.path)),
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
            JSONResponse: A generic 500 JSON error response.
        """
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_build_error_body(
                "internal_server_error",
                "An unexpected error occurred. Please try again later.",
                str(request.url.path),
            ),
        )

    logger.info("Registered Valora AI API exception handlers.")


__all__ = [
    "APIException",
    "ValidationException",
    "ModelLoadException",
    "InferenceException",
    "register_exception_handlers",
]