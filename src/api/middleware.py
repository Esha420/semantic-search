"""FastAPI middleware for Request-ID tracing and latency tracking."""

import time
import uuid
from collections.abc import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.logger import logger, request_id_ctx, tenant_id_ctx


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Attaches correlation IDs to requests, tracks latency,
    and logs structured JSON access records.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Extract or generate Request ID
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        tenant_id = request.headers.get("X-Tenant-ID") or "tenant_default"

        request_id_ctx.set(req_id)
        tenant_id_ctx.set(tenant_id)

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

            # Avoid spamming access logs on health checks
            if not request.url.path.startswith("/api/v1/health"):
                logger.info(
                    f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms}ms)",
                    extra={
                        "extra": {
                            "http_method": request.method,
                            "path": request.url.path,
                            "status_code": response.status_code,
                            "latency_ms": elapsed_ms,
                        }
                    },
                )

            return response
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                exc_info=True,
                extra={"extra": {"latency_ms": elapsed_ms}},
            )
            raise
