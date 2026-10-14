"""Structured logging configuration with context propagation."""

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Context variable for request-level tracing
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="system")
tenant_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="system")


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "tenant_id": tenant_id_ctx.get(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include custom extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_payload.update(record.extra)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


def setup_logger(name: str = "document_intelligence", level: str = "INFO") -> logging.Logger:
    """Configures and returns a logger instance with JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if re-called
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger()
