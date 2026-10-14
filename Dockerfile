# Production Multi-Stage Dockerfile for Document Intelligence API
FROM python:3.12-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for PostgreSQL, PyMuPDF, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazingly fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

# Copy dependency definition
COPY pyproject.toml README.md ./

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy application source code and configurations
COPY src/ /app/src/
COPY ui/ /app/ui/
COPY data/ /app/data/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY .env.example /app/.env.example

# Create unprivileged user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
