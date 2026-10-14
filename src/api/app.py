"""FastAPI application factory and middleware configuration."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import ObservabilityMiddleware
from src.api.routes import documents, evaluation, health, rag, retrieval
from src.config import settings
from src.database import init_db
from src.logger import logger, request_id_ctx


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle hooks: initialize database extensions on startup."""
    logger.info(f"Starting {settings.APP_NAME} in '{settings.ENVIRONMENT}' environment...")
    await init_db()
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Document Intelligence RAG API",
        description=(
            "Production-grade Document Intelligence and Grounded RAG API with "
            "PyMuPDF ingestion, PostgreSQL/pgvector hybrid retrieval, cross-encoder reranking, "
            "NVIDIA NIM LLM integration, verifiable citations, and explicit abstention."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability & Request Tracing Middleware
    app.add_middleware(ObservabilityMiddleware)

    # Global Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = request_id_ctx.get()
        logger.error(f"Global unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred.",
                "detail": str(exc) if settings.DEBUG else None,
                "request_id": req_id,
            },
        )

    # Register API v1 Sub-routers
    v1_prefix = settings.API_V1_PREFIX
    app.include_router(health.router, prefix=v1_prefix)
    app.include_router(documents.router, prefix=v1_prefix)
    app.include_router(retrieval.router, prefix=v1_prefix)
    app.include_router(rag.router, prefix=v1_prefix)
    app.include_router(evaluation.router, prefix=v1_prefix)

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.APP_NAME,
            "version": "1.0.0",
            "docs": "/docs",
            "health": f"{v1_prefix}/health",
        }

    return app


app = create_app()
