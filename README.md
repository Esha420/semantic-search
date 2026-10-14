# Document Intelligence API (Production RAG)

A production-grade, enterprise-ready **Retrieval-Augmented Generation (RAG) and Document Intelligence API** designed to ingest complex multi-page documents (PDF, TXT, MD), index their semantics into **PostgreSQL + pgvector**, perform **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**, and generate grounded answers with **verifiable source citations** using **NVIDIA NIM LLMs**.

Built specifically to demonstrate production AI backend engineering—avoiding generic chatbot wrappers in favor of idempotent ingestion, debuggable ranking, explicit abstention on absent evidence, and observable telemetry.

---

## Architecture Diagram

![Document Intelligence API Architecture](docs/assets/architecture.png)

---

## Key Production Capabilities

### 1. Reliable & Idempotent Ingestion
* **Page-Aware PDF Parsing**: Utilizes `pymupdf` to extract structured text per page, preserving exact page boundaries required for verified enterprise citations.
* **Content Hashing & Idempotency**: Computes cryptographic SHA-256 content hashes to detect duplicate uploads instantly, preventing redundant embedding costs.
* **Document Versioning**: If a modified document with an existing filename is ingested, the engine automatically increments the version counter (`v1 -> v2`) while preserving audit history.
* **Recursive Semantic Chunking**: Splits text along logical boundaries (paragraphs, sentences) with configurable token targets and overlap while tagging chunks with exact source page numbers and indices.

### 2. High-Quality Hybrid Retrieval & Reranking
* **Dense Vector Search**: Powered by `pgvector` with **HNSW indexing** (`m=16, ef_construction=64`) using cosine distance for sub-millisecond semantic similarity.
* **PostgreSQL Full-Text Search**: Leverages PostgreSQL `tsvector`, `ts_query`, and `ts_rank_cd` to ensure domain-specific terminology, policy numbers, and acronyms are accurately captured.
* **Reciprocal Rank Fusion (RRF)**: Combines dense vector ranks and sparse keyword ranks into a unified score ($RRF(d) = \sum \frac{1}{k + rank(d)}$), outperforming single-modality search.
* **Contextual Reranking**: Re-evaluates top candidate passages against query semantics with lexical overlap and cross-attention scoring.
* **Debuggable Response Payloads**: Retrieval endpoints return complete scoring breakdowns (dense score, sparse score, fused score, rerank rank, latency).

### 3. Grounded Generation with Citations & Abstention
* **NVIDIA NIM LLM Integration**: Powered by `meta/llama-3.2-11b-vision-instruct` for fast, highly articulate reasoning, coupled with `nvidia/nemotron-3-embed-1b` (2048 dimensions).
* **Verifiable Source Citations**: Every generated factual statement cites its exact supporting passage `[1]`, `[2]`, mapped directly to `document_title`, `filename`, `page_number`, and `snippet`.
* **Multi-Layered Abstention Defense**:
  1. *Early Abstention*: If no documents match or if top retrieval similarity falls below the `ABSTENTION_THRESHOLD` (default 0.35), the system abstains before invoking the LLM, saving token latency and cost (`LOW_RETRIEVAL_CONFIDENCE`).
  2. *Model Abstention*: If retrieved context does not contain the required facts, the LLM emits an explicit `INSUFFICIENT_EVIDENCE` token rather than hallucinating answers.
* **Prompt Injection Immunity**: Context passages are encapsulated inside secure delimiters `<context>` with strict system rules forbidding user text from overriding operational guidelines.

### 4. Model Context Protocol (MCP) Server
* Ships with an **MCP Server** (`src/mcp/server.py`) exposing `search_documents` and `ask_document_intelligence` tools, enabling AI IDEs (Cursor, Claude Desktop, Windsurf) to query the knowledge engine directly.

### 5. Automated Evaluation & Benchmarking
* Integrated evaluation suite (`/api/v1/evaluation/run`) measuring **Precision@K**, **Recall@K**, **Grounded Citation Rate**, and **Abstention Accuracy** against ground-truth benchmark datasets.

---

## Technology Stack

| Layer | Component | Choice | Rationale |
| :--- | :--- | :--- | :--- |
| **API & Framework** | REST API | Python 3.12 + FastAPI + Pydantic v2 | High-performance, async, strictly typed schemas and OpenAPI documentation |
| **Database** | Metadata & Vector Store | PostgreSQL 16 + pgvector | Unified relational metadata, ACID transactions, and HNSW vector index |
| **Full-Text Search** | Lexical Matching | PostgreSQL `tsvector` + GIN Index | Built-in keyword retrieval without managing separate Elasticsearch clusters |
| **LLM Provider** | Answer Generation | NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct`) | Enterprise inference with low latency and high instruction adherence |
| **Embedding Model** | Dense Representations | NVIDIA NIM (`nvidia/nemotron-3-embed-1b`) | 2048-dimensional dense vector embeddings with passage/query specialization |
| **PDF Extraction** | Document Ingestion | PyMuPDF (`pymupdf`) | High-speed C-native text extraction with page coordinate tracking |
| **Background Queue** | Asynchronous Processing | Celery + Redis | Asynchronous background processing for multi-megabyte enterprise documents |
| **Observability** | Telemetry & Logs | Structured JSON Logging + Request Correlation IDs | Standardized observability layer compatible with Datadog, OpenTelemetry, Opik |
| **User Interface** | Interactive Dashboard | Streamlit | Complete UI with ingestion hub, hybrid search explorer, and citation viewer |

---

## Project Structure

```
├── .env.example                  # Documented environment template
├── .env                          # Local active environment configuration
├── Dockerfile                    # Production multi-stage Dockerfile
├── docker-compose.yml            # Postgres (pgvector), Redis, API, Celery, Streamlit
├── pyproject.toml                # Modular dependencies managed with uv
├── README.md                     # Comprehensive documentation
├── docs/
│   └── assets/
│       └── architecture.png      # System architecture diagram
├── alembic/                      # Database migrations
│   ├── env.py
│   └── versions/
│       └── 001_initial_rag_schema.py
├── src/
│   ├── config.py                 # Pydantic BaseSettings
│   ├── logger.py                 # JSON-structured logging with request-id context
│   ├── database.py               # Async SQLAlchemy engine and session dependency
│   ├── models/                   # SQLAlchemy Database Models
│   │   ├── document.py           # Document (title, hash, status, tenant, version)
│   │   └── chunk.py              # DocumentChunk (Vector(2048), page_number, content)
│   ├── schemas/                  # Pydantic validation schemas
│   │   ├── document.py           # DocumentCreate, DocumentResponse, IngestionStatus
│   │   ├── retrieval.py          # RetrievalQuery, ChunkMatch, RetrievalResponse
│   │   ├── rag.py                # RAGQueryRequest, Citation, AbstentionReason, RAGResponse
│   │   └── evaluation.py         # BenchmarkTestCase, BenchmarkEvalResult
│   ├── ingestion/                # Document Ingestion Service
│   │   ├── parser.py             # PyMuPDF extractor with page boundary preservation
│   │   ├── hasher.py             # SHA-256 idempotency & version incrementing
│   │   ├── chunker.py            # Recursive semantic text chunker
│   │   └── pipeline.py           # Ingestion orchestrator
│   ├── embeddings/               # Swappable Embedding Provider Layer
│   │   ├── base.py               # Abstract base protocol
│   │   ├── nvidia.py             # NVIDIA NIM Nemotron-3 embedding API
│   │   └── mock.py               # Deterministic unit test vector generator
│   ├── retrieval/                # Hybrid Retrieval & Reranking Service
│   │   ├── vector_search.py      # pgvector cosine similarity search
│   │   ├── fts_search.py         # PostgreSQL full-text search
│   │   ├── hybrid.py             # Reciprocal Rank Fusion (RRF) algorithm
│   │   ├── reranker.py           # Contextual cross-encoder reranker
│   │   └── engine.py             # Unified retrieval engine
│   ├── generation/               # Grounded Generation & LLM Factory
│   │   ├── base.py               # Base LLM provider interface
│   │   ├── nvidia.py             # NVIDIA NIM Chat completion provider
│   │   ├── mock.py               # Deterministic mock LLM provider
│   │   ├── prompt_templates.py   # Grounded prompt template with anti-injection defenses
│   │   └── engine.py             # End-to-end RAG workflow with citation parser
│   ├── tasks/                    # Celery Background Workers
│   │   ├── celery_app.py         # Celery instance configuration
│   │   └── ingestion_tasks.py    # Background parsing and indexing task
│   ├── api/                      # FastAPI REST API Gateway
│   │   ├── deps.py               # Dependency injection (DB session, API keys, Tenant ID)
│   │   ├── middleware.py         # Request-ID tracking & latency profiling
│   │   ├── routes/               # Modular API routes
│   │   │   ├── health.py         # /health, /health/live, /health/ready
│   │   │   ├── documents.py      # Upload, list, detail, chunk inspector, delete
│   │   │   ├── retrieval.py      # Hybrid search with score breakdown
│   │   │   ├── rag.py            # Grounded Q&A with citations & abstention
│   │   │   └── evaluation.py     # Automated benchmark evaluation
│   │   └── app.py                # FastAPI application factory
│   ├── mcp/                      # Model Context Protocol
│   │   └── server.py             # MCP Tool Server (search_documents, ask_document_intelligence)
│   └── evaluation/               # Evaluation & Quality Metrics
│       └── metrics.py            # Precision@K, Recall@K, Abstention Accuracy
├── ui/
│   └── app.py                    # Streamlit web interface
├── data/
│   └── synthetic_samples/        # Sample insurance & enterprise policies for demo
│       ├── property_casualty_policy.txt
│       └── underwriting_risk_guidelines.txt
└── tests/                        # Comprehensive Pytest Suite (19 passed tests)
    ├── conftest.py               # In-memory test DB, async client, mock providers
    ├── test_parser.py            # PDF & text extraction tests
    ├── test_hasher.py            # SHA-256 idempotency & versioning tests
    ├── test_chunker.py           # Recursive chunking & page numbering tests
    ├── test_hybrid_retrieval.py  # RRF scoring & reranker tests
    ├── test_rag_engine.py        # Citations & abstention tests
    ├── test_api_endpoints.py     # FastAPI REST integration tests
    └── test_evaluation.py        # Benchmark calculation tests
```

---

## Getting Started

### Prerequisites
* Python `>=3.11` (or Docker & Docker Compose)
* [uv](https://docs.astral.sh/uv/) (recommended for instant package installation)
* NVIDIA NIM API Key (or run with mock providers in offline mode)

### 1. Environment Setup

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

Key environment parameters:
```ini
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.2-11b-vision-instruct

EMBEDDING_PROVIDER=nvidia
EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
EMBEDDING_DIMENSIONS=2048

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/document_intelligence
SYNC_DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/document_intelligence
```

### 2. Local Installation & Launch (with `uv`)

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,ui]"

# Run database migrations
alembic upgrade head

# Start FastAPI REST API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI documentation is immediately available at `http://localhost:8000/docs`.

### 3. Launching Interactive Streamlit Dashboard

In a separate terminal:
```bash
streamlit run ui/app.py --server.port 8501
```
Open `http://localhost:8501` to access the interactive web interface.

### 4. Running with Docker Compose (Full Stack)

Spin up PostgreSQL with pgvector, Redis, FastAPI, Celery worker, and Streamlit in one command:

```bash
docker compose up --build
```

---

## API Endpoints Reference

### 1. Document Ingestion (`POST /api/v1/documents/upload`)
Uploads and indexes a multi-page document into pgvector.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "X-Tenant-ID: verisk_commercial" \
  -F "file=@data/synthetic_samples/property_casualty_policy.txt" \
  -F "title=Commercial Property Policy 2026"
```

**Response (201 Created):**
```json
{
  "id": "e9b5f472-a1b2-4d44-8dfc-30919df01234",
  "tenant_id": "verisk_commercial",
  "title": "Commercial Property Policy 2026",
  "filename": "property_casualty_policy.txt",
  "file_type": ".txt",
  "file_size_bytes": 2842,
  "content_hash": "a82f9d14ec7b8f9e6a715f6...",
  "version": 1,
  "status": "completed",
  "total_chunks": 4,
  "created_at": "2026-10-14T20:30:00Z"
}
```

### 2. Hybrid Retrieval Search (`POST /api/v1/retrieval/search`)
Executes vector similarity + keyword search combined via Reciprocal Rank Fusion.

```bash
curl -X POST "http://localhost:8000/api/v1/retrieval/search" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: verisk_commercial" \
  -d '{
    "query": "What is the deductible for water damage?",
    "top_k": 3,
    "search_mode": "hybrid",
    "alpha": 0.65,
    "rerank": true
  }'
```

**Response (200 OK):**
```json
{
  "query": "What is the deductible for water damage?",
  "total_matches": 3,
  "search_mode": "hybrid",
  "alpha": 0.65,
  "reranking_applied": true,
  "latency_ms": 14.2,
  "matches": [
    {
      "chunk_id": "c1f8a84b-0123...",
      "document_id": "e9b5f472-a1b2...",
      "document_title": "Commercial Property Policy 2026",
      "filename": "property_casualty_policy.txt",
      "page_number": 1,
      "chunk_index": 1,
      "content": "2.3 Water Damage Deductible: $15,000 per occurrence for water leaks arising from plumbing or fire suppression failures.",
      "score": 0.9412,
      "dense_score": 0.892,
      "sparse_score": 0.965,
      "rerank_score": 0.9412,
      "rank": 1
    }
  ]
}
```

### 3. Grounded RAG Query (`POST /api/v1/rag/query`)
Answers user questions with explicit citations and abstention support.

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: verisk_commercial" \
  -d '{
    "query": "Under what conditions will a water leak claim be paid and what is the deductible?",
    "top_k": 5,
    "search_mode": "hybrid"
  }'
```

**Response (200 OK):**
```json
{
  "query": "Under what conditions will a water leak claim be paid and what is the deductible?",
  "answer": "Accidental water discharge from plumbing systems is a covered peril under Section 1.2 [1]. The water damage deductible is set at $15,000 per occurrence for plumbing or fire suppression failures [2]. However, continuous moisture penetration exceeding 14 calendar days is explicitly excluded [3].",
  "is_abstained": false,
  "abstention_reason": null,
  "citations": [
    {
      "citation_id": 1,
      "document_id": "e9b5f472-a1b2...",
      "document_title": "Commercial Property Policy 2026",
      "filename": "property_casualty_policy.txt",
      "page_number": 1,
      "chunk_index": 0,
      "snippet": "1.2 Covered Perils: (e) Sprinkler leakage and accidental water discharge from plumbing systems...",
      "relevance_score": 0.92
    },
    {
      "citation_id": 2,
      "document_id": "e9b5f472-a1b2...",
      "document_title": "Commercial Property Policy 2026",
      "filename": "property_casualty_policy.txt",
      "page_number": 1,
      "chunk_index": 1,
      "snippet": "2.3 Water Damage Deductible: $15,000 per occurrence for water leaks arising from plumbing...",
      "relevance_score": 0.94
    }
  ],
  "model_name": "meta/llama-3.2-11b-vision-instruct",
  "latency": {
    "retrieval_ms": 12.4,
    "generation_ms": 410.2,
    "total_ms": 422.6
  },
  "usage": {
    "prompt_tokens": 348,
    "completion_tokens": 68,
    "total_tokens": 416
  },
  "request_id": "f83a2190-2139-4ce8-b572-c2e36128091a"
}
```

### 4. Abstention Scenario (Absent Evidence)
When a query asks for information not present in the ingested corpus:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the policy limit for commercial satellites orbiting Mars?"
  }'
```

**Response (Abstention):**
```json
{
  "query": "What is the policy limit for commercial satellites orbiting Mars?",
  "answer": "The available documents have very low relevance to this question. Abstaining from generating an ungrounded response.",
  "is_abstained": true,
  "abstention_reason": {
    "code": "LOW_RETRIEVAL_CONFIDENCE",
    "details": "Top retrieval score (0.12) is below the confidence abstention threshold (0.35)."
  },
  "citations": []
}
```

---

## Automated Test Suite

The test suite validates:
1. `test_parser`: PyMuPDF page-preserving extraction and text format dispatch.
2. `test_hasher`: SHA-256 cryptographic idempotency and version tracking.
3. `test_chunker`: Recursive text splitting and page coordinate tagging.
4. `test_hybrid_retrieval`: Reciprocal Rank Fusion (RRF) math and contextual reranking.
5. `test_rag_engine`: Grounded answer generation, citation preservation, and abstention triggers.
6. `test_api_endpoints`: Health probes, file upload, search, and lifecycle deletion.
7. `test_evaluation`: Precision@K, Recall@K, and Abstention accuracy metrics.

Run the test suite locally:

```bash
.venv/bin/pytest -v
```

**Results:**
```
============================== 19 passed in 1.19s ==============================
```

---

## Interview Talking Points (Verisk Python Developer Role)

When discussing this project in technical interviews, highlight these architectural decisions:

1. **Why Hybrid Search (Vector + Full-Text) over Vector Alone?**
   * *Problem:* Pure dense vector search frequently fails on exact policy codes, endorsement identifiers (e.g., "Form FL-90"), numbers, and dates.
   * *Solution:* We combine PostgreSQL `pgvector` cosine similarity with PostgreSQL `tsvector` full-text search via **Reciprocal Rank Fusion (RRF)**, yielding both semantic understanding and exact keyword precision.

2. **How is Grounding and Hallucination Mitigated?**
   * *Defense-in-Depth:* 
     1. Retrieval confidence gate: If top candidate similarity $< 0.35$, the system abstains before LLM invocation, saving token costs.
     2. Strict System Prompt: Mandates that every sentence cite a bracketed number `[1]`, `[2]` corresponding to an ingested chunk.
     3. Explicit Token Abstention: If facts are missing, the LLM emits `INSUFFICIENT_EVIDENCE`.
     4. Post-processing citation verification: The engine parses cited markers and attaches verifiable metadata (Document, Page, Snippet).

3. **Why PostgreSQL + pgvector instead of a separate vector DB?**
   * *Simplest Production Architecture:* Rather than operating separate Milvus/Pinecone clusters alongside a relational DB, PostgreSQL + pgvector provides unified ACID transactions, relational document metadata, access controls, full-text search, and HNSW vector indexing in a single proven database engine.

4. **Idempotency in Ingestion Pipelines:**
   * Calculating SHA-256 hashes of incoming document bytes prevents duplicate indexing, protects vector store integrity, and saves external embedding API costs.

---

## License

This project is licensed under the MIT License.