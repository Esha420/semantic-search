"""Integration tests for FastAPI REST API endpoints."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoints(async_client: AsyncClient):
    # Liveness probe
    res_live = await async_client.get("/api/v1/health/live")
    assert res_live.status_code == 200
    assert res_live.json() == {"status": "alive"}

    # Basic health check
    res_health = await async_client.get("/api/v1/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert "Document Intelligence" in data["service"]


@pytest.mark.asyncio
async def test_document_lifecycle_endpoints(async_client: AsyncClient):
    # 1. Upload document
    sample_content = (
        "Verisk Risk Standards: Commercial roof inspections must be performed within 45 days. "
        "Roofs over 20 years old are limited to Actual Cash Value settlements."
    )
    files = {
        "file": ("risk_standards.txt", sample_content.encode("utf-8"), "text/plain"),
    }
    data = {"title": "Risk Standards 2026"}
    headers = {"X-Tenant-ID": "test_api_tenant"}

    res_upload = await async_client.post(
        "/api/v1/documents/upload",
        files=files,
        data=data,
        headers=headers,
    )
    assert res_upload.status_code == 201
    doc_data = res_upload.json()
    assert doc_data["title"] == "Risk Standards 2026"
    assert doc_data["status"] == "completed"
    assert doc_data["total_chunks"] >= 1
    doc_id = doc_data["id"]

    # 2. List documents
    res_list = await async_client.get("/api/v1/documents", headers=headers)
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1

    # 3. Get document details
    res_get = await async_client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert res_get.status_code == 200
    assert res_get.json()["id"] == doc_id

    # 4. Get chunks
    res_chunks = await async_client.get(f"/api/v1/documents/{doc_id}/chunks", headers=headers)
    assert res_chunks.status_code == 200
    chunks = res_chunks.json()
    assert len(chunks) >= 1
    assert "roof inspections" in chunks[0]["content"].lower()

    # 5. Hybrid search
    search_payload = {
        "query": "roof inspections",
        "top_k": 3,
        "search_mode": "keyword",
    }
    res_search = await async_client.post(
        "/api/v1/retrieval/search", json=search_payload, headers=headers
    )
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total_matches"] >= 1

    # 6. Delete document
    res_delete = await async_client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert res_delete.status_code == 204

    # Verify deletion
    res_not_found = await async_client.get(f"/api/v1/documents/{doc_id}", headers=headers)
    assert res_not_found.status_code == 404
