"""Streamlit Web UI for Document Intelligence RAG API."""

import io
import json
import os
import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .citation-tag {
        background-color: #0284c7;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .abstention-box {
        background-color: #451a03;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    tenant_id = st.text_input("Tenant ID", value="tenant_default")
    api_key = st.text_input("API Key (optional)", type="password", value="")

    headers = {"X-Tenant-ID": tenant_id}
    if api_key:
        headers["X-API-Key"] = api_key

    st.divider()
    st.markdown("### 🔌 API Health")
    if st.button("Check Service Status"):
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{api_url}/health/ready", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    st.success("API Online")
                    st.json(data.get("checks", {}))
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

st.markdown('<div class="main-header">Document Intelligence API</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade RAG with Hybrid Retrieval, Citations & NVIDIA NIM</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["📚 Ingestion Hub", "🔍 Hybrid Search Explorer", "🤖 Grounded RAG Assistant", "📊 Benchmark Eval"])

# ----------------- TAB 1: INGESTION HUB -----------------
with tabs[0]:
    st.markdown("#### Document Ingestion & Idempotency Pipeline")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Document (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            help="PyMuPDF extracts pages; chunks are indexed into PostgreSQL + pgvector.",
        )
        custom_title = st.text_input("Document Title (optional)")

        if st.button("Ingest Document", type="primary", disabled=uploaded_file is None):
            with st.spinner("Extracting, hashing, chunking, and generating embeddings..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    data = {}
                    if custom_title:
                        data["title"] = custom_title

                    with httpx.Client(timeout=60.0) as client:
                        res = client.post(f"{api_url}/documents/upload", headers=headers, files=files, data=data)
                        if res.status_code == 201:
                            doc = res.json()
                            st.success(f"Successfully Ingested: **{doc['title']}**")
                            st.json({
                                "Document ID": doc["id"],
                                "Content Hash": doc["content_hash"][:12] + "...",
                                "Total Chunks": doc["total_chunks"],
                                "Version": doc["version"],
                                "Status": doc["status"],
                            })
                        else:
                            st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

    with col2:
        st.markdown("#### Indexed Document Catalog")
        if st.button("Refresh Catalog"):
            st.rerun()

        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(f"{api_url}/documents", headers=headers)
                if res.status_code == 200:
                    catalog = res.json()
                    st.caption(f"Total documents: {catalog['total']}")
                    for doc in catalog.get("items", []):
                        with st.expander(f"📄 {doc['title']} (v{doc['version']})"):
                            st.write(f"**Filename:** `{doc['filename']}` | **Size:** {doc['file_size_bytes']} bytes")
                            st.write(f"**Status:** `{doc['status']}` | **Chunks:** {doc['total_chunks']}")
                            st.caption(f"Hash: `{doc['content_hash']}`")
                else:
                    st.info("No documents found or unable to fetch catalog.")
        except Exception:
            st.info("API server is starting or not reachable.")

# ----------------- TAB 2: HYBRID SEARCH EXPLORER -----------------
with tabs[1]:
    st.markdown("#### Hybrid Retrieval & Reciprocal Rank Fusion (RRF)")

    c_search, c_mode, c_alpha, c_topk = st.columns([3, 1, 1, 1])
    with c_search:
        search_query = st.text_input("Search query", placeholder="e.g. What are the roof inspection guidelines?")
    with c_mode:
        search_mode = st.selectbox("Search Mode", ["hybrid", "vector", "keyword"])
    with c_alpha:
        alpha = st.slider("Dense Alpha", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    with c_topk:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    rerank_toggle = st.checkbox("Enable Contextual Reranking", value=True)

    if st.button("Execute Search", type="primary"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": search_query,
                        "top_k": top_k,
                        "search_mode": search_mode,
                        "alpha": alpha,
                        "rerank": rerank_toggle,
                    }
                    with httpx.Client(timeout=20.0) as client:
                        res = client.post(f"{api_url}/retrieval/search", headers=headers, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.success(f"Retrieved {data['total_matches']} matches in {data['latency_ms']} ms")
                            for m in data.get("matches", []):
                                page_str = f"Page {m['page_number']}" if m['page_number'] else "Page N/A"
                                with st.container():
                                    st.markdown(f"**Rank #{m['rank']}** — `{m['filename']}` ({page_str})")
                                    st.caption(
                                        f"Combined Score: `{m['score']}` | "
                                        f"Dense: `{m.get('dense_score')}` | "
                                        f"Sparse: `{m.get('sparse_score')}`"
                                    )
                                    st.info(m["content"])
                                    st.divider()
                        else:
                            st.error(f"Search failed: {res.text}")
                except Exception as e:
                    st.error(f"Search error: {e}")

# ----------------- TAB 3: GROUNDED RAG ASSISTANT -----------------
with tabs[2]:
    st.markdown("#### Precision Q&A Grounded in Verified Citations")
    q_input = st.text_area(
        "Ask a Question",
        placeholder="e.g. Under what circumstances will a flood damage claim be denied?",
        height=90,
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        ask_btn = st.button("Generate Answer", type="primary")

    if ask_btn:
        if not q_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving evidence and querying NVIDIA LLM..."):
                try:
                    payload = {
                        "query": q_input,
                        "top_k": 5,
                        "search_mode": "hybrid",
                        "rerank": True,
                    }
                    with httpx.Client(timeout=60.0) as client:
                        res = client.post(f"{api_url}/rag/query", headers=headers, json=payload)
                        if res.status_code == 200:
                            rag_data = res.json()

                            # Check if model abstained
                            if rag_data.get("is_abstained"):
                                reason = rag_data.get("abstention_reason", {})
                                st.markdown(
                                    f"""
                                    <div class="abstention-box">
                                        <h4>⚠️ Abstention: Insufficient Evidence</h4>
                                        <p><strong>Code:</strong> <code>{reason.get('code')}</code></p>
                                        <p>{reason.get('details')}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.info(f"**System Response:** {rag_data['answer']}")
                            else:
                                st.markdown("### Answer")
                                st.write(rag_data["answer"])

                                # Source Citations
                                citations = rag_data.get("citations", [])
                                if citations:
                                    st.markdown("### 📑 Verified Evidence Citations")
                                    for c in citations:
                                        page_lbl = f"Page {c['page_number']}" if c['page_number'] else "Page N/A"
                                        st.markdown(
                                            f"**[{c['citation_id']}]** `{c['filename']}` ({page_lbl}) — *Score: {c['relevance_score']}*"
                                        )
                                        st.caption(f'"{c["snippet"]}"')

                            # Performance Metrics
                            lat = rag_data.get("latency", {})
                            st.caption(
                                f"⏱️ Retrieval: {lat.get('retrieval_ms', 0)}ms | "
                                f"Generation: {lat.get('generation_ms', 0)}ms | "
                                f"Total: {lat.get('total_ms', 0)}ms | "
                                f"Model: `{rag_data.get('model_name')}`"
                            )
                        else:
                            st.error(f"RAG Query Failed: {res.text}")
                except Exception as e:
                    st.error(f"Error communicating with API: {e}")

# ----------------- TAB 4: BENCHMARK EVALUATION -----------------
with tabs[3]:
    st.markdown("#### Automated Retrieval & Abstention Evaluation Benchmark")
    st.write(
        "Run repeatable evaluation measuring Precision@K, Recall@K, Grounded Citations, and Abstention Accuracy."
    )

    if st.button("Run Evaluation Suite"):
        sample_benchmark = [
            {
                "query": "What is the deductible for commercial water damage?",
                "is_abstention_case": False,
            },
            {
                "query": "What is the capital city of ancient Atlantis in year 2050?",
                "is_abstention_case": True,
            },
        ]
        with st.spinner("Running benchmark test cases..."):
            try:
                with httpx.Client(timeout=60.0) as client:
                    res = client.post(f"{api_url}/evaluation/run", headers=headers, json=sample_benchmark)
                    if res.status_code == 200:
                        eval_data = res.json()
                        st.success("Benchmark Run Complete!")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Abstention Accuracy", f"{eval_data['abstention_accuracy'] * 100:.1f}%")
                        m2.metric("Mean Precision@K", f"{eval_data['mean_precision_at_k'] * 100:.1f}%")
                        m3.metric("Citation Rate", f"{eval_data['grounded_citation_rate'] * 100:.1f}%")
                        m4.metric("Avg Latency", f"{eval_data['average_latency_ms']} ms")
                        st.json(eval_data.get("test_cases", []))
                    else:
                        st.error(f"Evaluation error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
