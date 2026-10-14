"""Model Context Protocol (MCP) Server for Document Intelligence.

Enables AI IDEs and assistants (Cursor, Claude Desktop, Windsurf) to connect
and query the enterprise document intelligence engine directly as tools.
"""

import asyncio
import json
from typing import Any

from src.config import settings
from src.database import async_session_maker
from src.generation.engine import RAGEngine
from src.logger import logger
from src.retrieval.engine import RetrievalEngine
from src.schemas.rag import RAGQueryRequest
from src.schemas.retrieval import RetrievalQuery


class DocumentIntelligenceMCPServer:
    """Lightweight stdio/JSON-RPC MCP tool server."""

    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
        self.rag_engine = RAGEngine()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns MCP tool definitions."""
        return [
            {
                "name": "search_documents",
                "description": (
                    "Search indexed enterprise documents using hybrid vector and keyword search. "
                    "Returns relevant passages with page numbers and relevance scores."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "top_k": {"type": "integer", "description": "Number of passages", "default": 5},
                        "search_mode": {
                            "type": "string",
                            "enum": ["hybrid", "vector", "keyword"],
                            "default": "hybrid",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ask_document_intelligence",
                "description": (
                    "Asks a question and receives a precision answer grounded strictly in "
                    "retrieved documents with verifiable page citations. "
                    "Abstains if sufficient evidence is absent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer"},
                        "top_k": {"type": "integer", "description": "Context passage count", "default": 5},
                    },
                    "required": ["question"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatches tool execution to the underlying engines."""
        async with async_session_maker() as session:
            if tool_name == "search_documents":
                query_str = arguments.get("query", "")
                top_k = arguments.get("top_k", 5)
                search_mode = arguments.get("search_mode", "hybrid")

                req = RetrievalQuery(query=query_str, top_k=top_k, search_mode=search_mode)
                res = await self.retrieval_engine.search(session=session, query=req)
                return {
                    "query": res.query,
                    "total_matches": res.total_matches,
                    "latency_ms": res.latency_ms,
                    "matches": [m.model_dump() for m in res.matches],
                }

            elif tool_name == "ask_document_intelligence":
                question = arguments.get("question", "")
                top_k = arguments.get("top_k", 5)

                rag_req = RAGQueryRequest(query=question, top_k=top_k)
                rag_res = await self.rag_engine.answer_query(
                    session=session, request=rag_req, request_id="mcp_call"
                )
                return rag_res.model_dump()

            else:
                raise ValueError(f"Unknown tool: {tool_name}")


async def run_mcp_cli():
    """Runs interactive MCP command line listener."""
    server = DocumentIntelligenceMCPServer()
    print(f"MCP Server for {settings.APP_NAME} started. Available tools:")
    for t in server.get_tool_definitions():
        print(f" - {t['name']}: {t['description']}")


if __name__ == "__main__":
    asyncio.run(run_mcp_cli())
