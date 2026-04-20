import os

import httpx
from fastmcp import FastMCP

# stateless_http=True: no session state stored between requests.
mcp = FastMCP("BGEM3")

ZT_IP = os.getenv("ZT_IP", "10.230.57.109")
BGEM3_URL = f"http://{ZT_IP}:8000"
RERANK_URL = f"http://{ZT_IP}:8002"
API_KEY = os.getenv("EMBEDDING_API_KEY", "m1macmini")
MCP_PORT = 8001

_EMBED_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
_HYBRID_TIMEOUT = httpx.Timeout(connect=5.0, read=45.0, write=5.0, pool=5.0)
_RERANK_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)


@mcp.tool()
async def embed(texts: list[str]) -> list[list[float]]:
    """Generate BGE-M3 dense embeddings for a list of texts.

    Args:
        texts: List of strings to embed. Max 32 texts per call.

    Returns:
        List of 1024-dimensional float vectors, one per input text.
    """
    async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
        try:
            r = await client.post(
                f"{BGEM3_URL}/embed",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=texts,
            )
            r.raise_for_status()
            return r.json()["embeddings"]
        except httpx.TimeoutException as e:
            raise ValueError(f"bgem3_embed timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"bgem3_embed error {e.response.status_code}: {e.response.text}"
            ) from e


@mcp.tool()
async def embed_hybrid(texts: list[str]) -> dict:
    """Generate BGE-M3 dense + sparse embeddings in a single pass.

    Use this for hybrid vector search (dense + BM25-style sparse).

    Args:
        texts: List of strings to embed. Max 8 texts per call.

    Returns:
        Dict with 'dense_embeddings' (1024-dim) and 'sparse_embeddings' (token weights).
    """
    async with httpx.AsyncClient(timeout=_HYBRID_TIMEOUT) as client:
        try:
            r = await client.post(
                f"{BGEM3_URL}/embed/hybrid",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=texts,
            )
            r.raise_for_status()
            return r.json()
        except httpx.TimeoutException as e:
            raise ValueError(f"bgem3_embed hybrid timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"bgem3_embed hybrid error {e.response.status_code}: {e.response.text}"
            ) from e


@mcp.tool()
async def rerank(query: str, passages: list[str], top_n: int = 5) -> list[dict]:
    """Rerank a list of passages for a given query using BGE-Reranker-v2-m3.

    Args:
        query: The search query.
        passages: List of passages to rerank.
        top_n: Number of top results to return.

    Returns:
        List of dicts with 'index', 'text', and 'score', sorted by score descending.
    """
    async with httpx.AsyncClient(timeout=_RERANK_TIMEOUT) as client:
        try:
            r = await client.post(
                f"{RERANK_URL}/rerank",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"query": query, "passages": passages, "top_n": top_n},
            )
            r.raise_for_status()
            return r.json()["results"]
        except httpx.TimeoutException as e:
            raise ValueError(f"bgem3_rerank timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"bgem3_rerank error {e.response.status_code}: {e.response.text}"
            ) from e


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=ZT_IP, port=MCP_PORT, stateless_http=True)
