import asyncio
import os
import time
from typing import Any

import anyio
import psutil
import mlx.core as mx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mlx_model import MLXReranker
from pydantic import BaseModel

load_dotenv()

_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_bearer_scheme = HTTPBearer(auto_error=False)


def _check_api_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    if not _API_KEY:
        return
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


MAX_QUEUE: int = int(os.getenv("RERANK_MAX_QUEUE", "4"))
INFERENCE_TIMEOUT: float = float(os.getenv("RERANK_INFERENCE_TIMEOUT", "60.0"))

_mlx_lock: asyncio.Semaphore
_reranker: MLXReranker = None
_queue_depth: int = 0


async def _run_with_gpu_lock(fn) -> Any:
    global _queue_depth

    if _queue_depth >= MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {_queue_depth} requests queued. Retry later.",
        )

    _queue_depth += 1
    try:
        async with _mlx_lock:
            try:
                result = await asyncio.wait_for(
                    anyio.to_thread.run_sync(fn),
                    timeout=INFERENCE_TIMEOUT,
                )
                return result
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"Inference timed out after {INFERENCE_TIMEOUT}s.",
                )
    finally:
        _queue_depth -= 1


async def lifespan(app: FastAPI):
    global _mlx_lock, _reranker
    _mlx_lock = asyncio.Semaphore(1)
    try:
        model_path = os.getenv("RERANK_MODEL_PATH", "models/bge-reranker-v2-m3-mlx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Reranker model not found at {model_path}")
            
        _reranker = MLXReranker(model_path)
        
        # Warmup
        _reranker.compute_score([["warmup query", "warmup passage"]], batch_size=1)
        
        print("bge-reranker-v2-m3 ready on MLX")
        if _API_KEY:
            print("Auth enabled")
    except Exception as e:
        print(f"Failed to initialise reranker: {e}")
        raise
    yield
    if _reranker is not None:
        del _reranker
        mx.metal.clear_cache()


app = FastAPI(title="BGE Reranker Service (MLX)", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}s"
    return response


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    top_n: int = 0


class ScoredPassage(BaseModel):
    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    results: list[ScoredPassage]
    query: str
    model: str
    total_passages: int
    returned: int


@app.get("/health")
def health() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "status": "healthy",
        "service": "reranker-mlx",
        "model": "BAAI/bge-reranker-v2-m3 (MLX)",
        "timestamp": int(time.time()),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": mem.percent,
        "mps_available": True,
        "semaphore_locked": _mlx_lock.locked() if _mlx_lock is not None else False,
        "queue_depth": _queue_depth,
    }


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "model": "BAAI/bge-reranker-v2-m3 (MLX)",
        "type": "cross-encoder",
        "device": "metal",
        "auth_enabled": bool(_API_KEY),
        "framework": "mlx",
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> RerankResponse:
    _check_api_key(credentials)

    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be empty")
    if not req.passages:
        raise HTTPException(status_code=422, detail="passages list cannot be empty")
    if len(req.passages) > 100:
        raise HTTPException(status_code=429, detail="max 100 passages")

    pairs = [[req.query, p] for p in req.passages]

    try:
        scores: list[float] = await _run_with_gpu_lock(
            lambda: _reranker.compute_score(pairs, batch_size=4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")

    scored = sorted(
        [
            ScoredPassage(index=i, score=float(s), text=p)
            for i, (p, s) in enumerate(zip(req.passages, scores))
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    top_n = req.top_n if req.top_n > 0 else len(scored)
    results = scored[:top_n]

    return RerankResponse(
        results=results,
        query=req.query,
        model="BAAI/bge-reranker-v2-m3-mlx",
        total_passages=len(req.passages),
        returned=len(results),
    )
