import asyncio
import math
import os
import time
from typing import Any, Dict

import anyio
import psutil
import mlx.core as mx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mlx_model import MLXBGEM3Model

load_dotenv()

# Module-level semaphore and model variable
_mlx_lock: asyncio.Semaphore
_model: MLXBGEM3Model = None

# Concurrency controls
MAX_QUEUE: int = int(os.getenv("EMBED_MAX_QUEUE", "8"))
INFERENCE_TIMEOUT: float = float(os.getenv("EMBED_INFERENCE_TIMEOUT", "30.0"))

# Chunking — max texts per GPU pass to keep memory predictable
EMBED_CHUNK_SIZE: int = int(os.getenv("EMBED_CHUNK_SIZE", "32"))
HYBRID_CHUNK_SIZE: int = int(os.getenv("HYBRID_CHUNK_SIZE", "8"))

# Hard cap — purely a sanity guard, not a concurrency limit
MAX_TEXTS: int = int(os.getenv("EMBED_MAX_TEXTS", "512"))

_queue_depth: int = 0

_EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
_bearer_scheme = HTTPBearer(auto_error=False)


def _check_api_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    if not _EMBEDDING_API_KEY:
        return
    if credentials is None or credentials.credentials != _EMBEDDING_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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


def _chunk(lst: list, size: int) -> list[list]:
    """Split list into sublists of at most `size` items."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


async def lifespan(app: FastAPI):
    global _mlx_lock, _model
    _mlx_lock = asyncio.Semaphore(1)
    try:
        model_path = os.getenv("EMBED_MODEL_PATH", "models/bge-m3-mlx")
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found. Ensure you ran convert_models.py")
            raise RuntimeError(f"Model not found at {model_path}")

        _model = MLXBGEM3Model(model_path)

        # Warmup
        _model.encode_dense(["warmup text"], batch_size=1)

        print("BGE-M3 ready on MLX")
        if _EMBEDDING_API_KEY:
            print("Auth enabled")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise
    yield
    if _model is not None:
        del _model
        mx.metal.clear_cache()


app = FastAPI(title="BGEM3 Embedding Service (MLX)", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    return {
        "status": "healthy",
        "service": "bgem3-mlx",
        "timestamp": int(time.time()),
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "mps_available": True,
        "semaphore_locked": _mlx_lock.locked() if _mlx_lock is not None else False,
        "queue_depth": _queue_depth,
    }


@app.get("/info")
def info() -> Dict[str, Any]:
    return {
        "model": "BAAI/bge-m3 (MLX)",
        "dimensions": 1024,
        "device": "metal",
        "framework": "mlx",
        "auth_enabled": bool(_EMBEDDING_API_KEY),
        "max_texts": MAX_TEXTS,
        "chunk_size": EMBED_CHUNK_SIZE,
    }


@app.post("/embed")
async def embed(
    texts: list[str],
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> Dict[str, Any]:
    _check_api_key(credentials)

    if not texts:
        raise HTTPException(status_code=422, detail="Text list cannot be empty")
    if len(texts) > MAX_TEXTS:
        raise HTTPException(status_code=422, detail=f"max {MAX_TEXTS} texts per request")

    try:
        chunks = _chunk(texts, EMBED_CHUNK_SIZE)
        all_embeddings = []

        for chunk in chunks:
            chunk_copy = chunk  # capture for lambda
            result = await _run_with_gpu_lock(
                lambda c=chunk_copy: _model.encode_dense(c, batch_size=8)
            )
            all_embeddings.extend(result.tolist())

        return {
            "embeddings": all_embeddings,
            "count": len(texts),
            "dimensions": 1024,
            "model": "BAAI/bge-m3-mlx",
            "chunks_processed": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/hybrid")
async def embed_hybrid(
    texts: list[str],
    credentials: HTTPAuthorizationCredentials | None = __import__("fastapi").Depends(
        _bearer_scheme
    ),
) -> Dict[str, Any]:
    _check_api_key(credentials)

    if not texts:
        raise HTTPException(status_code=422, detail="Text list cannot be empty")
    if len(texts) > MAX_TEXTS:
        raise HTTPException(status_code=422, detail=f"max {MAX_TEXTS} texts per request")

    try:
        chunks = _chunk(texts, HYBRID_CHUNK_SIZE)
        all_dense = []
        all_sparse = []

        for chunk in chunks:
            dense = await _run_with_gpu_lock(
                lambda c=chunk: _model.encode_dense(c, batch_size=4)
            )
            sparse = await _run_with_gpu_lock(
                lambda c=chunk: _model.encode_sparse(c, batch_size=4)
            )
            all_dense.extend(dense.tolist())
            all_sparse.extend(sparse)

        return {
            "dense_embeddings": all_dense,
            "sparse_embeddings": all_sparse,
            "model": "BAAI/bge-m3-mlx",
            "count": len(texts),
            "chunks_processed": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
