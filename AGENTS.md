# BGEM3 Agent Guide

This document provides essential information for AI agents working in the BGEM3 repository.

## CRITICAL CONVENTIONS - NEVER FORGET

> **ALWAYS use `uv`** for all Python commands — never `pip`, never bare `python`
> 
> **ALWAYS use ZeroTier IP** — default `10.230.57.109`, but always load from `ZT_IP` in `.env`
> 
> **ALWAYS use Authorization Bearer token** — not `X-API-Key` header

## Project Overview

BGEM3 provides embedding generation via BAAI/bge-m3. Fully converted to **MLX** for native Apple Silicon performance. Reranker and MCP services have been removed to focus on the core embedding API.

## Essential Commands

### Quick Start (recommended)

1. **Convert models**: `uv run convert_models.py`
2. **Preflight**: `uv run preflight.py`
3. **Restart**: `./restart.sh`

### Full Test

```bash
uv run test_service.py
```

Expected: `All 5 tests passed.`

### Preflight Checks

```bash
uv run preflight.py
```

Validates: Python 3.12, packages, GPU, cached weights, port 8000, `.env` has API key.

### Manual Service Start

```bash
# embed (port 8000)
uv run uvicorn bgem3_embed:app --host 10.230.57.109 --port 8000 &
```

### Stop Service

```bash
pkill -f "uvicorn bgem3_embed"
```

## Architecture and Data Flow

### Components

- `bgem3_embed.py`: FastAPI embedding service (port 8000)
- `restart.sh`: Primary startup script — kills stale, starts embed service
- `preflight.py`: Pre-startup validation
- `test_service.py`: Smoke tests

### Port Allocation

- 8000: bgem3_embed (BGE-M3 embeddings)

## API Usage

### Embedding (port 8000)

**CORRECT**:
```bash
curl -X POST http://10.230.57.109:8000/embed \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'
```

## Key Patterns and Conventions

### load_dotenv Required

`bgem3_embed.py` calls `load_dotenv()` at module level:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Model Loading

Models loaded at module level on import. Uses MLX backend:

```python
from mlx_model import MLXBGEM3
_model = MLXBGEM3("models/bge-m3")
```

### Logging

Logs in `logs/` directory:
- `logs/bgem3_embed.log`

### Authentication

- API key: `m1macmini` (set in `.env` via `EMBEDDING_API_KEY`)
- Use header: `Authorization: Bearer m1macmini`

## Testing

```bash
uv run test_service.py
```

Runs 5 tests:
- `/health` alive + healthy
- `/info` returning model info
- `/embed` returning 1024-dim vector
- `/embed/hybrid` returning dense + sparse
- `/embed` rejects bad API key (401)