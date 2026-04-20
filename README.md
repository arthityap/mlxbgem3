# BGEM3-MLX Embedding Service

Self-hosted embedding generation service using the [BAAI/bge-m3](https://github.com/flagattribute/FlagEmbedding) model, optimized for Apple Silicon Macs using the **MLX** framework.

## Purpose

- Generate **dense embeddings** (1024-dimensional) for text retrieval using MLX
- Generate **hybrid embeddings** (dense + sparse) for hybrid search
- Expose via REST API for AI agent integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BGEM3-MLX Service                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐                                     │
│  │bgem3_embed:8000   │                                     │
│  │   (FastAPI)       │                                     │
│  │ - /embed         │                                     │
│  │ - /embed/hybrid  │                                     │
│  │ - /health        │                                     │
│  │ - /info          │                                     │
│  └────────┬────────┘                                     │
│           │         BGE-M3 Model                        │
│           │    (mlx + mlx-embeddings)                   │
│           └──────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## CRITICAL - Always Use These Conventions

> **ALWAYS use `uv` for all Python commands** — never `pip`, never bare `python`
> 
> **ALWAYS use the ZeroTier IP `10.230.57.109`** — never `0.0.0.0`, never `127.0.0.1`
> 
> **ALWAYS use Authorization Bearer token** — not `X-API-Key` header

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3)
- ZeroTier connected to network `10.230.57.x` (IP: `10.230.57.109` by default)
- **Python 3.12+**

### Installation & Startup

**Step 1: Sync dependencies**
```bash
uv sync
```

**Step 2: Convert Model to MLX**
```bash
uv run convert_models.py
```
This downloads and converts BGE-M3 to MLX format.

**Step 3: Run preflight check**
```bash
uv run preflight.py
```

**Step 4: Start Service**
```bash
./restart.sh
```
*Note: Reranker and MCP services have been removed to focus on the core embedding service.*

### Test It
```bash
uv run test_service.py
```

## API Reference

### Embedding API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Generate dense embeddings (max 32 texts) |
| `/embed/hybrid` | POST | Generate dense + sparse embeddings (max 8 texts) |
| `/health` | GET | Health check with system metrics |
| `/info` | GET | Service and model information |

#### Authentication
Use Authorization Bearer:
```bash
curl -X POST http://10.230.57.109:8000/embed \
  -H "Authorization: Bearer m1macmini" \
  -H "Content-Type: application/json" \
  -d '["your text here"]'
```

## Files

| File | Purpose |
|------|----------|
| `restart.sh` | Starts Embedding service |
| `preflight.py` | Environment checks for MLX and Metal |
| `mlx_model.py` | Core MLX inference engine |
| `bgem3_embed.py` | FastAPI embedding service |
| `test_service.py` | Smoke tests |
| `changelog.md` | History of changes and MLX migration |

## Auto-Start (launchd)

The service is configured to start on boot via LaunchAgent:
```bash
launchctl load -w ~/Library/LaunchAgents/com.bgem3.embed.plist
```