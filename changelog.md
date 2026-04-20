# Changelog

## [2026-04-20] - Python Upgrade
- Upgraded project to Python 3.12 for better MLX compatibility and performance.
- Updated `.python-version`, `pyproject.toml`, `preflight.py`, and `AGENTS.md`.
- **Documentation**: Synchronized `README.md` and `AGENTS.md` with the new architecture and Python version.
- **Cleanup**: Removed stale Reranker logic from `convert_models.py` and `preflight.py`.
- Verified system integrity with `uv run preflight.py`.


### Changed
- **Service Simplification**: Dropped Reranker and MCP services to focus on the core Embedding API.
- **Automation**: Updated `restart.sh`, `start.py`, and `test_service.py` to only manage/verify the embedding service.
- **Documentation**: Updated `README.md` and `AGENTS.md` to reflect the single-service focus.
- **Security**: Finalized `.env` removal from git tracking.


## [2026-04-20] MLX Conversion

### Added
- `mlx_model.py`: Core inference engine using MLX and `mlx-embeddings`.
- `convert_models.py`: Script to download and convert BGE-M3 and Reranker models to MLX format.
- Added `mlx`, `mlx-embeddings`, `transformers`, and `huggingface_hub` to `pyproject.toml`.
- `test/api_tests.py`: Python script for verifying REST endpoints.
- `test/2604200737.md`: Documented test cases and appended successful test execution results.

### Fixed (System Integration)
- **Launchd Configuration**: Updated `com.bgem3.embed.plist` and `com.bgem3.mcp.plist` with the correct `WorkingDirectory` (`mlxbgem3`) and ZeroTier IP (`10.230.57.109`).
- **Service Optimization**: Disabled `com.bgem3.rerank.plist` and legacy `com.bgem3.server.plist` to prevent port conflicts and unnecessary resource usage.

### Fixed
- **MLXReranker Architecture**: Fixed `ImportError` for the base Model class and mocked missing pooler weights to support the BGE-Reranker checkpoint.
- **Model Conversion**: Implemented manual conversion for Reranker-v2-m3 using core MLX and transformers to bypass `mlx-embeddings` limitations.

### Changed
- `bgem3_embed.py`: Completely rewritten to use the MLX inference engine. Removed `torch` and `FlagEmbedding` dependencies.
- `bgem3_rerank.py`: Completely rewritten to use the MLX inference engine.
- `preflight.py`: Updated to verify MLX environment, Metal availability, and existence of converted models.
- `README.md`: Updated architecture diagrams and added model conversion steps.
- `AGENTS.md`: Updated for MLX stack.

### Removed
- `torch` and `FlagEmbedding` dependencies (migrated to pure MLX).
