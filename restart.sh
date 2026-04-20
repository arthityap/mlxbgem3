#!/bin/bash
set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

ZT_IP="${ZT_IP:-10.230.57.109}"
LOG_DIR="logs"

echo "=== Stopping existing services ==="
pkill -f "uvicorn bgem3_embed" 2>/dev/null || true
pkill -f "uvicorn bgem3_mcp" 2>/dev/null || true
pkill -f "uvicorn bgem3_rerank" 2>/dev/null || true
sleep 2

echo "=== Starting bgem3_embed (port 8000) ==="
nohup uv run uvicorn bgem3_embed:app --host "$ZT_IP" --port 8000 > "$LOG_DIR/bgem3_embed.log" 2>&1 &
sleep 15

echo "=== Verifying ==="
curl -s "http://$ZT_IP:8000/health" | grep -q "healthy" && echo "✓ embed OK" || echo "✗ embed FAILED"

echo "=== All services started (Reranker and MCP disabled) ==="