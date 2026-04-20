"""
start.py — One-shot startup validator for the embedding service.

  bgem3_embed    port 8000  (BGE-M3 embeddings)

NOTE: This script is a one-shot validator — it does NOT start or stop
      services, and exits cleanly after confirming everything is healthy.

Steps (fails fast on any error):
  1. Run preflight checks
  2. Wait for bgem3_embed /health
  3. Run smoke tests (test_service.py)
  4. Report status and exit 0

Usage:
    uv run python start.py
"""

import os
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────────────────────────
ZT_IP = os.getenv("ZT_IP", "10.230.57.109")
RAG_PORT = 8000
RAG_URL = f"http://{ZT_IP}:{RAG_PORT}"
HEALTH_TIMEOUT = 60  # seconds (model load ~30s)
HEALTH_POLL = 2
LOG_DIR = "logs"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────


def info(msg: str) -> None:
    print(f"{BOLD}[INFO]{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")
    sys.exit(1)


def wait_for_health(url: str, timeout: int, poll: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────────────────────────

print(f"\n{BOLD}=== BGE-M3 Startup Validator ==={RESET}\n")
os.makedirs(LOG_DIR, exist_ok=True)

# Step 1: Preflight
info("Step 1/3: Running preflight checks...")
result = subprocess.run([sys.executable, "preflight.py"])
if result.returncode != 0:
    fail("Preflight failed. Fix issues above before starting.")
ok("Preflight passed")

# Step 2: bgem3_embed health
info(f"Step 2/3: Waiting for bgem3_embed /health at {RAG_URL} ...")
if not wait_for_health(RAG_URL, HEALTH_TIMEOUT, HEALTH_POLL):
    fail(f"bgem3_embed not healthy after {HEALTH_TIMEOUT}s. Check logs/bgem3_embed.log")
ok(f"bgem3_embed healthy → {RAG_URL}")

# Step 3: Smoke tests
info("Step 3/3: Running smoke tests...")
result = subprocess.run([sys.executable, "test_service.py"])
if result.returncode != 0:
    fail("Smoke tests failed — service running but something is wrong. Check logs/.")
ok("All smoke tests passed")

print(f"\n{GREEN}{BOLD}=== Service up and healthy ==={RESET}")
print(f"  bgem3_embed   → {RAG_URL}    log: logs/bgem3_embed.log")
print()
sys.exit(0)
