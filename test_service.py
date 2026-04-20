"""
test_service.py — Smoke-test the embedding service.

Tests:
  bgem3_embed (8000):
    1. GET  /health          — alive + healthy
    2. GET  /info            — model/device info
    3. POST /embed           — returns 1024-dim vector
    4. POST /embed/hybrid    — returns dense + sparse
    5. POST /embed (bad key) — returns HTTP 401 when auth enabled

Usage:
    uv run python test_service.py
"""

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

ZT_IP = os.getenv("ZT_IP", "10.230.57.109")
RAG_URL = f"http://{ZT_IP}:8000"
API_KEY = os.getenv("EMBEDDING_API_KEY", "m1macmini")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
TEST_TEXT = "The Ten Gods in Bazi represent the relationship between the Day Master and other elements."

PASS = "\033[32m OK  \033[0m"
FAIL = "\033[31m FAIL\033[0m"

passed = failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        print(f"[{PASS}] {label}")
        passed += 1
    else:
        print(f"[{FAIL}] {label}: {detail}")
        failed += 1


print("\n=== BGE-M3 Service Tests ===\n")

with httpx.Client(timeout=60, follow_redirects=True) as client:
    # ── bgem3_embed (8000) ──────────────────────────────────────────────────────────────
    print("--- bgem3_embed (8000) ---")

    try:
        r = client.get(f"{RAG_URL}/health")
        data = r.json()
        check(
            "/health → 200 + healthy",
            r.status_code == 200 and data.get("status") == "healthy",
        )
        print(
            f"       cpu={data.get('cpu_percent')}%  mem={data.get('memory_percent')}%  mps={data.get('mps_available')}"
        )
    except Exception as e:
        check("/health", False, str(e))

    try:
        r = client.get(f"{RAG_URL}/info")
        data = r.json()
        check("/info → 200 + model info", r.status_code == 200 and "model" in data)
        print(
            f"       model={data.get('model')}  device={data.get('device')}  auth={data.get('auth_enabled')}"
        )
    except Exception as e:
        check("/info", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        vecs = data.get("embeddings", [])
        ok_ = r.status_code == 200 and len(vecs) == 1 and len(vecs[0]) == 1024
        check("/embed → 1x1024 vector", ok_, f"status={r.status_code}")
    except Exception as e:
        check("/embed", False, str(e))

    try:
        r = client.post(f"{RAG_URL}/embed/hybrid", headers=HEADERS, json=[TEST_TEXT])
        data = r.json()
        ok_ = (
            r.status_code == 200
            and "dense_embeddings" in data
            and "sparse_embeddings" in data
        )
        check("/embed/hybrid → dense + sparse", ok_, f"status={r.status_code}")
    except Exception as e:
        check("/embed/hybrid", False, str(e))

    try:
        r = client.post(
            f"{RAG_URL}/embed",
            headers={"Authorization": "Bearer wrongkey"},
            json=[TEST_TEXT],
        )
        if API_KEY:
            check(
                "/embed rejects bad key → 401",
                r.status_code == 401,
                f"got {r.status_code}",
            )
        else:
            check("/embed auth disabled", r.status_code == 200)
    except Exception as e:
        check("/embed bad-key test", False, str(e))

# Summary
print()
if failed == 0:
    print(f"\033[32mAll {passed} tests passed.\033[0m")
else:
    print(f"\033[31m{failed} test(s) failed, {passed} passed.\033[0m")
    sys.exit(1)
