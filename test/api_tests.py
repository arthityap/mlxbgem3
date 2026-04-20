import httpx
import os
from dotenv import load_dotenv

load_dotenv()

ZT_IP = os.getenv("ZT_IP", "10.230.57.109")
BASE_URL_EMBED = f"http://{ZT_IP}:8000"
BASE_URL_RERANK = f"http://{ZT_IP}:8002"
API_KEY = os.getenv("EMBEDDING_API_KEY", "m1macmini")

def test_health():
    print("Testing Health Endpoints...")
    r1 = httpx.get(f"{BASE_URL_EMBED}/health")
    assert r1.status_code == 200
    print("Embed Health: OK")
    
    r2 = httpx.get(f"{BASE_URL_RERANK}/health")
    assert r2.status_code == 200
    print("Rerank Health: OK")

def test_embed_dense():
    print("Testing Dense Embedding...")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    texts = ["This is a test sentence.", "Another test sentence."]
    r = httpx.post(f"{BASE_URL_EMBED}/embed", headers=headers, json=texts)
    assert r.status_code == 200
    data = r.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 1024
    print("Dense Embedding: OK")

def test_embed_hybrid():
    print("Testing Hybrid Embedding...")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    texts = ["Testing hybrid retrieval."]
    r = httpx.post(f"{BASE_URL_EMBED}/embed/hybrid", headers=headers, json=texts)
    assert r.status_code == 200
    data = r.json()
    assert "dense_embeddings" in data
    assert "sparse_embeddings" in data
    assert len(data["sparse_embeddings"]) == 1
    print("Hybrid Embedding: OK")

def test_rerank():
    print("Testing Rerank...")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "query": "What is MLX?",
        "passages": [
            "MLX is a machine learning framework for Apple Silicon.",
            "The weather is nice today.",
            "FastAPI is a web framework for Python."
        ]
    }
    r = httpx.post(f"{BASE_URL_RERANK}/rerank", headers=headers, json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["text"].startswith("MLX")
    print("Rerank: OK")

def test_auth_failure():
    print("Testing Auth Failure...")
    headers = {"Authorization": "Bearer wrong_key"}
    r = httpx.post(f"{BASE_URL_EMBED}/embed", headers=headers, json=["test"])
    assert r.status_code == 401
    print("Auth Failure Detection: OK")

if __name__ == "__main__":
    try:
        test_health()
        test_embed_dense()
        test_embed_hybrid()
        test_rerank()
        test_auth_failure()
        print("\nAll API tests passed!")
    except Exception as e:
        print(f"\nTests FAILED: {e}")
        exit(1)
