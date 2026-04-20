import os
import sys
from dotenv import load_dotenv

def check_python_version():
    print(f"Checking Python version: {sys.version.split()[0]}...", end=" ")
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print("OK")
    else:
        print("FAIL (3.12+ required)")
        return False
    return True

def check_packages():
    packages = ["mlx", "mlx_embeddings", "transformers", "fastapi", "uvicorn", "httpx"]
    print(f"Checking packages: {', '.join(packages)}...", end=" ")
    try:
        for pkg in packages:
            # mlx_embeddings uses underscore in package name but hyphen in pip
            mod_name = pkg.replace("-", "_")
            __import__(mod_name)
        print("OK")
    except ImportError as e:
        print(f"FAIL (missing {e.name})")
        return False
    return True

def check_mlx():
    print("Checking MLX Metal availability...", end=" ")
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            print("OK")
        else:
            print("FAIL (Metal not available)")
            return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False
    return True

def check_models():
    print("Checking MLX models in ./models/...", end=" ")
    embed_path = "models/bge-m3-mlx"
    if os.path.exists(embed_path):
        print("OK")
    else:
        print("FAIL (run 'uv run convert_models.py' first)")
        return False
    return True

def check_env():
    print("Checking .env for API key...", end=" ")
    load_dotenv()
    if os.getenv("EMBEDDING_API_KEY"):
        print("OK")
    else:
        print("WARNING (EMBEDDING_API_KEY not set, auth disabled)")
    return True

def run_preflight():
    print("=== BGEM3-MLX Preflight Checks ===\n")
    checks = [
        check_python_version(),
        check_packages(),
        check_mlx(),
        check_models(),
        check_env()
    ]
    if all(checks):
        print("\nPreflight SUCCESS: System ready.")
        return True
    else:
        print("\nPreflight FAILED: Fix the issues above.")
        return False

if __name__ == "__main__":
    if not run_preflight():
        sys.exit(1)
