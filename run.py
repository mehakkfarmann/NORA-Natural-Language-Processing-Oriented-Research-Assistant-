#!/usr/bin/env python3
import sys
import subprocess
import os
from pathlib import Path

def start_api():
    print("Starting NORA API backend...")
    print("Docs: http://localhost:8000/docs")
    print("Frontend: http://localhost:8000")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload" if os.getenv("DEBUG") == "1" else "--reload-dir", "backend"
    ])

def start_frontend_only():
    subprocess.run([
        sys.executable, "-m", "http.server", "8000",
        "--directory", "frontend"
    ], cwd=Path(__file__).parent)

def run_tests():
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])

if __name__ == "__main__":
    if "--frontend" in sys.argv:
        start_frontend_only()
    elif "--test" in sys.argv:
        run_tests()
    else:
        start_api()
