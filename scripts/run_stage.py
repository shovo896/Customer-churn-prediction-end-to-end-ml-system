import os
import subprocess
import sys
from pathlib import Path


def resolve_python() -> str:
    env_python = os.getenv("PROJECT_PYTHON")
    if env_python:
        return env_python

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / ".venv311" / "Scripts" / "python.exe",
        root / ".venv311" / "bin" / "python",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_stage.py <script_path>")
        return 2

    script = sys.argv[1]
    python_bin = resolve_python()

    cmd = [python_bin, script]
    print(f"Running stage with: {python_bin} {script}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
