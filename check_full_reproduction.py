"""Check whether this machine can run the full BAAIWorm simulation."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def command_version(command: str, *args: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=10, check=False
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except Exception as exc:  # diagnostic tool: preserve the failure reason
        return f"ERROR: {exc}"


def main() -> int:
    checks = {
        "OS": platform.platform(),
        "Python": sys.version.split()[0],
        "NEURON module": "found" if importlib.util.find_spec("neuron") else "missing",
        "nrnivmodl": command_version("nrnivmodl", "--version") or "missing",
        "CMake": command_version("cmake", "--version") or "missing",
        "NVIDIA GPU": command_version("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader") or "missing",
        "CUDA compiler": command_version("nvcc", "--version") or "missing",
    }
    for key, value in checks.items():
        print(f"{key:16}: {value}")
    ready = (
        platform.system() == "Linux"
        and checks["NEURON module"] == "found"
        and checks["nrnivmodl"] != "missing"
        and checks["CMake"] != "missing"
        and checks["CUDA compiler"] != "missing"
    )
    print(f"\nFull simulation ready: {ready}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
