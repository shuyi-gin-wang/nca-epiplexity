"""Wait for the current student sweep, then launch a p_drop=0 student sweep.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/sweeps/queue_student_pdrop0.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT

REPO_ROOT = str(REPO_ROOT)
PYTHON = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
SWEEP_SCRIPT = str(Path(__file__).resolve().with_name("student_probe.py"))
TARGET_SCRIPT_BASENAMES = ("student_probe.py", "run_student_sweep.py")


def student_sweep_running(self_pid: int) -> bool:
    """Return True iff another python process is executing the student sweep."""
    where_clause = " or ".join(
        f"CommandLine like '%{marker}%'" for marker in TARGET_SCRIPT_BASENAMES
    )
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", f"name='python.exe' and ({where_clause})",
             "get", "ProcessId,CommandLine", "/format:csv"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError:
        return False

    for line in out.splitlines():
        if not any(marker in line for marker in TARGET_SCRIPT_BASENAMES):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[-1].strip())
        except ValueError:
            continue
        if pid != self_pid:
            return True
    return False


def main() -> None:
    self_pid = os.getpid()
    print(f"queue_student_pdrop0: pid={self_pid}, waiting for existing student sweep...", flush=True)
    n_check = 0
    while student_sweep_running(self_pid):
        n_check += 1
        if n_check % 30 == 0:
            print(f"  ...still waiting (checked {n_check} times)", flush=True)
        time.sleep(60)

    print("student sweep is gone; launching --p-drop 0 --name-suffix _pdrop0 sweep", flush=True)
    sys.stdout.flush()
    rc = subprocess.call([
        PYTHON,
        SWEEP_SCRIPT,
        "--p-drop",
        "0",
        "--name-suffix",
        "_pdrop0",
    ])
    print(f"pdrop0 student-arch sweep exit={rc}", flush=True)


if __name__ == "__main__":
    main()
