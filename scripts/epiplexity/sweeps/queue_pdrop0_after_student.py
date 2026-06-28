"""Wait for the currently-running student sweep, then launch pdrop0 targets.

Run from repo root:
    .venv/Scripts/python.exe scripts/epiplexity/sweeps/queue_pdrop0_after_student.py
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
PDROP0_SWEEP = str(Path(__file__).resolve().with_name("pdrop0_targets.py"))
TARGET_SCRIPT_BASENAMES = ("student_probe.py", "run_student_sweep.py")
POLL_INTERVAL_SEC = 120


def driver_pids() -> list[int] | None:
    """Return PIDs of python processes whose command line mentions the student sweep."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/V", "/FO", "CSV", "/NH"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    pids = []
    for line in out.splitlines():
        if not any(marker in line for marker in TARGET_SCRIPT_BASENAMES):
            continue
        if "python" not in line.lower():
            continue
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def main() -> None:
    markers = ", ".join(TARGET_SCRIPT_BASENAMES)
    print(f"queue_pdrop0_after_student: polling every {POLL_INTERVAL_SEC}s for {markers}", flush=True)
    n_polls = 0
    consecutive_clear = 0
    while True:
        n_polls += 1
        try:
            pids = driver_pids()
        except Exception as exc:  # keep the waiter alive on unexpected polling errors
            print(f"  poll {n_polls}: unexpected error: {exc!r}; assuming still alive", flush=True)
            pids = None

        if pids is None:
            consecutive_clear = 0
            if n_polls % 15 == 0:
                print(f"  poll {n_polls}: tasklist unavailable, assuming still running", flush=True)
        elif len(pids) == 0:
            consecutive_clear += 1
            if consecutive_clear >= 2:
                print(f"  poll {n_polls}: driver gone for 2 consecutive checks; launching pdrop0 sweep", flush=True)
                break
            print(f"  poll {n_polls}: no driver PID found ({consecutive_clear}/2 to launch)", flush=True)
        else:
            consecutive_clear = 0
            if n_polls % 30 == 0:
                print(f"  poll {n_polls}: driver alive (pids={pids})", flush=True)
        time.sleep(POLL_INTERVAL_SEC)

    sys.stdout.flush()
    rc = subprocess.call([PYTHON, PDROP0_SWEEP])
    print(f"queue_pdrop0_after_student: pdrop0 sweep finished with exit={rc}", flush=True)


if __name__ == "__main__":
    main()
