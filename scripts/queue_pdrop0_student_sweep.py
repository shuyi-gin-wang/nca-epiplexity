"""Wait for the currently-running run_student_sweep.py to exit, then launch
a fresh run_student_sweep.py at --p-drop 0 with run-name suffix _pdrop0.

Same poll-by-wmic loop as queue_after_current.py. Lives as its own script so
we don't conflict with whatever queue_after_current was originally queued for.

Run from repo root:
    .venv/Scripts/python.exe scripts/queue_pdrop0_student_sweep.py
"""
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
SWEEP_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_student_sweep.py")
TARGET_SCRIPT_BASENAME = "run_student_sweep.py"


def student_sweep_running(self_pid: int) -> bool:
    """Return True iff at least one OTHER python is executing run_student_sweep.py.

    Excludes our own pid so we don't deadlock waiting for ourselves once we've
    launched the subprocess (we shell out via subprocess.call so we shouldn't,
    but excluding self is cheap insurance)."""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where",
             f"name='python.exe' and CommandLine like '%{TARGET_SCRIPT_BASENAME}%'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
    except subprocess.SubprocessError:
        return False
    for ln in out.splitlines():
        if TARGET_SCRIPT_BASENAME not in ln:
            continue
        # csv format: Node,CommandLine,ProcessId
        parts = ln.split(",")
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[-1].strip())
        except ValueError:
            continue
        if pid != self_pid:
            return True
    return False


def main():
    self_pid = os.getpid()
    print(f"queue_pdrop0_student_sweep: pid={self_pid}, waiting for run_student_sweep.py to finish...", flush=True)
    n_check = 0
    while student_sweep_running(self_pid):
        n_check += 1
        if n_check % 30 == 0:  # every ~30 min at 60s polls
            print(f"  ...still waiting (checked {n_check} times)", flush=True)
        time.sleep(60)
    print("run_student_sweep.py is gone; launching --p-drop 0 --name-suffix _pdrop0 sweep", flush=True)
    sys.stdout.flush()
    rc = subprocess.call([
        PYTHON, SWEEP_SCRIPT,
        "--p-drop", "0",
        "--name-suffix", "_pdrop0",
    ])
    print(f"pdrop0 student-arch sweep exit={rc}", flush=True)


if __name__ == "__main__":
    main()
