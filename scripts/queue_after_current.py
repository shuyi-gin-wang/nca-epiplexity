"""Wait for the currently-running run_student_sweep.py to exit, then launch
run_pdrop0_sweep.py.

Robust against polling errors: any exception during the alive-check is caught
and treated as "still alive" so we never accidentally launch the pdrop0 sweep
while the student sweep is still using the GPU.

Run from repo root:
    .venv/Scripts/python.exe scripts/queue_after_current.py
"""
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
PDROP0_SWEEP = os.path.join(REPO_ROOT, "scripts", "run_pdrop0_sweep.py")
TARGET_SCRIPT_BASENAME = "run_student_sweep.py"
POLL_INTERVAL_SEC = 120


def driver_pids():
    """Return PIDs of any python.exe whose command line mentions run_student_sweep.py.

    Uses tasklist /V which includes the window title (~/cmdline marker)."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/V", "/FO", "CSV", "/NH"],
            stderr=subprocess.DEVNULL, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None  # signal "unknown" → caller should assume still running
    pids = []
    for line in out.splitlines():
        if TARGET_SCRIPT_BASENAME in line and "python" in line.lower():
            parts = [p.strip('"') for p in line.split('","')]
            # CSV: image,pid,sessionname,session#,memuse,status,user,cputime,windowtitle
            if len(parts) >= 2 and parts[1].isdigit():
                pids.append(int(parts[1]))
    return pids


def main():
    print(f"queue_after_current: polling every {POLL_INTERVAL_SEC}s for '{TARGET_SCRIPT_BASENAME}'", flush=True)
    n_polls = 0
    consecutive_clear = 0
    while True:
        n_polls += 1
        try:
            pids = driver_pids()
        except Exception as e:  # never let an unexpected exception kill the waiter
            print(f"  poll {n_polls}: unexpected error: {e!r} — assuming still alive", flush=True)
            pids = None

        if pids is None:
            consecutive_clear = 0
            if n_polls % 15 == 0:
                print(f"  poll {n_polls}: tasklist unavailable, assuming still running", flush=True)
        elif len(pids) == 0:
            consecutive_clear += 1
            # Require two consecutive clear polls before launching, in case the
            # driver is briefly between subprocess invocations.
            if consecutive_clear >= 2:
                print(f"  poll {n_polls}: driver gone for 2 consecutive checks — launching pdrop0 sweep", flush=True)
                break
            print(f"  poll {n_polls}: no driver PID found ({consecutive_clear}/2 to launch)", flush=True)
        else:
            consecutive_clear = 0
            if n_polls % 30 == 0:
                print(f"  poll {n_polls}: driver alive (pids={pids})", flush=True)
        time.sleep(POLL_INTERVAL_SEC)

    sys.stdout.flush()
    rc = subprocess.call([PYTHON, PDROP0_SWEEP])
    print(f"queue_after_current: pdrop0 sweep finished with exit={rc}", flush=True)


if __name__ == "__main__":
    main()
