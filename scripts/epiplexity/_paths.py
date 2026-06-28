"""Shared paths for epiplexity scripts."""
from __future__ import annotations

import sys
import os
from pathlib import Path


EPIPLEXITY_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = EPIPLEXITY_ROOT.parent
REPO_ROOT = SCRIPT_ROOT.parent
DEMO_OUT = SCRIPT_ROOT / "demo_out"
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def add_repo_root_to_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
