"""Lightweight progress logging for notebook runs executed under Slurm."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def log_progress(message: str) -> None:
    """Print a progress message and mirror it to PTBXL_PROGRESS_LOG when set."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)

    progress_log = os.environ.get("PTBXL_PROGRESS_LOG")
    if not progress_log:
        return

    path = Path(progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")