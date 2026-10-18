"""Running the pipeline steps from a notebook, with output you can watch.

IPython's `!cmd` does not reliably stream a long-running child process on
Windows: a 100-epoch training run looks frozen for an hour and then dumps
everything at once. `sh` drives the subprocess itself and prints each line as it
arrives, so the per-epoch table shows up while it is training.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def emit(text: str) -> None:
    """Print a line the child produced, whatever encoding this kernel has.

    Progress bars use box-drawing characters; a kernel on a legacy code page
    would otherwise raise UnicodeEncodeError mid-run and kill the cell.
    """
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, "replace").decode(enc), flush=True)


def sh(*args: str, check: bool = True) -> int:
    """Run `python -u <args>` and stream its output. Returns the exit code.

    stderr is merged into stdout so progress bars come through in order, and the
    child is told to use UTF-8 and skip its own buffering.
    """
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    emit("$ " + " ".join(cmd[2:]))
    started = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(line.rstrip())
    proc.wait()
    emit(f"[exit {proc.returncode} after {(time.time() - started) / 60:.1f} min]")
    if check and proc.returncode:
        raise RuntimeError(f"{' '.join(cmd[2:])} failed with exit code {proc.returncode}")
    return proc.returncode
