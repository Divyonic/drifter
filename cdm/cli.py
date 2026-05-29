"""Command-line entry point for the Context Drift Monitor (``drifter``).

Subcommands::

    drifter            # launch the web app (default)
    drifter run        # launch the web app
    drifter watch      # run the clipboard auto-capture watcher in the foreground
    drifter version    # print the version

Any extra arguments after ``run`` are passed through to Streamlit, e.g.
``drifter run --server.port 9000``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

__all__ = ["main", "app_path"]


def app_path() -> str:
    """Absolute path to the bundled Streamlit app (``cdm/app.py``)."""
    return str(Path(__file__).resolve().parent / "app.py")


def _run_app(extra: Optional[List[str]] = None) -> int:
    """Launch the Streamlit app, forwarding any extra Streamlit args.

    Binds to localhost by default so the app is never exposed on the network;
    pass ``--server.address 0.0.0.0`` explicitly if you really want LAN access.
    """
    extra = list(extra or [])
    if not any(str(a).startswith("--server.address") for a in extra):
        extra = ["--server.address", "localhost", *extra]
    cmd = [sys.executable, "-m", "streamlit", "run", app_path(), *extra]
    return subprocess.call(cmd)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch a subcommand."""
    parser = argparse.ArgumentParser(
        prog="drifter",
        description="Context Drift Monitor — track and correct LLM goal drift.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Launch the Drifter web app (default).")
    sub.add_parser("app", help="Alias for 'run'.")
    sub.add_parser("watch", help="Run the clipboard auto-capture watcher.")
    sub.add_parser("version", help="Print the installed version.")

    args, extra = parser.parse_known_args(argv)
    command = args.command or "run"

    if command in ("run", "app"):
        return _run_app(extra)
    if command == "watch":
        from cdm.watcher import main as watch_main

        return watch_main()
    if command == "version":
        from cdm import __version__

        print(__version__)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
