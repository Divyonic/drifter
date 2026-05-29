#!/usr/bin/env bash
# Launch the Context Drift Monitor Streamlit app.
#
# Creates a local virtual environment (.venv) if missing, installs the runtime
# requirements into it, then launches the Streamlit app. Re-running is cheap:
# the venv and already-satisfied requirements are reused.
#
# Usage:
#   chmod +x run.sh   # once
#   ./run.sh
set -euo pipefail

# Resolve the directory this script lives in so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Pick a Python interpreter (prefer python3.14, then python3, then python).
PY_BIN=""
for candidate in python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY_BIN="$candidate"
        break
    fi
done
if [ -z "$PY_BIN" ]; then
    echo "Error: no Python interpreter found on PATH." >&2
    exit 1
fi

# Create the virtual environment if it does not already exist.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PY_BIN" -m venv "$VENV_DIR"
fi

# Activate the virtual environment.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install / update runtime dependencies.
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# Launch the app, bound to localhost only (not exposed on the network).
exec streamlit run app.py --server.address localhost
