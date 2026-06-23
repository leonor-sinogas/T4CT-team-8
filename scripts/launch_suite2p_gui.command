#!/bin/zsh
# Launch the suite2p desktop GUI locally (it can't run in Colab — it's a PyQt app).
# First run: creates a python3.12 venv and installs suite2p[gui]; later runs just open it.
# Reuse another venv with:  SUITE2P_VENV=/path/to/venv ./launch_suite2p_gui.command
set -e

VENV="${SUITE2P_VENV:-$HOME/suite2p_venv}"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 not found (suite2p needs Python 3.9-3.12, not 3.13)."
  echo "Install it with:  brew install python@3.12"
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating suite2p venv at: $VENV"
  python3.12 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
fi

if ! "$VENV/bin/python" -c "import suite2p" >/dev/null 2>&1; then
  echo "Installing suite2p[gui] into $VENV (a few minutes the first time)..."
  "$VENV/bin/pip" install "suite2p[gui]"
fi

echo "Launching suite2p GUI from $VENV ..."
exec "$VENV/bin/python" -m suite2p
