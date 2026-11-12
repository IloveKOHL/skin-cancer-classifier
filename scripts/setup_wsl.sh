#!/usr/bin/env bash
# Create the Python environment on Linux / WSL2 (run from the repository root).
#
# Prerequisite (once, needs sudo) - Python 3.11 on Ubuntu 24.04:
#   sudo add-apt-repository -y ppa:deadsnakes/ppa
#   sudo apt-get install -y python3.11 python3.11-venv
#
# The venv is created in the Linux home directory, not in the repository:
# on WSL2 the Windows drive (/mnt/c) is slow for many small files.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/skin-cancer}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

python3.11 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet torch torchvision --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install --quiet -r requirements.txt
"$VENV/bin/python" - <<'EOF'
import torch, torchvision, timm
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "Activate with: source $VENV/bin/activate"
