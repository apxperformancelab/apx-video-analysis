#!/usr/bin/env bash
# APX video analysis setup for a fresh chat (about 1-3 minutes).
#
#   bash /home/claude/apx-video-analysis/setup.sh
#
# Run it from the local clone of this repo. The scripts run straight from the clone (no copy to ~/apx/scripts).
# The two AI models come from this repo's `models` branch through git, not curl. No torch needed.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$HOME/apx/models"
mkdir -p "$M" "$HOME/apx/jobs"

python3 -c "import onnxruntime, cv2, pandas, scipy, openpyxl, PIL" 2>/dev/null || \
  pip install --break-system-packages -q --progress-bar off onnxruntime opencv-python-headless pandas scipy openpyxl pillow

if [ ! -s "$M/yolo11m.onnx" ] || [ ! -s "$M/yolo11x-pose.onnx" ]; then
  B=refs/remotes/origin/models
  git -C "$R" fetch -q --depth 1 origin "models:$B"
  git -C "$R" cat-file blob "$B:yolo11m.onnx" > "$M/yolo11m.onnx"
  : > "$M/yolo11x-pose.onnx"
  for p in 0 1 2; do git -C "$R" cat-file blob "$B:yolo11x-pose.onnx.part$p" >> "$M/yolo11x-pose.onnx"; done
  git -C "$R" cat-file blob "$B:SHA256SUMS" > "$M/SHA256SUMS"
fi
(cd "$M" && sed 's#/home/claude/onnx/##' SHA256SUMS | sha256sum -c --quiet) || \
  { echo "ERROR: models corrupted. Delete ~/apx/models and re-run setup."; exit 1; }

command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed for overlay videos)"
python3 -c "import onnxruntime as o; print('APX setup OK  onnxruntime', o.__version__)"
echo "Scripts: $R   (commit $(git -C "$R" rev-parse --short HEAD))"
ls -la "$M"
