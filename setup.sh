#!/usr/bin/env bash
# APX video analysis setup for a fresh chat (about 1 minute).
# Run it from the attached clone of this repo. Nothing is downloaded and piped into a shell:
#   bash /home/claude/apx-video-analysis/setup.sh
# Points ~/apx/scripts at this clone, fetches the two AI models from the `models` branch with git,
# checks their SHA256 sums and installs the Python packages if missing. No torch needed.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APX="$HOME/apx"; S="$APX/scripts"; M="$APX/models"; MSRC="$APX/models-src"
mkdir -p "$APX" "$M" "$APX/jobs"

# Scripts: ~/apx/scripts is a symlink to this clone, so every command runs the local copy.
if [ -e "$S" ] && [ ! -L "$S" ]; then
  echo "ERROR: $S exists and is not a symlink. Move it aside and re-run setup."; exit 1
fi
ln -sfn "$REPO_DIR" "$S"

python3 -c "import onnxruntime, cv2, pandas, scipy, openpyxl, PIL" 2>/dev/null || \
  pip install --break-system-packages -q --progress-bar off onnxruntime opencv-python-headless pandas scipy openpyxl pillow

# Models: shallow clone of the models branch only, then assemble the split pose model.
if [ ! -s "$M/yolo11m.onnx" ] || [ ! -s "$M/yolo11x-pose.onnx" ] || [ ! -s "$M/SHA256SUMS" ]; then
  REMOTE="$(git -C "$REPO_DIR" remote get-url origin)"
  rm -rf "$MSRC"
  git clone -q --depth 1 --branch models --single-branch "$REMOTE" "$MSRC"
  cp "$MSRC/yolo11m.onnx" "$MSRC/SHA256SUMS" "$M/"
  cat "$MSRC/yolo11x-pose.onnx.part0" "$MSRC/yolo11x-pose.onnx.part1" "$MSRC/yolo11x-pose.onnx.part2" > "$M/yolo11x-pose.onnx"
  rm -rf "$MSRC"
fi
(cd "$M" && sed 's#/home/claude/onnx/##' SHA256SUMS | sha256sum -c --quiet) || {
  echo "ERROR: model files corrupted. Delete ~/apx/models and re-run setup."; exit 1; }

command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed for overlay videos)"
python3 -c "import onnxruntime as o; print('APX setup OK  onnxruntime', o.__version__)"
echo "scripts: $S -> $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
ls -la "$M"
