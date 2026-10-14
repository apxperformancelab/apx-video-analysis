#!/usr/bin/env bash
# APX video analysis setup for a fresh chat (about 1-3 minutes).
#   curl -sSL https://raw.githubusercontent.com/apxperformancelab/apx-video-analysis/main/setup.sh | bash
# Downloads the scripts to ~/apx/scripts and the two AI models to ~/apx/models. No torch needed.
set -e
REPO=https://github.com/apxperformancelab/apx-video-analysis
S=$HOME/apx/scripts; M=$HOME/apx/models
mkdir -p "$M"
if [ -d "$S/.git" ]; then git -C "$S" pull -q; else rm -rf "$S"; git clone -q --depth 1 "$REPO" "$S"; fi
python3 -c "import onnxruntime, cv2, pandas, scipy, openpyxl, PIL" 2>/dev/null || \
  pip install --break-system-packages -q --progress-bar off onnxruntime opencv-python-headless pandas scipy openpyxl pillow
RAW=https://raw.githubusercontent.com/apxperformancelab/apx-video-analysis/models
[ -s "$M/yolo11m.onnx" ] || curl -sSL --fail -o "$M/yolo11m.onnx" "$RAW/yolo11m.onnx"
if [ ! -s "$M/yolo11x-pose.onnx" ]; then
  for p in 0 1 2; do curl -sSL --fail -o "$M/pose.part$p" "$RAW/yolo11x-pose.onnx.part$p"; done
  cat "$M"/pose.part0 "$M"/pose.part1 "$M"/pose.part2 > "$M/yolo11x-pose.onnx"; rm -f "$M"/pose.part*
fi
curl -sSL --fail -o "$M/SHA256SUMS" "$RAW/SHA256SUMS"
(cd "$M" && sed 's#/home/claude/onnx/##' SHA256SUMS | sha256sum -c --quiet) || { echo "ERROR: model download corrupted. Delete ~/apx/models and re-run setup."; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed for overlay videos)"
python3 -c "import onnxruntime as o; print('APX setup OK  onnxruntime', o.__version__)"
ls -la "$M"
