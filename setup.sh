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
for m in yolo11x-pose.onnx yolo11m.onnx; do
  [ -s "$M/$m" ] || curl -sSL --fail -o "$M/$m" "$REPO/releases/download/models-v1/$m"
done
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (needed for overlay videos)"
python3 -c "import onnxruntime as o; print('APX setup OK  onnxruntime', o.__version__)"
ls -la "$M"
