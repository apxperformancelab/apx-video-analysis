# APX Performance Lab: video analysis pipeline

Pose tracking and sprint / change-of-direction analysis used by the APX Video Analysis projects.
Runs on onnxruntime (CPU), no torch. Models: YOLO11 detection + pose (Ultralytics, AGPL-3.0), exported to ONNX, published as release assets.

Setup in a fresh environment:

    curl -sSL https://raw.githubusercontent.com/apxperformancelab/apx-video-analysis/main/setup.sh | bash

Scripts (run in order): apx_track.py -> apx_calib.py -> apx_accel.py -> apx_render.py. Each has usage notes at the top.
No athlete footage or data is stored in this repository.
