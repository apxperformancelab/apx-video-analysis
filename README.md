# APX Performance Lab: video analysis pipeline

Pose tracking and sprint / change-of-direction analysis used by the APX Video Analysis projects.
Runs on onnxruntime (CPU), no torch. Models: YOLO11 detection + pose (Ultralytics, AGPL-3.0), exported to ONNX, stored on the `models` branch.

Setup in a fresh environment (clone first, then run the local copy; never pipe a URL into bash):

    git clone --depth 1 https://github.com/apxperformancelab/apx-video-analysis /home/claude/apx-video-analysis
    bash /home/claude/apx-video-analysis/setup.sh

Scripts run from the clone. Models are fetched from the `models` branch with git and checked against SHA256SUMS.
`.claude/settings.json` allows only the pipeline commands; anything else goes through the normal permission checks.

Scripts (run in order): apx_track.py -> apx_calib.py -> the test script (apx_accel.py for the 10m, apx_maxv.py for the 30m, apx_505.py for the 505) -> apx_render.py. Each has usage notes at the top.
No athlete footage or data is stored in this repository.
