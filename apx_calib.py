"""APX video analysis, step 2: distance calibration helper.

  python3 apx_calib.py grid --work WORKDIR
      Writes calib_first.jpg, calib_mid.jpg, calib_last.jpg: frames with a labelled pixel grid.
      Look at them and read the pixel position (x, y) of the BASE of the start and finish cones/gates.

  python3 apx_calib.py convert --work WORKDIR --frame K --x X --y Y
      Converts a point read on frame K into reference (frame 0) coordinates, which correct for
      any camera pan. Use the converted x values for --start-x / --finish-x in apx_accel.py.
      If the camera was locked (no pan), frame-0 values can be used directly.
"""
import argparse, json, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['grid', 'convert'])
ap.add_argument('--work', required=True); ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--x', type=float); ap.add_argument('--y', type=float)
a = ap.parse_args()
S = np.array(json.load(open(f'{a.work}/stab.json')))
if a.mode == 'convert':
    p = S[a.frame] @ [a.x, a.y, 1]
    print(f'reference coords: x={p[0]:.1f} y={p[1]:.1f}')
    raise SystemExit
cap = cv2.VideoCapture(f'{a.work}/work.mp4'); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
N = len(frames); H, W = frames[0].shape[:2]
step = 50 if max(W, H) <= 1280 else 100
for name, k in (('first', 0), ('mid', N // 2), ('last', N - 1)):
    f = frames[k].copy()
    for x in range(0, W, step):
        cv2.line(f, (x, 0), (x, H), (0, 255, 204) if x % (step * 2) == 0 else (0, 140, 110), 1)
        if x % (step * 2) == 0:
            cv2.putText(f, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * step / 50, (255, 255, 255), 2)
    for y in range(0, H, step):
        cv2.line(f, (0, y), (W, y), (0, 255, 204) if y % (step * 2) == 0 else (0, 140, 110), 1)
        if y % (step * 2) == 0:
            cv2.putText(f, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * step / 50, (255, 255, 255), 2)
    cv2.putText(f, f'frame {k}', (W - 200, H - 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f'{a.work}/calib_{name}.jpg', f)
    print(f'calib_{name}.jpg = frame {k}')
