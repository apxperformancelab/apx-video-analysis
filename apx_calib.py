"""APX video analysis, step 2: distance calibration helper (and frame-rate fix).

  python3 apx_calib.py grid --work WORKDIR
      Writes calib_first.jpg, calib_mid.jpg, calib_last.jpg: frames with a labelled pixel grid.
      Look at them and read the pixel position (x, y) of the BASE of the start and finish cones/gates.

  python3 apx_calib.py crop --work WORKDIR --frame K --x X --y Y [--size 160] [--scale 4]
      Writes calib_crop_fK_xX_yY.jpg: a magnified square around (X, Y) on frame K with a 5 px grid
      labelled in that frame's own pixel coordinates. Use it when a cone or gate base can't be read
      to about 5 px on the full grid images.

  python3 apx_calib.py convert --work WORKDIR --frame K --x X --y Y
      Converts a point read on frame K into reference (frame 0) coordinates, which correct for
      any camera pan. Use the converted x values for --start-x / --finish-x in apx_accel.py.
      If the camera was locked (no pan), frame-0 values can be used directly.

  python3 apx_calib.py fps --work WORKDIR --fps F
      Frame-rate fix after a FAILED frame-rate check in apx_accel.py: sets "fps" in kp.json and
      "real_fps" (plus capture_fps and real_duration_s) in meta.json to F, then re-run apx_accel.py.
      No re-tracking needed. F is the real rate of the frames in work.mp4 (capture rate / stride).
"""
import argparse, json, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['grid', 'crop', 'convert', 'fps'])
ap.add_argument('--work', required=True); ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--x', type=float); ap.add_argument('--y', type=float)
ap.add_argument('--size', type=int, default=160, help='crop: side of the square in source pixels')
ap.add_argument('--scale', type=int, default=4, help='crop: magnification')
ap.add_argument('--fps', type=float, help='fps: real frame rate of work.mp4')
a = ap.parse_args()

if a.mode == 'convert':
    if a.x is None or a.y is None: raise SystemExit('convert needs --x and --y')
    S = np.array(json.load(open(f'{a.work}/stab.json')))
    p = S[a.frame] @ [a.x, a.y, 1]
    print(f'reference coords: x={p[0]:.1f} y={p[1]:.1f}')
    raise SystemExit

if a.mode == 'fps':
    if not a.fps or a.fps <= 0: raise SystemExit('fps needs --fps, e.g. --fps 120')
    kp = json.load(open(f'{a.work}/kp.json')); meta = json.load(open(f'{a.work}/meta.json'))
    old = kp.get('fps'); stride = meta.get('stride', 1)
    kp['fps'] = a.fps
    meta['real_fps'] = a.fps; meta['capture_fps'] = a.fps * stride
    meta['real_duration_s'] = round(meta.get('frames_used', len(kp['frames'])) / a.fps, 3)
    json.dump(kp, open(f'{a.work}/kp.json', 'w')); json.dump(meta, open(f'{a.work}/meta.json', 'w'), indent=1)
    print(f'fps {old:g} -> {a.fps:g} (capture {a.fps * stride:g} fps at stride {stride}, clip = {meta["real_duration_s"]} s real time). Re-run apx_accel.py.')
    raise SystemExit

cap = cv2.VideoCapture(f'{a.work}/work.mp4')

if a.mode == 'crop':
    if a.x is None or a.y is None: raise SystemExit('crop needs --x and --y')
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame); ok, f = cap.read()
    if not ok: raise SystemExit(f'could not read frame {a.frame}')
    H, W = f.shape[:2]; h = a.size // 2
    x0, y0 = int(max(0, min(W - a.size, round(a.x) - h))), int(max(0, min(H - a.size, round(a.y) - h)))
    c = cv2.resize(f[y0:y0 + a.size, x0:x0 + a.size], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
    ch, cw = c.shape[:2]
    def label(t, org, sz=0.45):
        cv2.putText(c, t, org, cv2.FONT_HERSHEY_SIMPLEX, sz, (0, 0, 0), 3)
        cv2.putText(c, t, org, cv2.FONT_HERSHEY_SIMPLEX, sz, (255, 255, 255), 1)
    for x in range((x0 + 4) // 5 * 5, x0 + a.size, 5):
        X = (x - x0) * a.scale; major = x % 20 == 0
        cv2.line(c, (X, 0), (X, ch), (0, 255, 204) if major else (0, 110, 90), 1)
        if major: label(str(x), (X + 2, 14))
    for y in range((y0 + 4) // 5 * 5, y0 + a.size, 5):
        Y = (y - y0) * a.scale; major = y % 20 == 0
        cv2.line(c, (0, Y), (cw, Y), (0, 255, 204) if major else (0, 110, 90), 1)
        if major: label(str(y), (2, Y - 3))
    cx, cy = int((a.x - x0) * a.scale), int((a.y - y0) * a.scale)
    cv2.drawMarker(c, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 18, 3)
    cv2.drawMarker(c, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 18, 1)
    label(f'frame {a.frame}  x {x0}-{x0 + a.size}  y {y0}-{y0 + a.size}', (6, ch - 8), 0.5)
    name = f'calib_crop_f{a.frame}_x{int(a.x)}_y{int(a.y)}.jpg'
    cv2.imwrite(f'{a.work}/{name}', c)
    print(f'{name} = frame {a.frame}, source x {x0}-{x0 + a.size}, y {y0}-{y0 + a.size}, cross at ({a.x:g}, {a.y:g})')
    raise SystemExit

frames = []
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
