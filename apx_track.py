"""APX video analysis, step 1: prepare clip, track athlete, estimate camera pan. (onnxruntime, no torch)

Usage:
  python3 apx_track.py --video CLIP.mp4 --out WORKDIR --capture-fps 240 --view side [--stride 2] [--gate-time 2.38 --gate-dist 10]

--capture-fps : the frame rate the phone RECORDED at (240 or 120 slo-mo). Each frame in the file is treated as
                1/capture-fps seconds of real time (true for a slo-mo clip with the speed ramp removed).
--stride      : process every Nth frame (2 for 240 fps clips halves processing time).
--view        : side | front | wide. Controls how the athlete is picked if several people are in shot.
--gate-time   : optional. Printed next to the clip's real-time length as an early frame-rate sanity check.

Writes WORKDIR/work.mp4, meta.json, kp.json, stab.json, check_tracking.jpg
"""
import argparse, json, os, sys, time, warnings, cv2, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apx.onnx_yolo import OnnxYolo

ap = argparse.ArgumentParser()
ap.add_argument('--video', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--capture-fps', type=float, required=True); ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--view', choices=['side', 'front', 'wide'], default='side')
ap.add_argument('--gate-time', type=float); ap.add_argument('--gate-dist', type=float, default=10)
ap.add_argument('--det-every', type=int, default=3, help='run person detection every Nth processed frame')
ap.add_argument('--models', default=os.path.expanduser('~/apx/models'))
ap.add_argument('--reuse-detections', action='store_true', help='reuse WORKDIR/detections.json from an earlier run (skips the slow detection pass)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); t0 = time.time()

cap = cv2.VideoCapture(a.video); file_fps = cap.get(cv2.CAP_PROP_FPS); frames, i = [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % a.stride == 0: frames.append(f)
    i += 1
if not frames: raise SystemExit('Could not read any frames from the clip.')
N = len(frames); H, W = frames[0].shape[:2]; real_fps = a.capture_fps / a.stride
meta = {'video': os.path.basename(a.video), 'file_fps': file_fps, 'file_frames': i, 'capture_fps': a.capture_fps,
        'stride': a.stride, 'real_fps': real_fps, 'frames_used': N, 'W': W, 'H': H, 'view': a.view,
        'real_duration_s': round(N / real_fps, 3)}
vw = cv2.VideoWriter(f'{a.out}/work.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
for f in frames: vw.write(f)
vw.release()
print(f'frames used {N} of {i}  ({W}x{H}, file {file_fps:.0f} fps, treated as {real_fps:.0f} fps real, clip = {N/real_fps:.2f} s real time)', flush=True)
if a.gate_time:
    print(f'gate time {a.gate_time:.2f} s for {a.gate_dist:g} m. If the clip covers the whole run but is SHORTER than the gate time, '
          f'--capture-fps is probably too high (try 120).', flush=True)
if max(W, H) < 1280:
    print(f'WARNING: low resolution clip ({W}x{H}). Ask for the original file (Frame.io download original).', flush=True)

# camera pan -> transforms into frame-0 coordinates
mats, M_cum, prev = [], np.eye(3), None
for f in frames:
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        p0 = cv2.goodFeaturesToTrack(prev, 800, 0.01, 8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
        M, _ = cv2.estimateAffinePartial2D(p1[st == 1], p0[st == 1], method=cv2.RANSAC, ransacReprojThreshold=2)
        if M is None: M = np.array([[1, 0, 0], [0, 1, 0]], float)
        M_cum = M_cum @ np.vstack([M, [0, 0, 1]])
    mats.append(M_cum.tolist()); prev = g
json.dump(mats, open(f'{a.out}/stab.json', 'w'))
pan = [m[0][2] for m in mats]; meta['camera_pan_px'] = round(max(pan) - min(pan), 1)
print(f'camera pan range {meta["camera_pan_px"]} px' + ('  (camera moved: tripod was not locked)' if meta['camera_pan_px'] > 15 else ''), flush=True)

# stage 1: detect people every Nth frame, link into tracks, pick the athlete
det = OnnxYolo(f'{a.models}/yolo11m.onnx'); pose = OnnxYolo(f'{a.models}/yolo11x-pose.onnx')
def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); inter = ix * iy
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter + 1e-9)
dframes = list(range(0, N, a.det_every)) + ([N - 1] if (N - 1) % a.det_every else [])
tracks = []  # list of dict frame->box
all_dets = {}
prev_dets = json.load(open(f'{a.out}/detections.json')) if a.reuse_detections and os.path.exists(f'{a.out}/detections.json') else None
for n, k in enumerate(dframes):
    if prev_dets is not None and str(k) in prev_dets: boxes = np.array(prev_dets[str(k)]).reshape(-1, 4)
    else: boxes, _ = det.persons(frames[k])
    all_dets[k] = boxes.tolist()
    used = set()
    for b in boxes:
        best, bs = None, 0
        for ti, t in enumerate(tracks):
            if ti in used: continue
            lk = max(t);
            if k - lk > 6 * a.det_every: continue
            lb = t[lk]; c1 = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2]); c2 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            s = iou(lb, b) + (0.5 if np.linalg.norm(c1 - c2) < 0.6 * (lb[3] - lb[1]) else 0)
            if s > bs: best, bs = ti, s
        if best is not None and bs > 0.3:
            tracks[best][k] = b.tolist(); used.add(best)
        else:
            tracks.append({k: b.tolist()}); used.add(len(tracks) - 1)
    if n % 20 == 0: print(f'  detect {k}/{N}', flush=True)
if not tracks: raise SystemExit('No person detected. Check the clip.')
Sm = np.array(mats)
def cen_ref(k, b):  # box centre in frame-0 (pan-corrected) coordinates
    c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0]); return (Sm[k] @ c)[:2]
# merge fragments of the same person (detector drop-outs, occlusion, camera pan)
merged = True
while merged:
    merged = False
    tracks.sort(key=lambda t: min(t))
    for ia in range(len(tracks)):
        A = tracks[ia]; ea = max(A); ca = cen_ref(ea, A[ea]); ha = A[ea][3] - A[ea][1]
        best, bd = None, 1e9
        for ib in range(len(tracks)):
            if ib == ia: continue
            B = tracks[ib]; sb = min(B)
            if not (-4 * a.det_every <= sb - ea <= 12 * a.det_every) or max(B) <= ea: continue
            if sb <= ea:   # small overlap: compare at the same frame
                kk = max(k_ for k_ in B if k_ <= ea) if any(k_ <= ea for k_ in B) else sb
                ka = max(k_ for k_ in A if k_ <= kk)
                dd = np.linalg.norm(cen_ref(kk, B[kk]) - cen_ref(ka, A[ka]))
            else:
                dd = np.linalg.norm(cen_ref(sb, B[sb]) - ca)
            if dd < 1.5 * ha * (1 + (sb - ea) / (6 * a.det_every)) and dd < bd: best, bd = ib, dd
        if best is not None:
            for k_, b_ in tracks[best].items():
                if k_ > ea: A[k_] = b_
            del tracks[best]; merged = True; break
def score(tb):
    ks_ = sorted(tb); cr = np.array([cen_ref(k, tb[k]) for k in ks_]); hs = [tb[k][3] - tb[k][1] for k in ks_]
    if a.view in ('side', 'wide'): return len(tb) * 0.2 * a.det_every + (cr[:, 0].max() - cr[:, 0].min())
    return len(tb) * 0.5 * a.det_every + (max(hs) - min(hs)) * 3
tracks.sort(key=lambda t: -score(t))
ath = dict(tracks[0])
# fill detection gaps from other tracks close to the athlete
for k in dframes:
    if k in ath: continue
    prevk = [x for x in ath if x < k]
    if not prevk: continue
    lb = ath[max(prevk)]; lc = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2]); best, bd = None, 1e9
    for t in tracks[1:]:
        if k in t:
            b = t[k]; dd = np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - lc)
            if dd < bd: best, bd = b, dd
    if best is not None and bd < (lb[3] - lb[1]): ath[k] = best
# interpolate boxes for every processed frame between detections
ks = np.array(sorted(ath)); B = np.array([ath[k] for k in ks])
box = {}
for k in range(ks[0], ks[-1] + 1):
    box[k] = [float(np.interp(k, ks, B[:, c])) for c in range(4)]
meta['athlete_track_frames'] = len(box)
meta['athlete_height_px_median'] = round(float(np.median([b[3] - b[1] for b in box.values()])), 1)
print(f'athlete tracked in {len(box)}/{N} frames, median height {meta["athlete_height_px_median"]} px', flush=True)

# stage 2: pose on an upscaled crop around the athlete (crop follows the previous pose box when available)
res, prev_pb = [], None
for k, f in enumerate(frames):
    rec = {'frame': k, 'kp': None, 'box': box.get(k)}
    if k in box:
        x1, y1, x2, y2 = box[k]
        if prev_pb is not None and iou(prev_pb, box[k]) > 0.3:  # blend in last pose box: tighter, follows limbs
            x1, y1, x2, y2 = [0.5 * u + 0.5 * v for u, v in zip(box[k], prev_pb)]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 1.5
        X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)); X2, Y2 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
        crop = f[Y1:Y2, X1:X2]; sc = 640 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        pb, ps, pk = pose.pose(crop)
        if len(ps):
            pb = pb / sc + [X1, Y1, X1, Y1]
            j = int(np.argmax([iou(bb, box[k]) + 0.1 * s_ for bb, s_ in zip(pb, ps)]))
            xy = pk[j, :, :2] / sc + [X1, Y1]; cf = pk[j, :, 2]
            rec['kp'] = np.c_[xy, cf].round(2).tolist(); prev_pb = pb[j].tolist()
    res.append(rec)
    if k % 50 == 0: print(f'  pose {k}/{N}', flush=True)
json.dump({'fps': real_fps, 'W': W, 'H': H, 'frames': res}, open(f'{a.out}/kp.json', 'w'))
json.dump({str(k): v for k, v in all_dets.items()}, open(f'{a.out}/detections.json', 'w'))

picks = np.linspace(0, N - 1, 8).astype(int); tiles = []
for k in picks:
    f = frames[k].copy()
    if k in box:
        x1, y1, x2, y2 = map(int, box[k]); cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 204), 3)
    cv2.putText(f, f'frame {k}', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
    tiles.append(cv2.resize(f, (480, int(480 * H / W))))
cv2.imwrite(f'{a.out}/check_tracking.jpg', np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
meta['processing_s'] = round(time.time() - t0)
json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
print(f'DONE in {meta["processing_s"]} s ({meta["processing_s"]/N:.2f} s per frame)', flush=True)
