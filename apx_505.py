"""APX video analysis: 505 change of direction (one turn direction per run).

Works on imperfect footage: distances are in the athlete's own leg lengths (hip to ankle), so a panning camera
or missing cones do not break it. Run once per clip.

Side-on clip:
  python3 apx_505.py --work WORKDIR --view side --plant-leg L [--gate-frames IN OUT] [--gate-time T] [--tenm-time T10]
Front-on clip (camera behind the turn line, athlete runs towards it):
  python3 apx_505.py --work WORKDIR --view front --plant-leg L

--plant-leg   : the leg the athlete plants on to turn (L or R), as filmed. Side-on video cannot tell left from right,
                so this labels the plant step; on a front clip it is checked against the pose model.
--gate-frames : frames where the hip crosses the timing line (5m from the turn line) going in and coming back:
                gives a video 505 time (backup only; gate time is the benchmark).
--gate-time   : the 505 gate time; with --gate-frames it checks the frame rate.
--tenm-time   : the athlete's best 10m time, for the COD deficit (505 time minus 10m time).
--slow-from/--slow-to : force the slow-motion window if the automatic check is wrong.

Outputs WORKDIR/analysis/: per_frame.csv, steps.csv (one row per foot contact, with its role), summary.json, K_img.npy
"""
import argparse, json, os, warnings, numpy as np, pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, uniform_filter1d
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--view', choices=['side', 'front'], required=True)
ap.add_argument('--plant-leg', choices=['L', 'R'], required=True)
ap.add_argument('--gate-frames', type=int, nargs=2); ap.add_argument('--gate-time', type=float); ap.add_argument('--tenm-time', type=float)
ap.add_argument('--slow-from', type=int); ap.add_argument('--slow-to', type=int)
ap.add_argument('--stance-speed', type=float, default=None, help='ankle speed (leg lengths/s) below which a foot counts as planted')
a = ap.parse_args()
OUT = f'{a.work}/analysis'; os.makedirs(OUT, exist_ok=True)
d = json.load(open(f'{a.work}/kp.json')); S = np.array(json.load(open(f'{a.work}/stab.json'))); meta = json.load(open(f'{a.work}/meta.json'))
FPS = d['fps']; N = len(d['frames']); W, H = meta['W'], meta['H']; FPS_RT = meta.get('file_fps', 30) or 30
odd = lambda x: int(x) // 2 * 2 + 1
notes = []
def note(s): notes.append(s); print('NOTE:', s)

# ---------------- keypoints: pan-corrected, cleaned, smoothed ----------------
K = np.full((N, 17, 3), np.nan)
for r in d['frames']:
    if r['kp'] is not None: K[r['frame']] = np.array(r['kp'])
valid = ~np.isnan(K[:, 0, 0]) & (np.nan_to_num(np.nanmean(K[:, :, 2], 1)) > 0.5)
K[~valid] = np.nan
idx = np.where(valid)[0]
if len(idx) < 0.3 * FPS: raise SystemExit('Athlete tracked in too few frames to analyse.')
P = np.full((N, 17, 2), np.nan)
for i in idx: P[i] = (S[i] @ np.c_[K[i, :, :2], np.ones(17)].T).T[:, :2]
def seglen(p, q): return np.linalg.norm(P[:, p] - P[:, q], axis=1)
legL = np.nanmax(np.c_[seglen(11, 13) + seglen(13, 15), seglen(12, 14) + seglen(14, 16)], 1)
LPX = uniform_filter1d(median_filter(np.interp(np.arange(N), idx, legL[idx]), odd(0.5 * FPS), mode='nearest'), odd(0.25 * FPS), mode='nearest')
nswaps = 0
for pl in ([(13, 14), (15, 16)], [(7, 8), (9, 10)]):
    prev = None
    for i in idx:
        if prev is not None and i - prev <= 2:
            h0, h1 = P[prev, [11, 12]].mean(0), P[i, [11, 12]].mean(0)
            keep = sum(np.linalg.norm((P[i, x] - h1) - (P[prev, x] - h0)) + np.linalg.norm((P[i, y] - h1) - (P[prev, y] - h0)) for x, y in pl)
            swap = sum(np.linalg.norm((P[i, y] - h1) - (P[prev, x] - h0)) + np.linalg.norm((P[i, x] - h1) - (P[prev, y] - h0)) for x, y in pl)
            if swap < keep * 0.5:
                for x, y in pl: P[i, [x, y]] = P[i, [y, x]]
                nswaps += 1
        prev = i
WH = max(5, odd(round(0.06 * FPS)))
for j in range(17):
    for c in range(2):
        s = np.interp(np.arange(N), idx, P[idx, j, c]); med = median_filter(s, WH, mode='nearest')
        mad = median_filter(np.abs(s - med), WH, mode='nearest') * 1.4826
        s[np.abs(s - med) > np.maximum(3 * mad, 0.06 * LPX)] = np.nan; ok = ~np.isnan(s)
        P[:, j, c] = np.interp(np.arange(N), np.where(ok)[0], s[ok])
Ps = savgol_filter(P, max(5, odd(round(0.06 * FPS))), 2, axis=0)
hip = Ps[:, [11, 12]].mean(1); sho = Ps[:, [5, 6]].mean(1)
F0, F1 = int(idx[0]), int(idx[-1])
ath_px = float(np.nanmedian(np.nanmax(K[:, :, 1], 1) - np.nanmin(K[:, :, 1], 1)))
PANNED = (meta.get('camera_pan_px') or 0) > 0.15 * W

# ---------------- slow-motion window (iPhone ramps show as an abrupt jump in movement per frame) ----------------
mv = np.linalg.norm(np.diff(hip, axis=0, prepend=hip[:1]), axis=1) / LPX
loc = median_filter(mv, 5, mode='nearest')
SLOW_FROM, SLOW_TO = (a.slow_from if a.slow_from is not None else F0), (a.slow_to if a.slow_to is not None else F1)
w_ = max(4, int(0.03 * FPS)); gap_ = max(4, int(0.05 * FPS))     # iPhone ramps take a few frames
if a.slow_from is None:
    for i in range(F0 + w_, min(F0 + int(0.5 * FPS), F1 - w_ - gap_)):
        before, after = np.median(loc[i - w_:i]), np.median(loc[i + gap_:i + gap_ + w_])
        if before > 2.5 * after and before > 0.05:
            SLOW_FROM = i + gap_; note(f'Slow motion starts at about frame {SLOW_FROM}; earlier frames are real time (timing not used there).'); break
if a.slow_to is None:
    for i in range(max(F1 - int(0.6 * FPS), F0 + w_), F1 - w_ - gap_):
        before, after = np.median(loc[i - w_:i]), np.median(loc[i + gap_:i + gap_ + w_])
        if after > 2.5 * before and after > 0.05:
            SLOW_TO = i; note(f'Slow motion ends at about frame {SLOW_TO}; later frames are real time (timing not used there).'); break

def tsec(f0, f1):
    """real seconds between two frames, allowing for real-time sections outside the slow window"""
    t = 0.0
    for f in range(int(f0), int(f1)):
        t += 1 / FPS if SLOW_FROM <= f < SLOW_TO else 1 / FPS_RT
    return t

# ---------------- turn point and travel direction ----------------
if a.view == 'side':
    along = hip[:, 0]
else:   # front: athlete approaches the camera, so "along" = down the image (ankles lowest = closest)
    along = np.nanmax(Ps[:, [15, 16], 1], 1)
along_s = savgol_filter(np.interp(np.arange(N), idx, along[idx]), max(7, odd(0.1 * FPS)), 2)
d_in = 1 if along_s[F0 + int(0.5 * (F1 - F0))] > along_s[F0] else -1
TF = int(np.argmax(along_s[F0:F1 + 1] * d_in) + F0)
turn_in_clip = (TF - F0) > 0.1 * FPS and (F1 - TF) > 0.1 * FPS
if not turn_in_clip: note('The turn point is at the edge of the clip: the turn may not be fully filmed.')
v_along = np.gradient(along_s) * FPS / LPX * d_in          # leg lengths per second, + towards the turn line
pk_in = float(np.nanpercentile(v_along[F0 + int(0.06 * FPS):TF], 90)) if TF > F0 + int(0.15 * FPS) else np.nan

# ---------------- foot contacts: a foot is planted while its ankle stays within a small box for a short window ----------------
# (pooled over both ankles, so left/right label swaps by the pose model do not break contacts)
from scipy.ndimage import maximum_filter1d
WIN = max(5, odd(0.05 * FPS)); HW = WIN // 2
def xy_range(j):
    rx = maximum_filter1d(Ps[:, j, 0], WIN) + maximum_filter1d(-Ps[:, j, 0], WIN)
    ry = maximum_filter1d(Ps[:, j, 1], WIN) + maximum_filter1d(-Ps[:, j, 1], WIN)
    return np.hypot(rx, ry) / LPX
thr = a.stance_speed or 0.15        # max ankle movement over the window, in leg lengths
planted = np.stack([xy_range(15) < thr, xy_range(16) < thr], 1)
open_, contacts = [], []
for f in range(F0, F1 + 1):
    pts = [(Ps[f, 15 + j], j) for j in (0, 1) if planted[f, j]]
    for c in open_: c['hit'] = False
    for p, j in pts:
        cand = [c for c in open_ if not c['hit'] and np.linalg.norm(c['xy'] - p) < 0.18 * LPX[f]]
        if cand:
            best = min(cand, key=lambda c: np.linalg.norm(c['xy'] - p)); best['frames'].append(f); best['hit'] = True; best['js'].append(j)
        elif not any(np.linalg.norm(c['xy'] - p) < 0.18 * LPX[f] for c in open_):
            open_.append({'xy': p.copy(), 'frames': [f], 'js': [j], 'hit': True})
    still_open = []
    for c in open_:
        if f - c['frames'][-1] > max(2, int(0.02 * FPS)): contacts.append(c)
        else: still_open.append(c)
    open_ = still_open
contacts += open_
for c in contacts:     # the window shrinks a contact at both ends: add it back
    c['frames'] = list(range(max(F0, c['frames'][0] - HW), min(F1, c['frames'][-1] + HW) + 1))
MINC = max(3, int(0.03 * FPS))
contacts = sorted([c for c in contacts if len(c['frames']) >= MINC], key=lambda c: c['frames'][0])
merged = []
for c in contacts:
    if merged and np.linalg.norm(merged[-1]['xy'] - c['xy']) < 0.15 * LPX[c['frames'][0]] and c['frames'][0] - merged[-1]['frames'][-1] <= int(0.05 * FPS):
        merged[-1]['frames'] = list(range(merged[-1]['frames'][0], max(merged[-1]['frames'][-1], c['frames'][-1]) + 1)); merged[-1]['js'] += c['js']
    else: merged.append(c)
contacts = merged

# ---------------- angles ----------------
dir_at = lambda f: d_in if f <= TF else -d_in        # travel direction (image x on side view)
def up_angle(top, bot, sgn): v = top - bot; return np.degrees(np.arctan2(v[0] * sgn, -v[1]))
def joint(p, q, r):
    v1, v2 = p - q, r - q; return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))
SIDE_J = {'L': (11, 13, 15), 'R': (12, 14, 16)}
df = pd.DataFrame({'frame': np.arange(N), 't_s': np.arange(N) / FPS, 'hip_x_m': np.nan, 'leg_px': LPX, 'valid': valid,
                   'speed_legs_per_s': v_along})
df['trunk_lean_deg'] = [up_angle(sho[i], hip[i], dir_at(i)) if a.view == 'side' else np.degrees(np.arctan2(sho[i, 0] - hip[i, 0], hip[i, 1] - sho[i, 1])) for i in range(N)]
for s_, (hp, kn, an) in SIDE_J.items():
    df[f'knee_{s_}'] = [joint(Ps[i, hp], Ps[i, kn], Ps[i, an]) for i in range(N)]
    df[f'shin_{s_}'] = [up_angle(Ps[i, kn], Ps[i, an], dir_at(i)) for i in range(N)]
df['hip_width_pctL'] = np.abs(Ps[:, 11, 0] - Ps[:, 12, 0]) / LPX * 100      # front view: wide = facing the camera
df['phase'] = ''

def stance_leg(c):
    fr = c['frames'][: max(2, len(c['frames']) // 3)]
    dl = np.mean([np.linalg.norm(Ps[f, 15] - c['xy']) for f in fr]); dr = np.mean([np.linalg.norm(Ps[f, 16] - c['xy']) for f in fr])
    return 'L' if dl <= dr else 'R'

# same foot reported twice (pivoting or a left/right label swap): overlapping contacts on the same tracked leg are one contact;
# away from the turn, two contacts overlapping most of their time are a double detection: keep the longer one
def ov(c1, c2): return max(0, min(c1['frames'][-1], c2['frames'][-1]) - max(c1['frames'][0], c2['frames'][0]) + 1)
changed = True
while changed:
    changed = False
    for i1 in range(len(contacts) - 1):
        for i2 in range(i1 + 1, min(i1 + 3, len(contacts))):
            c1, c2 = contacts[i1], contacts[i2]; o = ov(c1, c2); gap = c2['frames'][0] - c1['frames'][-1]
            same = stance_leg(c1) == stance_leg(c2) and np.linalg.norm(c1['xy'] - c2['xy']) < 0.5 * LPX[c2['frames'][0]] and gap <= max(2, int(0.03 * FPS))
            far_from_turn = min(abs(c1['frames'][0] - TF), abs(c2['frames'][0] - TF)) > 0.2 * FPS
            if same:
                c1['frames'] = list(range(c1['frames'][0], max(c1['frames'][-1], c2['frames'][-1]) + 1)); c1['js'] += c2['js']; del contacts[i2]; changed = True; break
            if i2 == i1 + 1 and o and far_from_turn and o > 0.5 * min(len(c1['frames']), len(c2['frames'])):
                del contacts[i1 + (1 if len(c1['frames']) >= len(c2['frames']) else 0)]; changed = True; break
        if changed: break

# ---------------- classify contacts ----------------
near = [k for k, c in enumerate(contacts) if c['frames'][0] <= TF <= c['frames'][-1]] or \
       [k for k, c in enumerate(contacts) if c['frames'][0] - int(0.06 * FPS) <= TF <= c['frames'][-1] + int(0.06 * FPS)]
if near:
    PL = max(near, key=lambda k: (contacts[k]['xy'][0] if a.view == 'side' else contacts[k]['xy'][1]) * d_in)
elif contacts:
    PL = int(np.argmin([min(abs(c['frames'][0] - TF), abs(c['frames'][-1] - TF)) for c in contacts])); note('No foot contact overlaps the turn point; the nearest contact is used as the plant.')
else:
    raise SystemExit('No foot contacts found. Check the tracking (check_tracking.jpg) or raise --stance-speed.')
ptd, pto = contacts[PL]['frames'][0], contacts[PL]['frames'][-1]
roles, seen_out = {}, False
for k, c in enumerate(contacts):
    td_ = c['frames'][0]
    if k == PL: roles[k] = 'plant'
    elif td_ < ptd: roles[k] = 'approach'
    elif td_ <= pto - int(0.03 * FPS): roles[k] = 'second foot (turn)'
    elif not seen_out: roles[k] = 'first step out'; seen_out = True
    else: roles[k] = 'exit'
before = [k for k in roles if roles[k] == 'approach']
if before: roles[before[-1]] = 'penultimate'
n_two = sum(1 for v in roles.values() if v == 'second foot (turn)')
if n_two: note('The other foot is also on the ground during the plant (two-foot stop or pivot): check on the video.')

def hip_height(f, gy): return (gy - hip[f, 1]) / LPX[f] * 100        # % leg length above the footprint
SAG_MAX, FRO_MIN = (18 if a.view == 'side' else 12), 25     # hip width in view (% leg): below = body side-on to this camera, above = facing / back to it
rows = []
for k, c in enumerate(contacts):
    td, to = c['frames'][0], c['frames'][-1]; gy = float(np.median([Ps[f, 15 + j, 1] for f, j in zip(c['frames'], c['js'])]))
    leg = stance_leg(c); hp, kn, an = SIDE_J[leg]
    stance = range(td, to + 1)
    knees = [df.at[f, f'knee_{leg}'] for f in stance]; hh = [hip_height(f, gy) for f in stance]
    fdeep = td + int(np.argmin(hh))      # deepest point = lowest hip
    facing = float(np.median(df.loc[td:to, 'hip_width_pctL']))
    sag, fro = facing < SAG_MAX, facing >= FRO_MIN
    row = {'contact': k + 1, 'role': roles[k], 'td_frame': td, 'to_frame': to, 'deepest_frame': fdeep,
           'leg_tracked': leg, 'timing_ok': bool(SLOW_FROM <= td and to <= SLOW_TO),
           'contact_ms': round(tsec(td, to + 1) * 1000) if SLOW_FROM <= td and to <= SLOW_TO else None,
           'hip_width_in_view_pctL': round(facing, 1),
           'body_to_camera': 'side-on' if sag else ('facing or back' if fro else 'angled'),
           'hip_height_TD_pctL': round(hip_height(td, gy), 1), 'hip_height_min_pctL': round(float(np.min(hh)), 1),
           'knee_flex_max': round(180 - float(np.min(knees)), 1) if sag else np.nan}
    sg = dir_at(td)
    row['foot_vs_hip_pctL'] = round(float((c['xy'][0] - hip[td, 0]) * sg / LPX[td] * 100), 1) if (a.view == 'side' and sag) else np.nan
    row['shin_TD'] = round(float(df.at[td, f'shin_{leg}']), 1) if (a.view == 'side' and sag) else np.nan
    row['trunk_TD'] = round(float(df.at[td, 'trunk_lean_deg']), 1) if (a.view == 'side' and sag) else np.nan
    row['trunk_deepest'] = round(float(df.at[fdeep, 'trunk_lean_deg']), 1) if (a.view == 'side' and sag) else np.nan
    if fro:
        # knee line in the image at the deepest point: knee offset from the hip-ankle line, + = towards the body midline (inward)
        hpt, knt, ant = Ps[fdeep, hp], Ps[fdeep, kn], Ps[fdeep, an]
        other_hip = Ps[fdeep, 12 if leg == 'L' else 11]
        u = (ant - hpt) / (np.linalg.norm(ant - hpt) + 1e-9); nrm = np.array([-u[1], u[0]])
        if np.dot(other_hip - hpt, nrm) < 0: nrm = -nrm
        off = float(np.dot(knt - hpt, nrm)); v1, v2 = knt - hpt, ant - knt
        fppa = float(np.degrees(np.arctan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), np.dot(v1, v2))))
        row['knee_inward_pctL'] = round(off / LPX[fdeep] * 100, 1)
        row['knee_line_deg'] = round(fppa * (1 if off > 0 else -1), 1)
        side_sign = 1 if (Ps[fdeep, an, 0] - hip[fdeep, 0]) > 0 else -1     # + = trunk leans over the stance foot
        row['trunk_side_lean_deg'] = round(float(np.degrees(np.arctan2((sho[fdeep, 0] - hip[fdeep, 0]) * side_sign, hip[fdeep, 1] - sho[fdeep, 1]))), 1)
        row['foot_width_pctL'] = round(float(abs(c['xy'][0] - hip[td, 0]) / LPX[td] * 100), 1)
    else:
        for c_ in ('knee_inward_pctL', 'knee_line_deg', 'trunk_side_lean_deg', 'foot_width_pctL'): row[c_] = np.nan
    rows.append(row)
st = pd.DataFrame(rows)
if a.view == 'front' and len(st):
    # straight at the camera: contacts far from the turn are small and the foot moves in depth, so they are not reliable
    keep = st.role.isin(['penultimate', 'plant', 'second foot (turn)', 'first step out'])
    if (~keep).any(): note(f'Front view: {int((~keep).sum())} contacts away from the turn left out (athlete running straight at the camera).')
    contacts = [c for c, k_ in zip(contacts, keep) if k_]; roles = {i: r_ for i, r_ in enumerate(st.role[keep])}
    st = st[keep].reset_index(drop=True); st['contact'] = np.arange(1, len(st) + 1)
# step length into each contact (leg lengths, along the image x on side view)
if a.view == 'side' and len(st):
    xs = np.array([c['xy'][0] for c in contacts]); st['step_length_legs'] = np.r_[np.nan, np.abs(np.diff(xs)) / LPX[st.td_frame.values[1:]]].round(2)
for k, c in enumerate(contacts):
    for f in range(c['frames'][0], c['frames'][-1] + 1):
        df.at[f, 'phase'] = roles[k]

# ---------------- turn summary ----------------
r = lambda role: st[st.role == role].iloc[0] if (st.role == role).any() else None
plant, pen, out1 = r('plant'), r('penultimate'), r('first step out')
appr = st[st.role.isin(['approach'])]
base_h = float(appr.hip_height_TD_pctL.head(3).median()) if (len(appr) and a.view == 'side') else None
leg_check = None
if a.view == 'front' and plant is not None:
    leg_check = 'matches' if plant.leg_tracked == a.plant_leg else f'pose model says {plant.leg_tracked}: check on the video (the model can mix up left and right when the athlete turns)'
turn = {
    'turn_frame': TF, 'plant_leg_given': a.plant_leg, 'plant_leg_check_front': leg_check,
    'approach_peak_speed_legs_per_s': round(pk_in, 2) if np.isfinite(pk_in) else None,
    'contacts_before_plant_in_clip': int((st.role.isin(['approach', 'penultimate'])).sum()),
}
if np.isfinite(pk_in) and a.view == 'side':
    on = [f for f in range(F0, TF) if v_along[f] >= 0.9 * pk_in]
    if on:
        dec0 = on[-1]; turn['decel_start_frame'] = dec0
        turn['braking_contacts'] = int(((st.td_frame >= dec0) & (st.td_frame <= TF)).sum())
        if SLOW_FROM <= dec0 and TF <= SLOW_TO: turn['braking_time_ms'] = round(tsec(dec0, TF) * 1000)
    back = [f for f in range(TF, F1 + 1) if -v_along[f] >= 0.5 * pk_in]
    slow_in = [f for f in range(F0, TF) if v_along[f] >= 0.5 * pk_in]
    if back and slow_in and SLOW_FROM <= slow_in[-1] and back[0] <= SLOW_TO:
        turn['time_below_half_speed_ms'] = round(tsec(slow_in[-1], back[0]) * 1000)
if plant is not None:
    turn['plant'] = {k_: plant[k_] for k_ in st.columns if k_ not in ('role',)}
    if base_h: turn['hip_drop_at_plant_pctL'] = round(base_h - float(plant.hip_height_min_pctL), 1)
if pen is not None:
    turn['penultimate'] = {k_: pen[k_] for k_ in st.columns if k_ not in ('role',)}
    if base_h: turn['hip_drop_at_penultimate_pctL'] = round(base_h - float(pen.hip_height_min_pctL), 1)
    if a.view == 'side' and len(appr) and 'step_length_legs' in st and pd.notna(pen.step_length_legs):
        prev_steps = appr.step_length_legs.dropna().tail(2)
        if len(prev_steps): turn['penultimate_step_vs_approach'] = round(float(pen.step_length_legs) / float(prev_steps.mean()), 2)
if out1 is not None and plant is not None:
    turn['first_step_out'] = {k_: out1[k_] for k_ in st.columns if k_ not in ('role',)}
    if a.view == 'side': turn['first_step_length_legs'] = out1.step_length_legs
    if SLOW_FROM <= plant['to_frame'] and out1['td_frame'] <= SLOW_TO: turn['plant_toeoff_to_first_step_ms'] = round(tsec(plant['to_frame'], out1['td_frame']) * 1000)

timing = None
if a.gate_frames:
    g0, g1 = a.gate_frames; vt = tsec(g0, g1)
    timing = {'video_505_s': round(vt, 3), 'note': 'video-timed hip crossing of the timing line: backup only, not interchangeable with gate times'}
    if a.gate_time:
        timing['gate_505_s'] = a.gate_time
        implied = (g1 - g0) / a.gate_time
        timing['implied_fps_if_all_slow'] = round(implied)
        if abs(vt - a.gate_time) / a.gate_time > 0.15:
            timing['warning'] = f'Video time {vt:.2f}s vs gate {a.gate_time:.2f}s: frame rate or crossing frames are wrong (all slow motion would mean about {implied:.0f} fps).'
            print('WARNING:', timing['warning'])
t505 = a.gate_time or (timing or {}).get('video_505_s')
cod_def = round(t505 - a.tenm_time, 2) if (t505 and a.tenm_time) else None

# ---------------- uncertainty ----------------
mv_med = float(np.nanmedian(mv[F0:F1]))
noise_pctL = round((1.5 * mv_med + 2.5 * 1.4 / float(np.median(LPX[F0:F1]))) * 100, 1)
noise_deg = (8 if ath_px < 150 else (6 if ath_px < 300 else 4)) + (1 if FPS < 200 else 0) + (2 if a.view == 'front' else 0)
summ = {
    'view': a.view, 'real_fps': FPS, 'camera_pan_px': meta.get('camera_pan_px'), 'panned_camera': PANNED,
    'athlete_height_px': round(ath_px), 'tracked_frames': [F0, F1], 'slow_window': [SLOW_FROM, SLOW_TO], 'notes': notes,
    'contact_detection_speed_legs_per_s': round(thr, 2), 'n_contacts': len(st), 'lr_swaps_fixed': nswaps,
    'turn': turn, 'video_timing': timing, 'cod_deficit_s': cod_def,
    'approach_hip_height_pctL': round(base_h, 1) if base_h else None,
    'uncertainty': {'single_contact_pctL': noise_pctL, 'single_contact_angle_deg': noise_deg,
                    'rule': f'On ONE contact, differences under {noise_pctL:.0f}% of leg length or {noise_deg} deg are noise. The plant is one contact per rep: compare reps (3 attempts) and left vs right turns, not single numbers against a threshold.'},
}
print('UNCERTAINTY:', summ['uncertainty']['rule'])
df.round(3).to_csv(f'{OUT}/per_frame.csv', index=False); st.to_csv(f'{OUT}/steps.csv', index=False)
json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
np.save(f'{OUT}/K_img.npy', np.stack([(np.linalg.inv(S[i]) @ np.c_[Ps[i], np.ones(17)].T).T[:, :2] for i in range(N)]))
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print(json.dumps(summ, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))); print(st.to_string(index=False))
