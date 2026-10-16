"""APX video analysis: MAX VELOCITY mechanics from a side-on or panning wide clip (30m test).

Works on imperfect footage. Nothing here needs cones in shot: distances are expressed relative to the athlete's
own leg length (hip to ankle), so a panning camera, a zoomed shot or a missing cone does not break it.
If --start-x/--finish-x (reference-frame x of the 0m and 30m lines, see apx_calib.py) are given, approximate
distance labels are added (approximate when the camera panned).

  python3 apx_maxv.py --work WORKDIR [--start-x XS --finish-x XF --dist 30] [--from-frame A --to-frame B]
                      [--height-cm 150] [--slow-from A --slow-to B]

--from/--to-frame : the part of the run to analyse as "max velocity" (default: the last 45% of the athlete's run)
--height-cm       : athlete height; turns leg-length units into approximate cm (leg length ~ 0.47 x height)

Outputs WORKDIR/analysis/: per_frame.csv, steps.csv, summary.json, K_img.npy  (compatible with apx_render.py)
"""
import argparse, json, os, warnings, numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True)
ap.add_argument('--start-x', type=float); ap.add_argument('--finish-x', type=float); ap.add_argument('--dist', type=float, default=30)
ap.add_argument('--line0', type=float, default=0, help='distance of the --start-x line from the start (e.g. 20 for a 20-30m side camera)')
ap.add_argument('--splits', type=float, nargs='*', default=[5, 10, 20, 30])
ap.add_argument('--from-frame', type=int); ap.add_argument('--to-frame', type=int)
ap.add_argument('--height-cm', type=float)
ap.add_argument('--slow-from', type=int); ap.add_argument('--slow-to', type=int)
ap.add_argument('--cross-frames', type=int, nargs=2, help='frames where the hip crosses the start and finish lines (self-calibrates distance from --dist)')
ap.add_argument('--gate-time', type=float, help='gate time over --dist, used to locate where slow motion starts if the clip has a ramp')
ap.add_argument('--hfov-deg', type=float, default=66.0, help='horizontal field of view of the 1x lens (landscape)')
a = ap.parse_args()
OUT = f'{a.work}/analysis'; os.makedirs(OUT, exist_ok=True)
d = json.load(open(f'{a.work}/kp.json')); S = np.array(json.load(open(f'{a.work}/stab.json'))); meta = json.load(open(f'{a.work}/meta.json'))
FPS = d['fps']; N = len(d['frames']); W, H = meta['W'], meta['H']
odd = lambda x: int(x) // 2 * 2 + 1

K = np.full((N, 17, 3), np.nan)
for r in d['frames']:
    if r['kp'] is not None: K[r['frame']] = np.array(r['kp'])
valid = ~np.isnan(K[:, 0, 0]) & (np.nan_to_num(np.nanmean(K[:, :, 2], 1)) > 0.5)
K[~valid] = np.nan
P = np.full((N, 17, 2), np.nan)
for i in range(N):
    if valid[i]: P[i] = (S[i] @ np.c_[K[i, :, :2], np.ones(17)].T).T[:, :2]

# leg length in px (hip to ankle along thigh + shank), rolling robust estimate: the local scale
def seglen(p, q): return np.linalg.norm(P[:, p] - P[:, q], axis=1)
legL = np.nanmax(np.c_[seglen(11, 13) + seglen(13, 15), seglen(12, 14) + seglen(14, 16)], 1)
idx = np.where(valid)[0]
legL_i = np.interp(np.arange(N), idx, legL[idx])
LPX = uniform_filter1d(median_filter(legL_i, odd(0.5 * FPS), mode='nearest'), odd(0.25 * FPS), mode='nearest')  # px per leg length
LEG_M = (0.47 * a.height_cm / 100) if a.height_cm else None

# label swap fix, outlier removal, smoothing (scale-aware: thresholds relative to leg length)
nswaps = 0
for pl in ([(13, 14), (15, 16)], [(7, 8), (9, 10)]):
    prev = None
    for i in range(N):
        if not valid[i]: continue
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
dirx = -1 if hip[idx[-1], 0] < hip[idx[0], 0] else 1

# viewing angle: camera pans; the athlete is closest (largest) where the line of sight is square to the lane
f_px = (W / 2) / np.tan(np.radians(a.hfov_deg / 2))
closest_frame = int(np.nanargmax(LPX[idx[0]:idx[-1] + 1]) + idx[0])
alpha = (hip[:, 0] - hip[closest_frame, 0]) / f_px            # radians off square-on (pan approximation)
cosA = np.clip(np.cos(alpha), 0.5, 1)

def up_angle(top, bot):
    v = top - bot; return np.degrees(np.arctan2(v[..., 0] * dirx / cosA, -v[..., 1]))
def down_angle(top, bot):
    v = bot - top; return np.degrees(np.arctan2(v[..., 0] * dirx / cosA, v[..., 1]))
def joint_angle(p, q, r):
    v1, v2 = p - q, r - q; v1 = v1 * np.c_[1 / cosA, np.ones(N)]; v2 = v2 * np.c_[1 / cosA, np.ones(N)]
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)), -1, 1)))

SIDES = {'L': (11, 13, 15, 5, 7, 9), 'R': (12, 14, 16, 6, 8, 10)}
df = pd.DataFrame({'frame': np.arange(N), 't_s': np.arange(N) / FPS})
# approximate distance along the run (only if both lines given; approximate when the camera panned)
if a.start_x is not None and a.finish_x is not None:
    df['hip_x_m'] = (hip[:, 0] - a.start_x) * dirx / (abs(a.finish_x - a.start_x) / a.dist)
else:
    df['hip_x_m'] = np.nan
df['leg_px'] = LPX; df['view_angle_deg'] = np.degrees(alpha)
df['trunk_lean_deg'] = up_angle(sho, hip)
nose_ahead = (Ps[:, 0, 0] - sho[:, 0]) * dirx / cosA / LPX
df['head_ahead_of_shoulder_pctL'] = nose_ahead * 100
for s_, (hp, kn, an, sh, el, wr) in SIDES.items():
    df[f'shin_{s_}'] = up_angle(Ps[:, kn], Ps[:, an]); df[f'thigh_{s_}'] = down_angle(Ps[:, hp], Ps[:, kn])
    df[f'knee_{s_}'] = joint_angle(Ps[:, hp], Ps[:, kn], Ps[:, an]); df[f'hip_{s_}'] = joint_angle(sho, Ps[:, hp], Ps[:, kn])
    df[f'elbow_{s_}'] = joint_angle(Ps[:, sh], Ps[:, el], Ps[:, wr]); df[f'shoulder_{s_}'] = down_angle(Ps[:, sh], Ps[:, el]) + df['trunk_lean_deg']
df['valid'] = valid

# speed proxy in leg lengths per second (scale-free) and slow-motion window
hipL = np.cumsum(np.r_[0, np.abs(np.diff(hip[:, 0])) / cosA[1:] / LPX[1:]])
v_legs = np.gradient(savgol_filter(hipL, max(7, odd(round(0.17 * FPS))), 2)) * FPS
df['speed_legs_per_s'] = v_legs
LEG_CAL = None
PXM_FIX = None
if a.start_x is not None and a.finish_x is not None and (meta.get('camera_pan_px') or 0) <= 0.15 * W:
    PXM_FIX = abs(a.finish_x - a.start_x) / a.dist
    df['hip_x_m'] = a.line0 + (hip[:, 0] - a.start_x) * dirx / PXM_FIX
    LEG_M = float(np.nanmedian(LPX[idx[0]:idx[-1]])) / PXM_FIX
    print(f'Fixed camera: {PXM_FIX:.1f} px per m from the lines; leg (hip-ankle) = {LEG_M:.3f} m')
if a.cross_frames and PXM_FIX is None:
    f0, f1 = a.cross_frames
    LEG_CAL = a.dist / (hipL[f1] - hipL[f0])          # metres per leg length, from the known test distance
    LEG_M = LEG_CAL
    df['hip_x_m'] = (hipL - hipL[f0]) * LEG_CAL
    print(f'Self-calibration: {a.dist:g} m between frames {f0} and {f1} = {hipL[f1]-hipL[f0]:.1f} leg lengths -> leg (hip-ankle) = {LEG_CAL:.3f} m')
move_frame = int(np.argmax(v_legs > 0.3))
SLOW_FROM, SLOW_TO = (a.slow_from or 0), (a.slow_to if a.slow_to is not None else N - 1)
if a.slow_to is None:
    dx_ = np.abs(np.diff(median_filter(hipL, 5, mode='nearest'), prepend=hipL[0])); loc_ = median_filter(dx_, 7, mode='nearest')
    mv_ = loc_[move_frame:]; ref_ = np.median(mv_[len(mv_) // 3: 2 * len(mv_) // 3]) if len(mv_) > 20 else np.median(loc_)
    end = int(idx[-1])
    while end > move_frame and loc_[end] < 0.2 * ref_: end -= 1
    tail = [f for f in range(max(end - int(0.3 * FPS), move_frame), end + 1) if loc_[f] > 1.6 * ref_]
    if len(tail) >= 4:
        i1 = tail[0] - 1
        while i1 > move_frame and loc_[i1] > 1.3 * ref_: i1 -= 1
        SLOW_TO = i1 - max(2, int(0.05 * FPS))
        print(f'NOTE: slow motion ends at about frame {SLOW_TO}; later frames are real time (timing not used).')

ramp_note = None
if a.slow_from is None and a.cross_frames and a.gate_time:
    f0, f1 = a.cross_frames; fps_rt = meta.get('file_fps', 30) or 30
    s_end = min(f1, SLOW_TO); tail_t = max(0, f1 - SLOW_TO) / fps_rt
    X = (a.gate_time - tail_t - s_end / FPS + f0 / fps_rt) / (1 / fps_rt - 1 / FPS)
    if f0 < X < f1:
        SLOW_FROM = int(round(X)) + max(2, int(0.05 * FPS))
        ramp_note = f'From the gate time: real time until about frame {int(round(X))}, slow motion after (abrupt-switch model). Timing used from frame {SLOW_FROM}.'
        print('NOTE:', ramp_note)
if ramp_note is None and a.slow_from is None and a.slow_to is None:
    dx = np.abs(np.diff(median_filter(hipL, 5, mode='nearest'), prepend=hipL[0])); loc = median_filter(dx, 7, mode='nearest')
    mov = loc[move_frame:]; ref = np.median(mov[len(mov) // 2:])     # late part of the run = slow motion if any
    fast = loc > 1.6 * ref; fast[:move_frame] = False; win = int(0.6 * FPS)
    lead = [f for f in range(move_frame, min(move_frame + win, N)) if fast[f]]
    if len(lead) >= 4:
        i0 = lead[-1] + 1
        while i0 < N and loc[i0] > 1.0 * ref: i0 += 1
        SLOW_FROM = i0 + max(2, int(0.05 * FPS))
        ramp_note = f'Slow motion starts at about frame {SLOW_FROM}; earlier frames are real time (positions and angles valid, timing not).'
        print('NOTE:', ramp_note)

# frame-rate check from step rhythm (independent of calibration)
seg = np.abs(Ps[:, 15, 0] - Ps[:, 16, 0])[SLOW_FROM:]
seg = seg - seg.mean(); ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]; ac /= ac[0] if ac[0] else 1
pk_ = [i for i in range(4, min(len(ac) - 1, int(FPS)) ) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2]
step_frames = pk_[0] if pk_ else None
step_rate_check = None
if step_frames:
    hz = FPS / step_frames
    step_rate_check = {'frames_per_step': step_frames, 'step_freq_hz_at_set_fps': round(hz, 2)}
    if not (3.3 <= hz <= 5.3):
        alt = [c for c in (120, 240, 60) if 3.3 <= c / step_frames <= 5.3]
        step_rate_check['warning'] = f'{hz:.1f} steps/s is implausible for youth sprinting at {FPS:.0f} fps. Likely capture rate: {alt[0] if alt else "unknown"} fps.'
        print('WARNING:', step_rate_check['warning'])

# section to analyse
end_frame = int(idx[-1]); run_frames = end_frame - move_frame
A0 = a.from_frame if a.from_frame is not None else int(move_frame + 0.55 * run_frames)
A1 = a.to_frame if a.to_frame is not None else min(end_frame, SLOW_TO)

# footprints (pan-corrected x, thresholds in leg lengths)
axw = np.r_[Ps[:, 15, 0], Ps[:, 16, 0]]; fidx = np.r_[np.arange(N), np.arange(N)]; Lw = np.r_[LPX, LPX]
lower = np.r_[Ps[:, 15, 1] >= Ps[:, 16, 1] - 0.03 * LPX, Ps[:, 16, 1] >= Ps[:, 15, 1] - 0.03 * LPX]
sel = lower & (fidx >= max(A0 - int(0.1 * FPS), 0)) & (fidx <= A1)
Lm = float(np.median(LPX[A0:A1 + 1]))
grid = np.arange(axw[sel].min() - 20, axw[sel].max() + 20, 0.5)
dens = np.exp(-0.5 * ((grid[:, None] - axw[sel][None]) / (0.03 * Lm)) ** 2).sum(1)
pk, _ = find_peaks(dens, distance=max(1, int(0.9 * Lm / 0.5)), height=max(3.0, 0.025 * FPS))
contacts = []
for p in pk:
    xf = grid[p]; near = sel & (np.abs(axw - xf) < 0.10 * Lw); fr = np.unique(fidx[near])
    if len(fr) < 3: continue
    runs, cur = [], [fr[0]]
    for f in fr[1:]:
        if f - cur[-1] <= max(2, int(0.017 * FPS)): cur.append(f)
        else: runs.append(cur); cur = [f]
    runs.append(cur); run = max(runs, key=len)
    if len(run) >= 3: contacts.append((xf, int(run[0]), int(run[-1])))
contacts.sort(key=lambda c: c[1])

def leg_at(xf, frs):
    return 'L' if np.mean([abs(Ps[f, 15, 0] - xf) for f in frs]) <= np.mean([abs(Ps[f, 16, 0] - xf) for f in frs]) else 'R'
rows = []
for xf, td, to in contacts:
    if td < A0 - int(0.05 * FPS) or td > A1: continue
    tdw = [td, min(td + 1, N - 1)]; tow = [max(to - 1, 0), to]; s_ = leg_at(xf, tdw); s2 = leg_at(xf, tow)
    m = lambda col, w: float(df.loc[w, col].mean())
    timing_ok = td >= SLOW_FROM and to <= SLOW_TO
    fvh = np.mean([(xf - hip[f, 0]) * dirx / cosA[f] / LPX[f] for f in tdw])
    stance = df.loc[td:to]
    rows.append({
        'step': len(rows) + 1, 'td_frame': td, 'to_frame': to, 'timing_ok': timing_ok, 'leg': s_,
        'foot_pos_m': round(float(df.loc[td, 'hip_x_m']), 1) if not np.isnan(df.loc[td, 'hip_x_m']) else np.nan,
        'foot_vs_hip_pctL': round(fvh * 100, 1),
        'foot_vs_hip_cm': round(fvh * LEG_M * 100) if LEG_M else np.nan,
        'shin_TD': round(m(f'shin_{s_}', tdw), 1), 'trunk_TD': round(m('trunk_lean_deg', tdw), 1),
        'knee_TD': round(m(f'knee_{s_}', tdw), 1),
        'knee_min_stance': round(float(stance[f'knee_{s_}'].min()), 1),
        'hip_TO': round(m(f'hip_{s2}', tow), 1),
        'contact_ms': round((to - td + 1) / FPS * 1000) if timing_ok else None,
        'view_angle_deg': round(float(np.degrees(alpha[td])), 0),
    })
st = pd.DataFrame(rows)
PANNED = (meta.get('camera_pan_px') or 0) > 0.15 * W
if PANNED:
    print('WARNING: camera panned heavily: distances, speeds and step lengths in metres are not reliable on this clip and are left out. '
          'Angles, step rhythm and foot-vs-hip as % of leg length are kept (with the printed uncertainty).')
if len(st):
    if PANNED:
        for c in ('foot_pos_m', 'foot_vs_hip_cm'): st[c] = np.nan
    # leg identity: steps must alternate; side-on tracking cannot tell left from right reliably, so label A / B
    st['leg'] = ['A' if i % 2 == 0 else 'B' for i in range(len(st))]
    gaps = st['td_frame'].diff()
    if len(st) > 2 and (gaps > 1.6 * gaps.median()).any():
        print('WARNING: a step appears to be missing (long gap between touchdowns); A/B labels after it may be swapped.')
    st['knee_flex_stance'] = (st['knee_TD'] - st['knee_min_stance']).round(1)
    st['step_length_legs'] = (st['td_frame'].map(lambda f: hipL[f]).diff()).round(2)
    st['step_length_m'] = (st['step_length_legs'] * LEG_M).round(2) if (LEG_M and not PANNED) else np.nan
    ok2 = st['timing_ok'] & st['timing_ok'].shift(fill_value=False)
    st['step_time_ms'] = np.where(ok2, (st['td_frame'].diff() / FPS * 1000).round(), np.nan)
    st['step_freq_hz'] = (1000 / st['step_time_ms']).round(2)
    st['flight_ms'] = np.where(ok2, ((st['td_frame'] - st['to_frame'].shift() - 1) / FPS * 1000).round(), np.nan)
    pk_th = []
    for i in range(len(st)):
        e = int(st.td_frame.iloc[i + 1]) if i + 1 < len(st) else min(int(st.to_frame.iloc[i]) + int(0.12 * FPS), N - 1)
        w = df.loc[int(st.td_frame.iloc[i]):e]; pk_th.append(round(float(np.maximum(w.thigh_L, w.thigh_R).max()), 1))
    st['peak_front_thigh'] = pk_th

# uncertainty for single steps (touchdown +-1.5 frames, keypoints +-2.5 px)
hip_legs_per_frame = float(np.nanmedian(np.abs(np.diff(hipL[A0:A1]))))
noise_pctL = round((1.5 * hip_legs_per_frame + 2.5 * 1.4 / Lm) * 100, 1)
ath_px = float(np.nanmedian(np.nanmax(K[:, :, 1], 1) - np.nanmin(K[:, :, 1], 1)))
noise_deg = (8 if ath_px < 150 else (6 if ath_px < 300 else 4)) + (1 if FPS < 200 else 0) + (2 if np.nanmax(np.abs(np.degrees(alpha[A0:A1]))) > 30 else 0)

def lr(col):
    if not len(st) or st[col].isna().all(): return None
    g = st.groupby('leg')[col].mean()
    return {k: round(float(v), 1) for k, v in g.items()}
summ = {
    'real_fps': FPS, 'running_dir': 'left' if dirx < 0 else 'right', 'camera_pan_px': meta.get('camera_pan_px'),
    'athlete_height_px': round(ath_px), 'leg_length_px_in_section': round(Lm, 1),
    'analysed_frames': [A0, A1], 'move_frame': move_frame, 'slow_window': [SLOW_FROM, SLOW_TO], 'ramp_note': ramp_note,
    'step_rate_check': step_rate_check, 'n_steps': len(st), 'lr_swaps_fixed': nswaps,
    'view_angle_range_deg': [round(float(np.degrees(alpha[A0:A1]).min())), round(float(np.degrees(alpha[A0:A1]).max()))],
    'means': {c: (round(float(st[c].mean()), 1) if len(st) and st[c].notna().any() else None) for c in
              ['foot_vs_hip_pctL', 'shin_TD', 'trunk_TD', 'knee_flex_stance', 'contact_ms', 'flight_ms', 'step_freq_hz', 'step_length_legs', 'peak_front_thigh']},
    'left_vs_right': {c: lr(c) for c in ['contact_ms', 'foot_vs_hip_pctL', 'shin_TD', 'knee_flex_stance', 'step_length_legs']},
    'head_ahead_of_shoulder_pctL_median': round(float(np.nanmedian(df.loc[A0:A1, 'head_ahead_of_shoulder_pctL'])), 1),
    'uncertainty': {'single_step_foot_vs_hip_pctL': noise_pctL, 'single_step_angle_deg': noise_deg,
                    'rule': f'On ONE step, differences under {noise_pctL:.0f}% of leg length or {noise_deg} deg are noise. Flags need 2+ steps or the section mean. Left vs right: compare means of 2+ steps per side.'},
    'leg_length_m': (round(LEG_M, 3) if LEG_M else None), 'leg_length_source': ('self-calibrated from the test distance' if LEG_CAL else ('0.47 x height' if a.height_cm else None)),
    'speed_ms_section': (round(float(np.nanmedian(df.loc[max(A0, SLOW_FROM):A1, 'speed_legs_per_s'])) * LEG_M, 2) if (LEG_M and not PANNED) else None),
    'panned_camera': PANNED, 'metric_outputs': ('left out: panning camera' if PANNED else ('included' if LEG_M else 'left out: no calibration or height')),
}
# ---- timing from video (fixed camera with lines only) ----
timing = None
if PXM_FIX is not None and not PANNED:
    fps_rt = meta.get('file_fps', 30) or 30
    tvec = np.array([(min(f, SLOW_FROM) / fps_rt if SLOW_FROM > 0 else 0) + (min(f, SLOW_TO) - SLOW_FROM) / FPS * (f > SLOW_FROM) + max(0, f - SLOW_TO) / fps_rt for f in range(N)])
    xm = savgol_filter(np.nan_to_num(df['hip_x_m'].values, nan=0), max(7, odd(round(0.1 * FPS))), 2)
    def tcross(x):
        if not (xm >= x).any() or xm[0] >= x: return None
        i = int(np.argmax(xm >= x)); fr_ = i - 1 + (x - xm[i - 1]) / (xm[i] - xm[i - 1])
        return float(np.interp(fr_, np.arange(N), tvec)), fr_
    c0 = tcross(a.line0 if a.line0 > 0 else 0.0)
    timing = {'note': 'video-timed: hip crossing each line; not interchangeable with gate times'}
    if c0:
        for sd in a.splits:
            if a.line0 < sd <= a.line0 + a.dist + 0.01:
                c = tcross(sd)
                if c: timing[f'{a.line0:g}-{sd:g}m_s'] = round(c[0] - c0[0], 3) + 0.0
                if c and (c0[1] < SLOW_FROM or c[1] > SLOW_TO): timing['warning'] = 'Part of this interval is before slow motion starts: times use the gate-time ramp model and are approximate.'
    sw = df.loc[max(SLOW_FROM, move_frame):min(SLOW_TO, idx[-1])]
    if len(sw) > 10:
        vms = np.gradient(savgol_filter(np.nan_to_num(sw['hip_x_m'].values), max(7, odd(round(0.17 * FPS))), 2)) * FPS
        timing['top_speed_ms'] = round(float(np.percentile(vms, 95)), 2)
    # force-velocity profile (Samozino method, no air resistance) when the run is seen from the start
    try:
        from scipy.optimize import curve_fit
        if a.line0 == 0 and c0 and a.dist >= 20:
            mask = (np.arange(N) >= move_frame) & (np.arange(N) <= min(idx[-1], SLOW_TO)) & ~np.isnan(df['hip_x_m'].values)
            tt, xx = tvec[mask], xm[mask]
            mdl = lambda t, vmax, tau, t0, x0: np.where(t > t0, vmax * ((t - t0) + tau * np.exp(-(t - t0) / tau) - tau), 0) + x0
            (vmax, tau, t0, x0), _ = curve_fit(mdl, tt, xx, p0=[7, 1.2, tt[0], xx[0]], maxfev=20000)
            timing['fv_profile_estimate'] = {'vmax_ms': round(vmax, 2), 'tau_s': round(tau, 2), 'F0_N_per_kg': round(vmax / tau, 2),
                'Pmax_W_per_kg': round(vmax * vmax / tau / 4, 2), 'note': 'Samozino method from video positions, no air-resistance correction: estimate only'}
    except Exception as e:
        timing['fv_profile_estimate'] = f'not fitted: {e}'
summ['video_timing'] = timing
print('UNCERTAINTY:', summ['uncertainty']['rule'])
if PANNED: df['hip_x_m'] = np.nan
df.round(3).to_csv(f'{OUT}/per_frame.csv', index=False); st.to_csv(f'{OUT}/steps.csv', index=False)
json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
np.save(f'{OUT}/K_img.npy', np.stack([(np.linalg.inv(S[i]) @ np.c_[Ps[i], np.ones(17)].T).T[:, :2] for i in range(N)]))
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print(json.dumps(summ, indent=1, default=str)); print(st.to_string(index=False))
