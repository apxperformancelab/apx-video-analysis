"""APX video analysis, step 3: 10m acceleration metrics from the SIDE-ON clip.

  python3 apx_accel.py --work WORKDIR --start-x XS --finish-x XF [--dist 10] [--slow-from A --slow-to B]

--start-x / --finish-x : x pixel of the start and finish line at the athlete's lane, in reference (frame 0)
                         coordinates (see apx_calib.py). Use the midpoint of a gate pair / cone pair.
--slow-from/--slow-to  : only if the clip still contains a slo-mo speed ramp: the frame range that is truly
                         slow motion. Timing metrics (contact, step frequency, speed) are only reported inside it.

Outputs to WORKDIR/analysis/: per_frame.csv, steps.csv, phases.csv, summary.json, K_img.npy
"""
import argparse, json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter

ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True); ap.add_argument('--start-x', type=float, required=True)
ap.add_argument('--finish-x', type=float, required=True); ap.add_argument('--dist', type=float, default=10.0)
ap.add_argument('--slow-from', type=int, default=None); ap.add_argument('--slow-to', type=int, default=None)
ap.add_argument('--gate-time', type=float, default=None, help='gate time for this attempt over --dist, for the frame-rate check')
ap.add_argument('--no-auto-ramp', action='store_true')
a = ap.parse_args()
OUT = f'{a.work}/analysis'; os.makedirs(OUT, exist_ok=True)
d = json.load(open(f'{a.work}/kp.json')); S = np.array(json.load(open(f'{a.work}/stab.json')))
meta = json.load(open(f'{a.work}/meta.json'))
FPS = d['fps']; N = len(d['frames'])
PXM = abs(a.finish_x - a.start_x) / a.dist
X_START = a.start_x
SLOW_FROM = a.slow_from if a.slow_from is not None else 0
SLOW_TO = a.slow_to if a.slow_to is not None else N - 1
odd = lambda x: int(x) // 2 * 2 + 1
W_SMOOTH = max(5, odd(round(0.06 * FPS)))      # ~60 ms keypoint smoothing
W_SPEED = max(7, odd(round(0.17 * FPS)))       # ~170 ms speed smoothing
W_HAMPEL = max(5, odd(round(0.06 * FPS)))

K = np.full((N, 17, 3), np.nan)
for r in d['frames']:
    if r['kp'] is not None:
        K[r['frame']] = np.array(r['kp'])
with np.errstate(all='ignore'):
    valid = ~np.isnan(K[:, 0, 0]) & (np.nanmean(K[:, :, 2], 1) > 0.5)
LEGCONF = np.nan_to_num(np.nanmin(K[:, 13:17, 2], 1)) if True else None
K[~valid] = np.nan
if valid.sum() < 0.6 * N:
    print(f'WARNING: athlete pose found in only {valid.sum()}/{N} frames')

# 1. into frame-0 coordinates (removes camera pan)
P = np.full((N, 17, 2), np.nan)
for i in range(N):
    if valid[i]:
        P[i] = (S[i] @ np.c_[K[i, :, :2], np.ones(17)].T).T[:, :2]

# 2. left/right label swap fix (continuity relative to hip)
nswaps = 0
for pl in ([(13, 14), (15, 16)], [(7, 8), (9, 10)]):
    prev = None
    for i in range(N):
        if not valid[i]:
            continue
        if prev is not None and i - prev <= 2:
            h0, h1 = P[prev, [11, 12]].mean(0), P[i, [11, 12]].mean(0)
            keep = sum(np.linalg.norm((P[i, x] - h1) - (P[prev, x] - h0)) + np.linalg.norm((P[i, y] - h1) - (P[prev, y] - h0)) for x, y in pl)
            swap = sum(np.linalg.norm((P[i, y] - h1) - (P[prev, x] - h0)) + np.linalg.norm((P[i, x] - h1) - (P[prev, y] - h0)) for x, y in pl)
            if swap < keep * 0.5:
                for x, y in pl:
                    P[i, [x, y]] = P[i, [y, x]]
                nswaps += 1
        prev = i

# 3. outlier removal (Hampel) + gap fill + smoothing
idx = np.where(valid)[0]
for j in range(17):
    for c in range(2):
        s = np.interp(np.arange(N), idx, P[idx, j, c])
        med = median_filter(s, W_HAMPEL, mode='nearest')
        mad = median_filter(np.abs(s - med), W_HAMPEL, mode='nearest') * 1.4826
        s[np.abs(s - med) > np.maximum(3 * mad, 0.045 * PXM)] = np.nan
        ok = ~np.isnan(s)
        P[:, j, c] = np.interp(np.arange(N), np.where(ok)[0], s[ok])
Ps = savgol_filter(P, W_SMOOTH, 2, axis=0)
hip = Ps[:, [11, 12]].mean(1); sho = Ps[:, [5, 6]].mean(1)
dirx = -1 if hip[idx[-1], 0] < hip[idx[0], 0] else 1
if np.sign(a.finish_x - a.start_x) != dirx:
    print('WARNING: start/finish x look reversed relative to running direction. Check calibration.')

def up_angle(top, bot):
    v = top - bot; return np.degrees(np.arctan2(v[..., 0] * dirx, -v[..., 1]))
def down_angle(top, bot):
    v = bot - top; return np.degrees(np.arctan2(v[..., 0] * dirx, v[..., 1]))
def joint_angle(p, q, r):
    v1, v2 = p - q, r - q
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)), -1, 1)))

SIDES = {'L': (11, 13, 15, 5, 7, 9), 'R': (12, 14, 16, 6, 8, 10)}
df = pd.DataFrame({'frame': np.arange(N), 't_s': np.arange(N) / FPS})
df['hip_x_m'] = (hip[:, 0] - X_START) * dirx / PXM
df['trunk_lean_deg'] = up_angle(sho, hip)
for s, (hp, kn, an, sh, el, wr) in SIDES.items():
    df[f'shin_{s}'] = up_angle(Ps[:, kn], Ps[:, an])
    df[f'thigh_{s}'] = down_angle(Ps[:, hp], Ps[:, kn])
    df[f'knee_{s}'] = joint_angle(Ps[:, hp], Ps[:, kn], Ps[:, an])
    df[f'hip_{s}'] = joint_angle(sho, Ps[:, hp], Ps[:, kn])
    df[f'elbow_{s}'] = joint_angle(Ps[:, sh], Ps[:, el], Ps[:, wr])
    df[f'shoulder_{s}'] = down_angle(Ps[:, sh], Ps[:, el]) + df['trunk_lean_deg']
df['valid'] = valid
df['speed_ms'] = np.gradient(savgol_filter(df['hip_x_m'].values, W_SPEED, 2)) * FPS
move_frame = int(np.argmax(df['speed_ms'].values > 0.3))

# --- automatic speed-ramp detection: real-time frames move the hip several times further per frame ---
ramp_note = None
if a.slow_from is None and a.slow_to is None and not a.no_auto_ramp:
    hx = (P[:, [11, 12], 0].mean(1) - X_START) * dirx / PXM
    dx = np.abs(np.diff(median_filter(hx, 5, mode='nearest'), prepend=hx[0]))
    loc = median_filter(dx, 7, mode='nearest')
    mov = loc[move_frame:]
    if len(mov) > 20:
        ref = np.median(mov[len(mov) // 4: 3 * len(mov) // 4])
        fast = loc > 1.6 * ref
        fast[:move_frame] = False
        buf = max(2, int(0.05 * FPS))
        win = int(0.3 * FPS)
        lead_idx = [f for f in range(move_frame, min(move_frame + win, N)) if fast[f]]
        lead_len = len(lead_idx)
        if lead_len >= 4:
            i0 = lead_idx[-1] + 1
            while i0 < N and loc[i0] > 1.0 * ref: i0 += 1
            i0 += buf
        else: i0 = move_frame
        end = N - 1
        while end > i0 and loc[end] < 0.2 * ref: end -= 1      # athlete stopped / out of shot
        trail_idx = [f for f in range(max(end - win, i0), end + 1) if fast[f]]
        trail_len = len(trail_idx)
        if trail_len >= 4:
            i1 = trail_idx[0] - 1
            while i1 > i0 and loc[i1] > 1.3 * ref: i1 -= 1
            i1 -= buf
        else: i1 = N - 1
        has_lead = lead_len >= 4; has_trail = trail_len >= 4
        if has_lead or has_trail:
            SLOW_FROM = i0 if has_lead else 0; SLOW_TO = i1 if has_trail else N - 1
            ramp_note = f'Auto-detected slo-mo speed ramp: real slow motion only in frames {SLOW_FROM}-{SLOW_TO}. Check this against the per-frame hip movement; override with --slow-from/--slow-to.'
            print('NOTE:', ramp_note)
def cross(x):
    v = df['hip_x_m'].values
    if not (v >= x).any(): return np.nan
    i = int(np.argmax(v >= x))
    return i if i == 0 else i - 1 + (x - v[i - 1]) / (v[i] - v[i - 1])
fin_c = cross(a.dist); fin_frame = int(np.ceil(fin_c)) if not np.isnan(fin_c) else N - 1

# 4. footprints: where an ankle (either label) stays planted
axw = np.r_[Ps[:, 15, 0], Ps[:, 16, 0]]; fidx = np.r_[np.arange(N), np.arange(N)]
lower = np.r_[Ps[:, 15, 1] >= Ps[:, 16, 1] - 0.025 * PXM, Ps[:, 16, 1] >= Ps[:, 15, 1] - 0.025 * PXM]
sel = lower & (fidx >= max(move_frame - 2, 0)) & (fidx <= min(fin_frame + int(0.15 * FPS), N - 1))
grid = np.arange(axw[sel].min() - 20, axw[sel].max() + 20, 0.5)
dens = np.exp(-0.5 * ((grid[:, None] - axw[sel][None]) / (0.023 * PXM)) ** 2).sum(1)
pk, _ = find_peaks(dens, distance=max(1, int(0.45 * PXM / 0.5)), height=max(3.0, 0.025 * FPS))
contacts = []
for p in pk:
    xf = grid[p]; near = sel & (np.abs(axw - xf) < 0.05 * PXM); fr = np.unique(fidx[near])
    if len(fr) < 3: continue
    runs, cur = [], [fr[0]]
    for f in fr[1:]:
        if f - cur[-1] <= max(2, int(0.017 * FPS)): cur.append(f)
        else: runs.append(cur); cur = [f]
    runs.append(cur); run = max(runs, key=len)
    if len(run) >= 3:
        td, to = int(run[0]), int(run[-1])
        contacts.append((xf, td, to))
contacts.sort(key=lambda c: c[1])

def leg_at(xf, frs):
    return 'L' if np.mean([abs(Ps[f, 15, 0] - xf) for f in frs]) <= np.mean([abs(Ps[f, 16, 0] - xf) for f in frs]) else 'R'
oth = lambda s: 'R' if s == 'L' else 'L'
rows = []
for xf, td, to in contacts:
    pos = (xf - X_START) * dirx / PXM
    if pos < -0.8 or pos > a.dist + 0.8: continue
    tdw = [td, min(td + 1, N - 1)]; tow = [max(to - 1, 0), to]
    s = leg_at(xf, tdw); s2 = leg_at(xf, tow); an = SIDES[s][2]
    m = lambda col, w: float(df.loc[w, col].mean())
    timing_ok = td >= SLOW_FROM and to <= SLOW_TO
    rows.append({
        'step': len(rows) + 1, 'td_frame': td, 'to_frame': to, 'timing_ok': timing_ok,
        'foot_pos_m': round(pos, 2),
        'trunk_TD': round(m('trunk_lean_deg', tdw), 1),
        'shin_TD': round(m(f'shin_{s}', tdw), 1),
        'shin_trunk_gap': round(m('trunk_lean_deg', tdw) - m(f'shin_{s}', tdw), 1),
        'foot_vs_hip_cm': round(np.mean([(xf - hip[f, 0]) * dirx for f in tdw]) / PXM * 100),
        'knee_TD': round(m(f'knee_{s}', tdw), 1),
        'hip_TO': round(m(f'hip_{s2}', tow), 1),
        'knee_TO': round(m(f'knee_{s2}', tow), 1),
        'body_lean_TO': round(float(np.mean([np.degrees(np.arctan2((hip[f, 0] - xf) * dirx, -(hip[f, 1] - Ps[f, SIDES[s2][2], 1]))) for f in tow])), 1),
        'contact_ms': round((to - td + 1) / FPS * 1000) if timing_ok else None,
        'speed_ms': round(float(df.loc[td:to, 'speed_ms'].mean()), 2) if timing_ok else None,
    })
st = pd.DataFrame(rows)
if len(st):
    st['step_length_m'] = st['foot_pos_m'].diff().round(2)
    ok2 = st['timing_ok'] & st['timing_ok'].shift(fill_value=False)
    st['step_time_ms'] = np.where(ok2, (st['td_frame'].diff() / FPS * 1000).round(), np.nan)
    st['step_freq_hz'] = (1000 / st['step_time_ms']).round(2)
    st['flight_ms'] = np.where(ok2, ((st['td_frame'] - st['to_frame'].shift() - 1) / FPS * 1000).round(), np.nan)
    # peak front-thigh (knee drive) between this touchdown and the next
    pk_th = []
    for i in range(len(st)):
        e = int(st.td_frame.iloc[i + 1]) if i + 1 < len(st) else min(int(st.to_frame.iloc[i]) + int(0.12 * FPS), N - 1)
        w = df.loc[int(st.td_frame.iloc[i]):e]
        pk_th.append(round(float(np.maximum(w.thigh_L, w.thigh_R).max()), 1))
    st['peak_front_thigh'] = pk_th

# 5. phases (distance bands) for the findings doc
bands = [(-0.8, 1.0, '0-1m'), (1.0, 3.0, '1-3m'), (3.0, 6.0, '3-6m'), (6.0, a.dist + 0.2, f'6-{a.dist:g}m')]
ph = []
for lo, hi, lab in bands:
    b = st[(st.foot_pos_m >= lo) & (st.foot_pos_m < hi) & (st.foot_pos_m > 0.2)] if len(st) else st
    if not len(b): continue
    ph.append({'phase': lab, 'steps': ','.join(map(str, b.step)), 'trunk_TD': round(b.trunk_TD.mean(), 1),
               'shin_TD': round(b.shin_TD.mean(), 1), 'shin_trunk_gap': round(b.shin_trunk_gap.mean(), 1),
               'foot_vs_hip_cm': round(b.foot_vs_hip_cm.mean()), 'step_length_m': round(b.step_length_m.mean(), 2),
               'contact_ms': round(b.contact_ms.dropna().mean()) if b.contact_ms.notna().any() else None})
phases = pd.DataFrame(ph)

# 6. slot frames for report images
early = st[st.foot_pos_m > 0.3].iloc[0] if (len(st) and (st.foot_pos_m > 0.3).any()) else None
df['leg_conf'] = LEGCONF
zone = df[(df.hip_x_m > -0.3) & (df.hip_x_m < 3) & df.valid & (df.frame >= move_frame) & (df.leg_conf > 0.7)]
peak_lean_frame = int(zone.loc[zone.trunk_lean_deg.idxmax(), 'frame']) if len(zone) else move_frame

# 7. quality checks
warn = []
slow = df[(df.frame >= max(SLOW_FROM, move_frame)) & (df.frame <= min(SLOW_TO, fin_frame))]
vmax = float(slow[(slow.hip_x_m > 1) & (slow.hip_x_m < a.dist)].speed_ms.max()) if len(slow) else np.nan
if vmax > 9: warn.append(f'Peak speed {vmax:.1f} m/s is implausible: clip probably still has a slo-mo speed ramp, wrong --capture-fps, or bad calibration.')
if vmax < 3: warn.append(f'Peak speed {vmax:.1f} m/s is unusually low: check --capture-fps (was it 120 not 240?) and calibration.')
if meta['camera_pan_px'] > 15: warn.append(f'Camera panned {meta["camera_pan_px"]:.0f} px; corrected, but distances carry extra error. Lock the tripod.')
if meta['athlete_height_px_median'] < 150: warn.append(f'Athlete only ~{meta["athlete_height_px_median"]:.0f} px tall: fine detail (arms, ankles) is unreliable. Film closer or send the original file.')
if max(meta['W'], meta['H']) < 1280: warn.append(f'Low resolution file ({meta["W"]}x{meta["H"]}): download the original from Frame.io, not a preview.')
if a.slow_from is not None or a.slow_to is not None or ramp_note: warn.append(f'Speed ramp present: timing metrics only for frames {SLOW_FROM}-{SLOW_TO}. Remove the ramp before sending next time.')
fps_check = None
if a.gate_time:
    c0, c1 = cross(0), fin_c
    if not np.isnan(c0) and not np.isnan(c1) and SLOW_FROM <= c0 and c1 <= SLOW_TO:
        t_video = (c1 - c0) / FPS
        implied = (c1 - c0) / a.gate_time
        fps_check = {'video_0_to_dist_s': round(t_video, 2), 'gate_s': a.gate_time, 'implied_capture_fps': round(implied * meta.get('stride', 1))}
        if abs(t_video - a.gate_time) / a.gate_time > 0.15:
            warn.append(f'Frame-rate check FAILED: video gives {t_video:.2f} s for 0-{a.dist:g} m vs gate {a.gate_time:.2f} s. Implied capture rate about {fps_check["implied_capture_fps"]} fps. Fix "fps" in kp.json and "real_fps" in meta.json, then re-run.')
    else:
        fps_check = 'not possible: 0m or finish crossing is outside the slow-motion section or not in shot'
if len(st) < 4: warn.append('Fewer than 4 steps detected: check the tracking contact sheet.')

# measurement uncertainty for single-step values: touchdown can be +-1.5 frames, keypoints +-2.5 px
hip_cm_per_frame = float(np.nanmedian(np.abs(np.diff(df.loc[move_frame:fin_frame, 'hip_x_m'].values)))) * 100
px_cm = 100 / PXM
noise_cm = round(1.5 * hip_cm_per_frame + 2.5 * px_cm * 1.4, 1)
ath_px = meta['athlete_height_px_median']
noise_deg = 8 if ath_px < 150 else (6 if ath_px < 300 else 4)
if FPS < 200: noise_deg += 1
summary_noise = {'single_step_foot_vs_hip_cm': noise_cm, 'single_step_angle_deg': noise_deg,
    'rule': f'Differences smaller than {noise_cm:.0f} cm or {noise_deg} deg on ONE step are noise for this clip. Flags need the pattern on 2+ consecutive steps or a phase average.'}
print('UNCERTAINTY:', summary_noise['rule'])

summary = {
    'real_fps': FPS, 'px_per_m': round(PXM, 1), 'running_dir': 'left' if dirx < 0 else 'right',
    'move_frame': move_frame, 'finish_frame': fin_frame, 'n_steps': len(st), 'lr_swaps_fixed': nswaps,
    'start_offset_m': round(float(df.hip_x_m.iloc[0]), 2),
    'time_0_to_dist_s_hip': round((fin_c - cross(0)) / FPS, 2) if (SLOW_FROM <= cross(0) and fin_c <= SLOW_TO) else None,
    'speed_at_finish_ms': round(float(df.loc[max(fin_frame - int(0.05 * FPS), 0):fin_frame, 'speed_ms'].mean()), 2) if fin_frame <= SLOW_TO else None,
    'peak_speed_ms': round(vmax, 2),
    'trunk_lean_TD_first_vs_last': [float(st.trunk_TD.iloc[0]), float(st.trunk_TD.iloc[-1])] if len(st) else None,
    'arms_side_view_approx': {s: {k: [round(float(df.loc[move_frame:fin_frame, f'{k}_{s}'].quantile(q)), 1) for q in (0.05, 0.95)] for k in ('shoulder', 'elbow')} for s in 'LR'},
    'slot_frames': {'early_drive': int(early.td_frame) if early is not None else None, 'peak_lean': peak_lean_frame},
    'uncertainty': summary_noise, 'fps_check': fps_check, 'slow_window': [SLOW_FROM, SLOW_TO],
    'warnings': warn,
}
df.round(3).to_csv(f'{OUT}/per_frame.csv', index=False); st.to_csv(f'{OUT}/steps.csv', index=False)
phases.to_csv(f'{OUT}/phases.csv', index=False); json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1)
Kimg = np.stack([(np.linalg.inv(S[i]) @ np.c_[Ps[i], np.ones(17)].T).T[:, :2] for i in range(N)])
np.save(f'{OUT}/K_img.npy', Kimg)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(json.dumps(summary, indent=1)); print(st.to_string(index=False)); print(phases.to_string(index=False))
