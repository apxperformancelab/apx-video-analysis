"""APX video analysis, step 4: outputs.

Side-on 10m (after apx_accel.py):
  python3 apx_render.py accel --work WORKDIR --name "Athlete Name" --out OUTDIR
    -> APX_<Name>_10m_Side_Overlay.mp4, APX_<Name>_10m_Touchdown_Reel.jpg,
       APX_<Name>_10m_Early_Drive_[annotated|clean].jpg, APX_<Name>_10m_Peak_Drive_Lean_[annotated|clean].jpg,
       APX_<Name>_10m_Data.xlsx

Any clip, skeleton only (front-on until front metrics are validated):
  python3 apx_render.py plain --work WORKDIR --name "Athlete Name" --out OUTDIR --label "10m Front"
    -> APX_<Name>_<label>_Overlay.mp4 and <WORKDIR>/pick_frames.jpg (numbered frames to choose a still from)

Export a still from any analysed clip:
  python3 apx_render.py still --work WORKDIR --name "Athlete Name" --out OUTDIR --frame K --slot "10m Front-On Arm and Knee Drive"
"""
import argparse, json, os, subprocess, warnings, cv2, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['accel', 'maxv', 'cod', 'plain', 'still', 'reel', 'video'])
ap.add_argument('--work', required=True); ap.add_argument('--name', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--label', default='10m Front'); ap.add_argument('--frame', type=int); ap.add_argument('--slot', default='Still'); ap.add_argument('--caption', default=''); ap.add_argument('--trunk-only', action='store_true')
ap.add_argument('--stills', nargs='*', default=[]); ap.add_argument('--inputs', nargs='*', default=[]); ap.add_argument('--test', default=None); ap.add_argument('--dist', type=float, default=None)
a = ap.parse_args()
if a.test is None: a.test = {'maxv': '30m Max Velocity', 'cod': '505'}.get(a.mode, '10m Acceleration')
if a.dist is None: a.dist = 30.0 if (a.mode == 'maxv' or a.test.startswith('30')) else 10.0
TESTTAG = a.test.replace(' Max Velocity', '').replace(' Acceleration', '').replace(' ', '_')
os.makedirs(a.out, exist_ok=True)
TAG = a.name.strip().replace(' ', '_')

# ---------- brand ----------
FLUORO = (204, 255, 0); WHITE = (255, 255, 255); BG = (10, 10, 10); GREY = (150, 150, 150); TRUNK = (77, 195, 255)
def font(sz, bold=True):
    for p in (['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
bgr = lambda c: (c[2], c[1], c[0])

def load_frames(path):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr
frames = load_frames(f'{a.work}/work.mp4'); N = len(frames); H, W = frames[0].shape[:2]
A = f'{a.work}/analysis'
if os.path.exists(f'{A}/K_img.npy'):
    K = np.load(f'{A}/K_img.npy')
else:  # plain mode without analysis: raw keypoints, lightly cleaned
    d = json.load(open(f'{a.work}/kp.json')); K = np.full((N, 17, 2), np.nan)
    for r in d['frames']:
        if r['kp'] is not None: K[r['frame']] = np.array(r['kp'])[:, :2]
    from scipy.signal import savgol_filter
    ok = ~np.isnan(K[:, 0, 0]); idx = np.where(ok)[0]
    for j in range(17):
        for c in range(2):
            K[:, j, c] = np.interp(np.arange(N), idx, K[idx, j, c])
    K = savgol_filter(K, 5, 2, axis=0)
valid = ~np.isnan(K[:, 0, 0])

def draw_skel(img, k, sc=1.0, off=(0, 0), th=3):
    q = lambda xy: (int((xy[0] - off[0]) * sc), int((xy[1] - off[1]) * sc))
    hm = (k[11] + k[12]) / 2; sm = (k[5] + k[6]) / 2
    for col, (el, wr, kn, an) in ((FLUORO, (7, 9, 13, 15)), (WHITE, (8, 10, 14, 16))):
        c = bgr(col)
        for p1, p2 in ((sm, k[el]), (k[el], k[wr]), (hm, k[kn]), (k[kn], k[an])):
            cv2.line(img, q(p1), q(p2), c, th, cv2.LINE_AA)
        for j in (el, wr, kn, an): cv2.circle(img, q(k[j]), th + 2, c, -1, cv2.LINE_AA)
    cv2.line(img, q(hm), q(sm), bgr(TRUNK), th + 1, cv2.LINE_AA)
    for p in (hm, sm): cv2.circle(img, q(p), th + 3, bgr(TRUNK), -1, cv2.LINE_AA)

def put(img_bgr, items):
    """items: list of (text, (x,y), size, colour). Draw with PIL for clean type."""
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(im)
    for t, xy, sz, col in items: dr.text(xy, t, font=font(sz), fill=col)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

hipc = K[:, [11, 12]].mean(1)
from scipy.ndimage import uniform_filter1d
cx = uniform_filter1d(hipc[:, 0], 9, mode='nearest'); cy = uniform_filter1d(hipc[:, 1], 25, mode='nearest')
span = np.nanmedian(np.nanmax(K[:, :, 1], 1) - np.nanmin(K[:, :, 1], 1))  # athlete height in px
CS = int(np.clip(span * 1.45, 80, min(W, H)))
spanf = uniform_filter1d(np.nan_to_num(np.nanmax(K[:, :, 1], 1) - np.nanmin(K[:, :, 1], 1), nan=span), 9, mode='nearest')

def crop_box(i, size=CS, cyf=0.55):
    x0 = int(np.clip(cx[i] - size / 2, 0, W - size)); y0 = int(np.clip(cy[i] - size * cyf, 0, H - size)); return x0, y0

def encode(raw, final):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', final], check=True)
    os.remove(raw)

def angle_marker(img, base, ang, length, sc_off, col, label):
    """Draw a dashed vertical reference at `base` (image px) and the angle label."""
    (bx, by) = base
    for yy in range(0, int(length), 14):
        cv2.line(img, (bx, by - yy), (bx, by - min(yy + 8, int(length))), bgr(GREY), 2, cv2.LINE_AA)
    return (label, (bx + 12, by - int(length) - 4))

def touchdown_tiles(st, dist, w=400):
    tiles = []
    sel_ = st if st.foot_pos_m.isna().all() or a.mode == 'maxv' else st[(st.foot_pos_m > 0.3) & (st.foot_pos_m <= dist + 0.2)]
    for s_ in sel_.itertuples():
        i = int(s_.td_frame); size = int(CS * 0.95); x0, y0 = crop_box(i, size)
        z = cv2.resize(frames[i][y0:y0 + size, x0:x0 + size], (w, w), interpolation=cv2.INTER_CUBIC)
        draw_skel(z, K[i], w / size, (x0, y0), 2)
        pad = np.zeros((130, w, 3), np.uint8); pad[:] = bgr(BG)
        pos = f'  ·  {s_.foot_pos_m:.1f} m' if pd.notna(s_.foot_pos_m) else ''
        fv = f'Foot {s_.foot_vs_hip_cm:+.0f} cm vs hip' if pd.notna(s_.foot_vs_hip_cm) else f'Foot {getattr(s_, "foot_vs_hip_pctL", float("nan")):+.0f}% leg vs hip'
        tiles.append(put(np.vstack([z, pad]), [(f'STEP {s_.step}{pos}', (14, w + 10), 26, FLUORO),
                    (f'Shin {s_.shin_TD:.0f}°   Trunk {s_.trunk_TD:.0f}°', (14, w + 50), 23, WHITE), (fv, (14, w + 86), 23, WHITE)]))
    return tiles

def cod_tiles(st, w=400):
    tiles = []
    ap_ = st[st.role == 'approach'].tail(2)
    st = pd.concat([ap_, st[st.role.isin(['penultimate', 'plant', 'second foot (turn)', 'first step out'])]])
    for s_ in st.itertuples():
        i = int(s_.deepest_frame); size = int(CS * 0.95); x0, y0 = crop_box(i, size)
        z = cv2.resize(frames[i][y0:y0 + size, x0:x0 + size], (w, w), interpolation=cv2.INTER_CUBIC)
        draw_skel(z, K[i], w / size, (x0, y0), 2)
        pad = np.zeros((130, w, 3), np.uint8); pad[:] = bgr(BG)
        l2 = f'Hip height {s_.hip_height_min_pctL:.0f}% leg' + (f'   ~{s_.contact_ms:.0f} ms' if pd.notna(s_.contact_ms) else '')
        l3 = '   '.join(([f'Knee bend {s_.knee_flex_max:.0f}°'] if pd.notna(s_.knee_flex_max) else []) + ([f'Trunk {s_.trunk_deepest:.0f}°'] if pd.notna(getattr(s_, 'trunk_deepest', np.nan)) else [])
                       + ([f'Knee line {s_.knee_line_deg:+.0f}°'] if pd.notna(getattr(s_, 'knee_line_deg', np.nan)) else [])) or f'Body {s_.body_to_camera}'
        tiles.append(put(np.vstack([z, pad]), [(f'{s_.role.upper()}', (14, w + 10), 26, FLUORO), (l2, (14, w + 50), 23, WHITE), (l3, (14, w + 86), 23, WHITE)]))
    return tiles

def still(i, annotate, title, lines, fname, step=None, trunk_only=False, only_leg=None, knee_line=None):
    size = int(np.clip(spanf[i] * 1.6, 80, min(W, H))); x0, y0 = crop_box(i, size)
    z = cv2.resize(frames[i][y0:y0 + size, x0:x0 + size], (1080, 1080), interpolation=cv2.INTER_CUBIC)
    if annotate:
        sc = 1080 / size; k = K[i]
        if only_leg is not None:   # trunk + one leg (e.g. the push-off leg at toe-off)
            qq = lambda xy: (int((xy[0] - x0) * sc), int((xy[1] - y0) * sc))
            h_, s_ = qq((k[11] + k[12]) / 2), qq((k[5] + k[6]) / 2); kn, an = (13, 15) if only_leg == 'L' else (14, 16)
            cv2.line(z, h_, s_, bgr(TRUNK), 6, cv2.LINE_AA); cv2.line(z, h_, qq(k[kn]), bgr(FLUORO), 5, cv2.LINE_AA); cv2.line(z, qq(k[kn]), qq(k[an]), bgr(FLUORO), 5, cv2.LINE_AA)
            for p_ in (h_, s_): cv2.circle(z, p_, 9, bgr(TRUNK), -1, cv2.LINE_AA)
            for p_ in (qq(k[kn]), qq(k[an])): cv2.circle(z, p_, 8, bgr(FLUORO), -1, cv2.LINE_AA)
        elif trunk_only:
            qq = lambda xy: (int((xy[0] - x0) * sc), int((xy[1] - y0) * sc))
            h_, s_ = qq((k[11] + k[12]) / 2), qq((k[5] + k[6]) / 2)
            ext = (int(h_[0] + (s_[0] - h_[0]) * 1.25), int(h_[1] + (s_[1] - h_[1]) * 1.25))
            cv2.line(z, h_, ext, bgr(TRUNK), 6, cv2.LINE_AA)
            for p_ in (h_, s_): cv2.circle(z, p_, 9, bgr(TRUNK), -1, cv2.LINE_AA)
        else:
            draw_skel(z, k, sc, (x0, y0), 4)
        q = lambda xy: (int((xy[0] - x0) * sc), int((xy[1] - y0) * sc))
        if knee_line is not None:   # front view: dashed hip-to-ankle line and the knee, to show knee in/out
            hp_, kn_, an_ = (11, 13, 15) if knee_line == 'L' else (12, 14, 16)
            p0, p1 = np.array(q(k[hp_]), float), np.array(q(k[an_]), float); L_ = np.linalg.norm(p1 - p0)
            for t0 in np.arange(0, 1, 0.08): cv2.line(z, tuple((p0 + (p1 - p0) * t0).astype(int)), tuple((p0 + (p1 - p0) * min(t0 + 0.05, 1)).astype(int)), bgr(TRUNK), 4, cv2.LINE_AA)
            cv2.circle(z, q(k[kn_]), 14, bgr(FLUORO), 3, cv2.LINE_AA)
        labels = []
        hm = q((k[11] + k[12]) / 2); sm = q((k[5] + k[6]) / 2)
        if knee_line is None: labels.append(angle_marker(z, hm, None, abs(hm[1] - sm[1]), None, TRUNK, ''))
        if step is not None:
            an = q(k[15] if step.get('leg') == 'L' else k[16]); kn = q(k[13] if step.get('leg') == 'L' else k[14])
            angle_marker(z, an, None, abs(an[1] - kn[1]), None, FLUORO, '')
        bar = np.zeros((170 + (40 if len(lines) > 2 else 0), 1080, 3), np.uint8); bar[:] = bgr(BG)
        z = np.vstack([z, bar])
        items = [(title.upper(), (36, 1080 + 22), 34, FLUORO)] + [(ln, (36, 1080 + 72 + 44 * j), 30 if j < 2 else 22, WHITE if j < 2 else GREY) for j, ln in enumerate(lines)]
        items.append(('APX PERFORMANCE LAB', (1080 - 330, 1080 + 30), 22, GREY))
        z = put(z, items)
    cv2.imwrite(fname, z, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if annotate:
        json.dump({'title': title, 'lines': lines, 'frame': int(i)}, open(fname.replace('.jpg', '.json'), 'w'))

if a.mode in ('accel', 'maxv'):
    df = pd.read_csv(f'{A}/per_frame.csv'); st = pd.read_csv(f'{A}/steps.csv')
    ph = pd.read_csv(f'{A}/phases.csv') if os.path.exists(f'{A}/phases.csv') else pd.DataFrame()
    summ = json.load(open(f'{A}/summary.json')); dist = a.dist
    # which leg is planted at each touchdown (closest ankle to footprint) for drawing the shin reference
    def planted_leg(r):
        f = int(r.td_frame); xf = None
        return 'L' if abs(df.loc[f, 'shin_L'] - r.shin_TD) <= abs(df.loc[f, 'shin_R'] - r.shin_TD) else 'R'
    td = {int(r.td_frame): r for r in st.itertuples()}
    OW, OH = 1280, 720
    raw = f'{a.out}/_raw.mp4'; vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (OW, OH)); hold = None
    for i in range(N):
        cv_ = np.zeros((OH, OW, 3), np.uint8); cv_[:] = bgr(BG)
        x0, y0 = crop_box(i); z = cv2.resize(frames[i][y0:y0 + CS, x0:x0 + CS], (OH, OH), interpolation=cv2.INTER_CUBIC)
        if valid[i]: draw_skel(z, K[i], OH / CS, (x0, y0), 3)
        cv_[:, :OH] = z
        full = frames[i].copy(); cv2.rectangle(full, (x0, y0), (x0 + CS, y0 + CS), bgr(FLUORO), max(2, W // 400))
        iw = 520; ih = int(iw * H / W); cv_[20:20 + ih, OH + 20:OH + 20 + iw] = cv2.resize(full, (iw, ih))
        r = df.iloc[i]; X = OH + 24; Y = 20 + ih + 24; items = []
        items.append((f'{a.name.upper()}  |  {a.test.upper()}', (X, Y), 22, FLUORO)); Y += 44
        if pd.notna(r.hip_x_m): items.append((f'Distance   {max(r.hip_x_m, 0):4.1f} m', (X, Y), 26, WHITE))
        Y += 38
        items.append((f'Trunk lean   {r.trunk_lean_deg:3.0f}°', (X, Y), 26, TRUNK)); Y += 46
        if i in td: hold = (td[i], 14)
        if hold:
            s, n = hold
            items.append((f'TOUCHDOWN  ·  STEP {s.step}', (X, Y), 24, FLUORO)); Y += 36
            fv = f'Foot {s.foot_vs_hip_cm:+.0f} cm vs hip' if pd.notna(s.foot_vs_hip_cm) else (f'Foot {s.foot_vs_hip_pctL:+.0f}% leg vs hip' if 'foot_vs_hip_pctL' in st else '')
            items.append((f'Shin {s.shin_TD:.0f}°   {fv}', (X, Y), 22, WHITE)); Y += 32
            extra = f'Step {s.step_length_m:.2f} m' if pd.notna(s.step_length_m) else ''
            if pd.notna(s.contact_ms): extra += f'   Contact ~{s.contact_ms:.0f} ms'
            items.append((extra, (X, Y), 22, WHITE))
            hold = (s, n - 1) if n > 1 else None
        by = OH - 56; bx0, bx1 = X, OW - 36
        cv2.line(cv_, (bx0, by), (bx1, by), bgr(GREY), 2)
        for mm in (0, dist / 2, dist):
            xx = int(bx0 + (bx1 - bx0) * mm / dist); cv2.line(cv_, (xx, by - 8), (xx, by + 8), bgr(GREY), 2)
            items.append((f'{mm:g}m', (xx - 12, by + 14), 16, GREY))
        for s in st.itertuples():
            if s.td_frame <= i and pd.notna(s.foot_pos_m) and 0 <= s.foot_pos_m <= dist + 0.5:
                cv2.circle(cv_, (int(bx0 + (bx1 - bx0) * min(s.foot_pos_m, dist) / dist), by), 6, bgr(FLUORO), -1, cv2.LINE_AA)
        if pd.notna(r.hip_x_m): cv2.circle(cv_, (int(bx0 + (bx1 - bx0) * np.clip(r.hip_x_m, 0, dist) / dist), by - 20), 8, bgr(WHITE), -1, cv2.LINE_AA)
        items.append(('Fluoro / white = the two legs & arms   Blue = trunk   Dots = footprints', (20, OH - 30), 15, WHITE))
        vw.write(put(cv_, items))
    vw.release(); encode(raw, f'{a.out}/APX_{TAG}_{TESTTAG}_{"Side" if a.mode == "accel" else "Wide" if summ.get("camera_pan_px", 0) and summ.get("camera_pan_px", 0) > 200 else "Side"}_Overlay.mp4')

    # touchdown reel
    tiles = touchdown_tiles(st, dist)
    per = 5; rows_ = []
    for r0 in range(0, len(tiles), per):
        row = tiles[r0:r0 + per]
        while len(row) < per: row.append(np.full_like(tiles[0], BG[0]))
        rows_.append(np.hstack(row))
    head = np.zeros((90, rows_[0].shape[1], 3), np.uint8); head[:] = bgr(BG)
    head = put(head, [(f'{a.name.upper()}  |  {a.test.upper()}  |  TOUCHDOWN REEL', (20, 26), 32, FLUORO),
                      ('APX PERFORMANCE LAB', (rows_[0].shape[1] - 340, 32), 24, GREY)])
    cv2.imwrite(f'{a.out}/APX_{TAG}_{TESTTAG}_Touchdown_Reel.jpg', np.vstack([head] + rows_), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # report slot stills
    sf_ = summ.get('slot_frames', {})
    if a.mode == 'maxv':
        def legs_apart(i):
            k = K[i]; h = spanf[i]
            return np.linalg.norm(k[13] - k[14]) > 0.06 * h or np.linalg.norm(k[15] - k[16]) > 0.06 * h
        good = st[st.shin_TD.notna() & st.td_frame.map(lambda f: legs_apart(int(f))) & st.to_frame.map(lambda f: legs_apart(int(f)))]
        if not len(good): print('WARNING: no step with both legs tracked separately at touchdown and toe-off; stills skipped')
        if len(good):
            mid = good.iloc[len(good) // 2]
            fv = f'{mid.foot_vs_hip_cm:+.0f} cm' if pd.notna(mid.foot_vs_hip_cm) else f'{mid.foot_vs_hip_pctL:+.0f}% leg'
            for ann in (True, False):
                still(int(mid['td_frame']), ann, 'Max Velocity · Ground Contact', [f'Step {int(mid.step)}' + (f' at {mid.foot_pos_m:.1f} m' if pd.notna(mid.foot_pos_m) else '') + f'   Shin {mid.shin_TD:.0f}°   Trunk {mid.trunk_TD:.0f}°', f'Foot {fv} vs hip at touchdown'],
                      f'{a.out}/APX_{TAG}_{TESTTAG}_Ground_Contact_{"annotated" if ann else "clean"}.jpg')
                fto = int(mid['to_frame']); dirx_ = -1 if summ.get('running_dir') == 'left' else 1
                push = 'L' if (K[fto, 15, 0] - K[fto, 11:13, 0].mean()) * dirx_ < (K[fto, 16, 0] - K[fto, 11:13, 0].mean()) * dirx_ else 'R'
                still(fto, ann, 'Max Velocity · Toe Off', [f'Step {int(mid.step)}: push-off leg and trunk', f'Hip angle {mid.hip_TO:.0f}° (approx)'],
                      f'{a.out}/APX_{TAG}_{TESTTAG}_Toe_Off_{"annotated" if ann else "clean"}.jpg', only_leg=push)
    ed = sf_.get('early_drive') if a.mode == 'accel' else None
    if ed is not None:
        s = st[st.td_frame == ed].iloc[0]
        lines = [f'Step {int(s.step)} at {s.foot_pos_m:.1f} m   Shin {s.shin_TD:.0f}°   Trunk {s.trunk_TD:.0f}°', f'Foot {s.foot_vs_hip_cm:+.0f} cm vs hip at touchdown']
        for ann in (True, False):
            still(ed, ann, 'Acceleration · Early Drive Phase', lines, f'{a.out}/APX_{TAG}_10m_Early_Drive_{"annotated" if ann else "clean"}.jpg', {'leg': planted_leg(s)})
    pl = sf_.get('peak_lean') if a.mode == 'accel' else None
    for ann in ((True, False) if pl is not None else ()):
        r = df.iloc[pl]
        still(pl, ann, 'Acceleration · Peak Drive Lean', [f'Trunk lean {r.trunk_lean_deg:.0f}° at {max(r.hip_x_m, 0):.1f} m'],
              f'{a.out}/APX_{TAG}_10m_Peak_Drive_Lean_{"annotated" if ann else "clean"}.jpg', trunk_only=True)

    # data workbook
    nice = st.rename(columns={'step': 'Step', 'foot_pos_m': 'Foot lands at (m)', 'step_length_m': 'Step length (m)',
        'trunk_TD': 'Trunk lean at touchdown (°)', 'shin_TD': 'Shin angle at touchdown (°, + = knee ahead of ankle)',
        'shin_trunk_gap': 'Trunk minus shin (°)', 'foot_vs_hip_cm': 'Foot vs hip at touchdown (cm, + = ahead)',
        'knee_TD': 'Knee angle at touchdown (°)', 'hip_TO': 'Hip angle at toe-off (°, approx)', 'knee_TO': 'Knee angle at toe-off (°, approx)',
        'body_lean_TO': 'Body lean at toe-off, foot-to-hip (°)', 'peak_front_thigh': 'Peak front thigh (° from vertical, approx)',
        'contact_ms': 'Ground contact (ms, approx)', 'flight_ms': 'Flight (ms, approx)', 'step_freq_hz': 'Step frequency (steps/s)',
        'speed_ms': 'Speed (m/s)', 'td_frame': 'Touchdown frame', 'to_frame': 'Toe-off frame', 'foot_vs_hip_pctL': 'Foot vs hip (% leg length)', 'knee_flex_stance': 'Knee flexion in stance (°, approx)', 'step_length_legs': 'Step length (leg lengths)', 'view_angle_deg': 'Camera view angle off square (°)', 'leg': 'Leg (A/B alternate)'}).drop(columns=['timing_ok', 'step_time_ms', 'knee_min_stance'], errors='ignore')
    with pd.ExcelWriter(f'{a.out}/APX_{TAG}_{TESTTAG}_Data.xlsx') as xw:
        nice.to_excel(xw, sheet_name='Steps', index=False)
        if len(ph): ph.to_excel(xw, sheet_name='Phases', index=False)
        pd.DataFrame([{'item': k, 'value': json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in summ.items()]).to_excel(xw, sheet_name='Summary', index=False)
    import openpyxl
    wb = openpyxl.load_workbook(f'{a.out}/APX_{TAG}_{TESTTAG}_Data.xlsx')
    for ws in wb.worksheets:
        for col in ws.iter_cols(min_row=1, max_row=1):
            ws.column_dimensions[col[0].column_letter].width = 24
    wb.save(f'{a.out}/APX_{TAG}_{TESTTAG}_Data.xlsx')
    print('accel outputs written to', a.out)

elif a.mode == 'cod':
    df = pd.read_csv(f'{A}/per_frame.csv'); st = pd.read_csv(f'{A}/steps.csv'); summ = json.load(open(f'{A}/summary.json'))
    view = summ['view']; VW = 'Side' if view == 'side' else 'Front'; plant_leg = summ['turn']['plant_leg_given']
    hold_at = {int(r.td_frame): r for r in st.itertuples()}
    OW, OH = 1280, 720; raw = f'{a.out}/_raw.mp4'; vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (OW, OH)); hold = None
    sc_in = min(520 / W, 330 / H); iw, ih = int(W * sc_in), int(H * sc_in)
    for i in range(N):
        cv_ = np.zeros((OH, OW, 3), np.uint8); cv_[:] = bgr(BG)
        x0, y0 = crop_box(i); z = cv2.resize(frames[i][y0:y0 + CS, x0:x0 + CS], (OH, OH), interpolation=cv2.INTER_CUBIC)
        if valid[i]: draw_skel(z, K[i], OH / CS, (x0, y0), 3)
        cv_[:, :OH] = z
        full = frames[i].copy(); cv2.rectangle(full, (x0, y0), (x0 + CS, y0 + CS), bgr(FLUORO), max(2, W // 300))
        cv_[20:20 + ih, OH + 20:OH + 20 + iw] = cv2.resize(full, (iw, ih))
        r = df.iloc[i]; X = OH + 24; Y = 20 + ih + 24; items = [(f'{a.name.upper()}  |  {a.test.upper()}', (X, Y), 22, FLUORO)]; Y += 44
        if view == 'side': items.append((f'Trunk lean   {r.trunk_lean_deg:3.0f}°', (X, Y), 26, TRUNK))
        Y += 44
        if i in hold_at: hold = (hold_at[i], 18)
        if hold:
            s_, n = hold
            items.append((f'{s_.role.upper()}', (X, Y), 26, FLUORO)); Y += 38
            l1 = (f'Hip height {s_.hip_height_min_pctL:.0f}% leg' if view == 'side' else '') + (f'   Knee bend {s_.knee_flex_max:.0f}°' if pd.notna(s_.knee_flex_max) else '')
            items.append((l1, (X, Y), 21, WHITE)); Y += 32
            if pd.notna(getattr(s_, 'knee_line_deg', np.nan)):
                items.append((f'Knee line {s_.knee_line_deg:+.0f}° (+ = inward, approx)', (X, Y), 21, WHITE)); Y += 32
            if pd.notna(s_.contact_ms): items.append((f'Contact ~{s_.contact_ms:.0f} ms', (X, Y), 21, WHITE))
            hold = (s_, n - 1) if n > 1 else None
        items.append(('Fluoro / white = the two legs & arms   Blue = trunk', (20, OH - 30), 15, WHITE))
        vw.write(put(cv_, items))
    vw.release(); encode(raw, f'{a.out}/APX_{TAG}_{TESTTAG}_{VW}_Overlay.mp4')
    pl = st[st.role == 'plant']; pn = st[st.role == 'penultimate']
    def cod_lines(s_):
        L1, L2 = [], []
        if view == 'side': L1.append(f'Hip height {s_.hip_height_min_pctL:.0f}% of leg')
        if pd.notna(s_.knee_flex_max): L1.append(f'Knee bend {s_.knee_flex_max:.0f}°')
        if pd.notna(getattr(s_, 'knee_line_deg', np.nan)): L1.append(f'Knee line {s_.knee_line_deg:+.0f}°')
        if pd.notna(getattr(s_, 'trunk_deepest', np.nan)): L2.append(f'Trunk {s_.trunk_deepest:.0f}°')
        if pd.notna(getattr(s_, 'trunk_side_lean_deg', np.nan)): L2.append(f'Side lean {s_.trunk_side_lean_deg:.0f}°')
        if pd.notna(s_.contact_ms): L2.append(f'Contact ~{s_.contact_ms:.0f} ms')
        if not L1: L1 = [f'Body {s_.body_to_camera} to this camera: angles not measured']
        return ['   '.join(L1), '   '.join(L2) if L2 else 'Approx values, coach to confirm on video', f'Approx. Drawn: stance leg ({s_.leg_tracked}) and trunk. Knee line + = inward']
    side_word = a.test.split()[-1]
    for nm, sel_, ttl in ((('Penultimate_Step', pn, 'Penultimate Step'), ('Plant', pl, 'Plant (deepest point)')) if view == 'side' else
                          (('Front_Penultimate', pn, 'Front-On Penultimate Step'), ('Front_Plant', pl, 'Front-On Plant'))):
        if not len(sel_): print('WARNING: no', nm, 'contact found; still skipped'); continue
        s_ = sel_.iloc[0]
        for ann in (True, False):
            still(int(s_.deepest_frame), ann, f'505 {side_word} · {ttl}', cod_lines(s_), f'{a.out}/APX_{TAG}_{TESTTAG}_{nm}_{"annotated" if ann else "clean"}.jpg',
                  knee_line=(s_.leg_tracked if pd.notna(getattr(s_, 'knee_line_deg', np.nan)) else None), only_leg=s_.leg_tracked)
    nice = st.rename(columns={'contact': 'Contact', 'role': 'Role', 'td_frame': 'Touchdown frame', 'to_frame': 'Toe-off frame', 'deepest_frame': 'Deepest frame',
        'leg_tracked': 'Leg (pose model, check)', 'contact_ms': 'Ground contact (ms, approx)', 'hip_height_TD_pctL': 'Hip height at touchdown (% leg)',
        'hip_height_min_pctL': 'Lowest hip height (% leg)', 'knee_TD': 'Knee angle at touchdown (°)', 'knee_flex_max': 'Max knee bend (°)',
        'foot_vs_hip_pctL': 'Foot vs hip at touchdown (% leg, + = ahead)', 'shin_TD': 'Shin at touchdown (°, + = knee ahead of ankle)',
        'trunk_TD': 'Trunk at touchdown (°, + = forward)', 'trunk_deepest': 'Trunk at deepest point (°)', 'step_length_legs': 'Step length (leg lengths)',
        'facing_camera_pctL': 'Hip width in view (% leg; low = side-on)', 'knee_inward_pctL': 'Knee inside hip-ankle line (% leg)',
        'knee_line_deg': 'Knee line (°, + = inward, approx)', 'trunk_side_lean_deg': 'Trunk side lean (°)', 'foot_width_pctL': 'Foot out from hip (% leg)',
        'measurable': 'Knee line measurable'}).drop(columns=['timing_ok'], errors='ignore')
    xl = f'{a.out}/APX_{TAG}_{TESTTAG}_{VW}_Data.xlsx'
    with pd.ExcelWriter(xl) as xw:
        nice.to_excel(xw, sheet_name='Contacts', index=False)
        flat = []
        def walk(pre, v):
            if isinstance(v, dict):
                for k_, v_ in v.items(): walk(f'{pre}.{k_}' if pre else k_, v_)
            else: flat.append({'item': pre, 'value': json.dumps(v) if isinstance(v, list) else v})
        walk('', summ); pd.DataFrame(flat).to_excel(xw, sheet_name='Summary', index=False)
    import openpyxl
    wb = openpyxl.load_workbook(xl)
    for ws in wb.worksheets:
        for col in ws.iter_cols(min_row=1, max_row=1): ws.column_dimensions[col[0].column_letter].width = 26
    wb.save(xl); print('505 outputs written to', a.out)

elif a.mode == 'plain':
    OW = 1280; OH = 720
    raw = f'{a.out}/_raw.mp4'; vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (OW, OH))
    for i in range(N):
        f = frames[i].copy()
        if valid[i]: draw_skel(f, K[i], 1.0, (0, 0), max(2, W // 500))
        s = min(OW / W, OH / H); f = cv2.resize(f, (int(W * s), int(H * s)))
        c = np.zeros((OH, OW, 3), np.uint8); c[:] = bgr(BG); oy, ox = (OH - f.shape[0]) // 2, (OW - f.shape[1]) // 2
        c[oy:oy + f.shape[0], ox:ox + f.shape[1]] = f
        vw.write(put(c, [(f'{a.name.upper()}  |  {a.label.upper()}', (20, 16), 22, FLUORO)]))
    vw.release(); encode(raw, f'{a.out}/APX_{TAG}_{a.label.replace(" ", "_")}_Overlay.mp4')
    # candidate stills: athlete big enough, fully in frame, two legs tracked separately
    good = []
    for i in range(N):
        if not valid[i]: continue
        k = K[i]; h = spanf[i]
        if np.nanmin(k[:, 1]) < 0.01 * H or np.nanmax(k[:, 1]) > 0.99 * H or np.nanmin(k[:, 0]) < 0.01 * W or np.nanmax(k[:, 0]) > 0.99 * W: continue
        if h < 0.45 * np.nanmax(spanf): continue
        if np.linalg.norm(k[13] - k[14]) < 0.06 * h and np.linalg.norm(k[15] - k[16]) < 0.06 * h: continue  # both legs on one leg
        good.append(i)
    picks = [good[j] for j in np.linspace(0, len(good) - 1, min(12, len(good))).astype(int)] if good else list(np.linspace(0, N - 1, 12).astype(int))
    tiles = []
    for i in picks:
        size = int(np.clip(spanf[i] * 1.5, 80, min(W, H))); x0, y0 = crop_box(i, size)
        z = cv2.resize(frames[i][y0:y0 + size, x0:x0 + size], (320, 320))
        if valid[i]: draw_skel(z, K[i], 320 / size, (x0, y0), 2)
        tiles.append(put(z, [(f'frame {i}', (8, 6), 18, FLUORO)]))
    while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f'{a.work}/pick_frames.jpg', np.vstack([np.hstack(tiles[r:r + 6]) for r in (0, 6)]))
    print(f'{len(good)} usable frames for a still (athlete big enough, in frame, legs tracked separately)')
    print('plain overlay written; choose a still from', f'{a.work}/pick_frames.jpg')

elif a.mode == 'still':
    for ann in (True, False):
        still(a.frame, ann, a.slot, [a.caption] if a.caption else [], trunk_only=a.trunk_only, fname= f'{a.out}/APX_{TAG}_{a.slot.replace(" ", "_").replace("·", "").replace("__", "_")}_{"annotated" if ann else "clean"}.jpg')
    print('still written')

elif a.mode == 'reel':
    # One report image per pillar: key stills (with captions) on top, every touchdown below.
    RW = 1600; st = pd.read_csv(f'{A}/steps.csv')
    def bar(h, items):
        b = np.zeros((h, RW, 3), np.uint8); b[:] = bgr(BG); return put(b, items)
    parts = [bar(84, [(f'{a.name.upper()}  |  {a.test.upper()}', (24, 22), 34, FLUORO), ('APX PERFORMANCE LAB', (RW - 330, 30), 24, GREY)])]
    keys = []
    for pth in a.stills:
        img = cv2.imread(pth)
        if img is None: print('WARNING: still not found, skipped:', pth); continue
        if os.path.exists(pth.replace('.jpg', '.json')): meta_ = json.load(open(pth.replace('.jpg', '.json')))
        else:
            print('WARNING: caption file missing for', pth, '(do not delete the .json files before building the reel). Using the file name.')
            meta_ = {'title': os.path.basename(pth).replace('_annotated.jpg', '').split('_10m_')[-1].replace('_', ' '), 'lines': []}
        kw = RW // 3; im = cv2.resize(img[:1080], (kw, kw), interpolation=cv2.INTER_AREA)
        cap_ = np.zeros((120, kw, 3), np.uint8); cap_[:] = bgr(BG)
        items = [(meta_['title'].split('·')[-1].strip().upper(), (14, 10), 24, FLUORO)] + [(ln, (14, 46 + 34 * j), 19, WHITE) for j, ln in enumerate(meta_['lines'][:2])]
        keys.append(np.vstack([im, put(cap_, items)]))
    if keys:
        parts.append(bar(50, [('KEY POSITIONS', (24, 12), 24, GREY)]))
        row = np.zeros((keys[0].shape[0], RW, 3), np.uint8); row[:] = bgr(BG)
        x = (RW - len(keys) * (RW // 3)) // 2
        for k_ in keys: row[:, x:x + k_.shape[1]] = k_; x += k_.shape[1]
        parts.append(row)
    tiles = cod_tiles(st) if 'role' in st else touchdown_tiles(st, a.dist)
    if tiles:
        parts.append(bar(50, [('FOOT CONTACTS INTO AND OUT OF THE TURN  (deepest point of each contact)' if 'role' in st else 'EVERY TOUCHDOWN  (side-on, measured at foot contact)', (24, 12), 24, GREY)]))
        for r0 in range(0, len(tiles), 4):
            row = tiles[r0:r0 + 4]
            while len(row) < 4: row.append(np.full_like(tiles[0], BG[0]))
            parts.append(np.hstack(row))
    parts.append(bar(46, [(('Blue = trunk   Hip height = hip above the foot, % of leg length   Knee line: + = knee inside the hip-ankle line   All approx' if 'role' in st else 'Fluoro / white = the two legs and arms   Blue = trunk   Shin: + = knee ahead of ankle   Foot vs hip: + = foot lands ahead'), (24, 12), 18, GREY)]))
    out = f'{a.out}/APX_{TAG}_{TESTTAG}_Report_Reel.jpg'
    cv2.imwrite(out, np.vstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('reel written', out, np.vstack(parts).shape)

elif a.mode == 'video':
    # One compressed video per pillar for embedding in the HTML report: clips played back to back.
    miss = [p_ for p_ in a.inputs if not os.path.exists(p_)]
    if miss: print('WARNING: input not found, skipped:', miss); a.inputs = [p_ for p_ in a.inputs if os.path.exists(p_)]
    lst = f'{a.out}/_concat.txt'
    open(lst, 'w').write(''.join(f"file '{os.path.abspath(p)}'\n" for p in a.inputs))
    out = f'{a.out}/APX_{TAG}_{TESTTAG}_Report_Video.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-vf', 'scale=1280:720,fps=30',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '30', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', out], check=True)
    os.remove(lst)
    crf = 30
    while os.path.getsize(out) > 4e6 and crf < 38:
        crf += 3; tmp = out.replace('.mp4', '_tmp.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', out, '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', tmp], check=True)
        os.replace(tmp, out)
    print('report video written', out, round(os.path.getsize(out) / 1e6, 2), 'MB')
