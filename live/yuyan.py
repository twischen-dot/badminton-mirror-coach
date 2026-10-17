"""羽眼现场流程：找击球候选 → 骨骼识别 → 和示范动作在击球瞬间对齐 → 并排慢放出片。

用法：
  python live/yuyan.py hits  VIDEO [--sheet out.png]            # 列出击球声候选，并生成候选帧联系表供人工挑选
  python live/yuyan.py render --player VIDEO --hit 4.07 \
         --ref REF.mp4 --ref-hit 2.10 --name 球友A [--observation ...] [--drill ...]
                                                                 # 生成 sessions/<id>/compare.mp4 + result.json
不传 --hit 时取最强的击球声候选，并在结果里标注“未人工确认”。
"""
import argparse, json, os, secrets, string, subprocess, sys, tempfile, time, tomllib, wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("YUYAN_MODEL", Path.home() / "badminton-ai/pose_landmarker_heavy.task"))
FONT = os.environ.get("YUYAN_FONT", "/System/Library/Fonts/Hiragino Sans GB.ttc")
FF = os.environ.get("BADMINTON_FFMPEG", "/opt/homebrew/bin/ffmpeg")
W, H, FPS = 1920, 1080, 30
PRE, POST, SLOW, HOLD = 1.0, 0.5, 4, 0.8      # 击球前 1 秒、后 0.5 秒，慢放 4 倍，击球帧停 0.8 秒
CLIP_H = 720                                    # 分析用的片段统一缩放到 720 高
EV = dict(highpass_hz=2500, onset_db=12, min_gap_s=0.2)   # 与 pipeline/config.toml [events] 一致
# 身体主要连线（MediaPipe 33 点编号）：肩、臂、躯干、腿
CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28)]


def font(sz, _c={}):
    if sz not in _c: _c[sz] = ImageFont.truetype(FONT, sz)
    return _c[sz]


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def new_session_id():
    return "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(6))


# ---------------------------------------------------------------- 击球声候选
def hit_candidates(video, tmp):
    wav = Path(tmp) / "a.wav"
    run(FF, "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "22050", wav)
    w = wave.open(str(wav)); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    if len(x) < sr: return []
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / sr) < EV["highpass_hz"]] = 0
    h = np.fft.irfft(X, len(x)); hop = int(sr * 0.005)
    env = np.sqrt(np.mean(h[: len(h) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    db = 20 * np.log10(env + 1e-9)
    onset = db - np.convolve(db, np.ones(200) / 200, mode="same")
    half = int(EV["min_gap_s"] / 0.005)
    return [(round(i * 0.005, 3), round(float(onset[i]), 1)) for i in range(1, len(onset) - 1)
            if onset[i] > EV["onset_db"] and onset[i] == onset[max(0, i - half):i + half].max()]


def contact_sheet(video, cands, out):
    cap = cv2.VideoCapture(str(video)); tiles = []
    for k, (t, v) in enumerate(cands):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        if not ok: continue
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); im.thumbnail((360, 360))
        d = ImageDraw.Draw(im); d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
        d.text((6, 4), f"#{k}  {t:.2f}s  {v:.0f}dB", font=font(22), fill=(255, 210, 0)); tiles.append(im)
    if not tiles: return False
    cols = 4; tw = max(i.width for i in tiles); th = max(i.height for i in tiles)
    sheet = Image.new("RGB", (cols * tw, -(-len(tiles) // cols) * th), (20, 20, 20))
    for k, im in enumerate(tiles): sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
    sheet.save(out); return True


# ---------------------------------------------------------------- 片段与骨骼
def load_clip(video, hit, tmp, tag):
    """取击球前 PRE 秒到后 POST 秒，按 30fps、720 高解码（ffmpeg 会按手机旋转标记自动转正）。"""
    t0 = max(0.0, hit - PRE); mp4 = Path(tmp) / f"{tag}.mp4"
    run(FF, "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{PRE + POST:.3f}", "-i", video,
        "-vf", f"fps={FPS},scale=-2:{CLIP_H}", "-an", "-c:v", "libx264", "-crf", "16", mp4)
    cap = cv2.VideoCapture(str(mp4)); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    hit_i = min(len(frames) - 1, int(round((hit - t0) * FPS)))
    return frames, hit_i


_LM = {"lm": None, "ts": 0}


def detect(frames):
    """整个进程只建一个检测器并一直复用：MediaPipe 的 close() 在本机会卡约 60 秒，所以不关闭，进程结束时直接退出。"""
    import mediapipe as mp
    if _LM["lm"] is None:
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=2,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        _LM["lm"] = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    lm = _LM["lm"]; out = []; fh, fw = frames[0].shape[:2]
    _LM["ts"] += 10000                      # VIDEO 模式要求时间戳递增；换片段时跳开 10 秒
    for i, f in enumerate(frames):
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)),
                                _LM["ts"] + int(i * 1000 / FPS))
        if not r.pose_landmarks: out.append(None); continue
        # 多人时取画面里最高大的人（离机位最近的体验者）
        p = max(r.pose_landmarks, key=lambda q: max(a.y for a in q) - min(a.y for a in q))
        out.append([(a.x * fw, a.y * fh, a.visibility) for a in p])
    _LM["ts"] += int(len(frames) * 1000 / FPS)
    return out


def crop_box(frames, lms):
    pts = np.array([p[:2] for L in lms if L for p in L if p[2] > 0.5])
    fh, fw = frames[0].shape[:2]
    if len(pts) < 10: s = min(fh, fw); return ((fw - s) // 2, (fh - s) // 2, s)
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    span = max(np.ptp(np.percentile(pts[:, 0], [2, 98])), np.ptp(np.percentile(pts[:, 1], [2, 98]))) * 1.35
    s = int(min(max(span, 300), fh, fw))
    return (int(np.clip(cx - s / 2, 0, fw - s)), int(np.clip(cy - s / 2, 0, fh - s)), s)


def metrics_at(lms, hit_i, handed):
    """击球帧前后 2 帧里取可见度最好的一帧，算两项二维画面估计。"""
    sh, el, wr, hp = (12, 14, 16, 24) if handed == "right" else (11, 13, 15, 23)
    best = None
    for i in range(max(0, hit_i - 2), min(len(lms), hit_i + 3)):
        L = lms[i]
        if L and min(L[sh][2], L[el][2], L[hp][2]) > 0.5:
            vis = L[wr][2]
            if best is None or vis > best[0]: best = (vis, L)
    if not best: return {}
    _, L = best
    torso = np.hypot(L[sh][0] - L[hp][0], L[sh][1] - L[hp][1]) or 1
    m = {"elbow_above_shoulder_pct_torso": round((L[sh][1] - L[el][1]) / torso * 100)}
    if L[wr][2] > 0.5:
        a = np.array(L[sh][:2]) - L[el][:2]; b = np.array(L[wr][:2]) - L[el][:2]
        m["elbow_angle_deg"] = round(float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))))
    return m


# ---------------------------------------------------------------- 渲染
def panel(clip, i, size, hit):
    f, L, (bx, by, s) = clip["frames"][i], clip["lms"][i], clip["box"]
    img = Image.fromarray(f[by:by + s, bx:bx + s]).resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(img); k = size / s
    if L:
        for a, b in CONN:
            if L[a][2] > 0.5 and L[b][2] > 0.5:
                d.line([((L[a][0] - bx) * k, (L[a][1] - by) * k), ((L[b][0] - bx) * k, (L[b][1] - by) * k)],
                       fill=(0, 255, 200), width=max(3, size // 250))
    if hit:
        d.rectangle([0, 0, size - 1, size - 1], outline=(255, 210, 0), width=8)
        d.text((size // 2, size - 40), "击球瞬间", font=font(44), fill=(255, 210, 0), anchor="mm")
    return img


def canvas(title, sub, footer):
    im = Image.new("RGB", (W, H), (18, 20, 24)); d = ImageDraw.Draw(im)
    d.text((60, 36), title, font=font(46), fill="white")
    if sub: d.text((60, 98), sub, font=font(28), fill=(170, 175, 185))
    d.text((60, H - 36), footer, font=font(24), fill=(120, 125, 135), anchor="ls")
    d.text((W - 60, H - 36), "羽眼", font=font(26), fill=(120, 125, 135), anchor="rs")
    return im


def aligned(me, ref):
    """以击球帧对齐：两段都从击球前 n 帧取到击球后 m 帧，缺的帧用首/末帧补齐。"""
    n = min(me["hit_i"], ref["hit_i"]); m = min(len(me["frames"]) - me["hit_i"], len(ref["frames"]) - ref["hit_i"])
    return [(me["hit_i"] + k, ref["hit_i"] + k, k == 0) for k in range(-n, m)]


def render(me, ref, out, title, footer, passes):
    sz = 820; enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                                      "-preset", "fast", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    def emit(im, k):
        b = im.tobytes()
        for _ in range(k): enc.stdin.write(b)
    im = canvas(title, None, footer); d = ImageDraw.Draw(im)
    d.text((W // 2, 470), "你和示范，同一个动作", font=font(88), fill="white", anchor="mm")
    d.text((W // 2, 590), "击球瞬间对齐 · 慢放 4 倍 · AI 骨骼识别", font=font(40), fill=(200, 205, 215), anchor="mm")
    emit(im, int(1.5 * FPS)); still = None
    for p in range(1, passes + 1):
        for i, j, hit in aligned(me, ref):
            im = canvas(title, f"第 {p}/{passes} 遍", footer); d = ImageDraw.Draw(im)
            for col, (c, idx) in enumerate(((me, i), (ref, j))):
                x = 100 + col * (sz + 140)
                im.paste(panel(c, idx, sz, hit), (x, 165))
                d.text((x + sz // 2, 158), c["label"], font=font(36), fill="white", anchor="mb")
            if hit and still is None: still = im.copy()
            emit(im, int(HOLD * FPS) if hit else SLOW)
    enc.stdin.close()
    if enc.wait() != 0: raise RuntimeError("ffmpeg 编码失败")
    return still


def draft_observation(mm, mr, handed):
    a, b = mm.get("elbow_above_shoulder_pct_torso"), mr.get("elbow_above_shoulder_pct_torso")
    if a is None or b is None: return "这一项暂时无法判断：击球瞬间肩、肘或髋没有被稳定识别。"
    side = "右" if handed == "right" else "左"
    if a < b - 10:
        return f"击球瞬间，你的{side}肘相对肩的高度比示范低（画面估计：你 {a}%、示范 {b}% 躯干长）。"
    if a > b + 10:
        return f"击球瞬间，你的{side}肘相对肩的高度比示范高（画面估计：你 {a}%、示范 {b}% 躯干长）。"
    return f"击球瞬间，你的{side}肘高度和示范接近（画面估计：你 {a}%、示范 {b}% 躯干长）。"


def cmd_render(a):
    t_start = time.time(); sid = new_session_id(); sdir = Path(a.sessions) / sid; sdir.mkdir(parents=True)
    timing = {}
    with tempfile.TemporaryDirectory() as tmp:
        hit, hit_src = a.hit, "人工指定"
        if hit is None:
            c = hit_candidates(a.player, tmp)
            if not c: sys.exit("没有找到击球声候选，请用 --hit 手动指定击球时刻")
            hit, hit_src = max(c, key=lambda x: x[1])[0], "击球声自动推荐（未人工确认）"
        clips = {}
        for tag, path, h, label in (("me", a.player, hit, f"你（{a.name}）"), ("ref", a.ref, a.ref_hit, a.ref_label)):
            t = time.time(); frames, hi = load_clip(path, h, tmp, tag); timing[f"decode_{tag}"] = round(time.time() - t, 1)
            if not frames: sys.exit(f"{path} 在 {h}s 附近读不到画面")
            t = time.time(); lms = detect(frames); timing[f"pose_{tag}"] = round(time.time() - t, 1)
            clips[tag] = dict(frames=frames, lms=lms, hit_i=hi, box=crop_box(frames, lms), label=label,
                              pose_ok=sum(L is not None for L in lms), n=len(frames))
        mm = metrics_at(clips["me"]["lms"], clips["me"]["hit_i"], a.handed)
        mr = metrics_at(clips["ref"]["lms"], clips["ref"]["hit_i"], a.ref_handed)
        footer = f"骨骼为 AI 识别；击球时刻：{hit_src}；示范：{a.ref_source}"
        t = time.time()
        still = render(clips["me"], clips["ref"], sdir / "compare.mp4", f"{a.name} · {a.stroke}", footer, a.passes)
        timing["render"] = round(time.time() - t, 1)
        if still: still.save(sdir / "hitframe.jpg", quality=88)
    metrics = []
    if "elbow_above_shoulder_pct_torso" in mm and "elbow_above_shoulder_pct_torso" in mr:
        metrics.append({"name": "击球瞬间肘部高于肩部（占躯干长，画面估计）",
                        "you": mm["elbow_above_shoulder_pct_torso"], "ref": mr["elbow_above_shoulder_pct_torso"], "unit": "%"})
    if "elbow_angle_deg" in mm and "elbow_angle_deg" in mr:
        metrics.append({"name": "击球瞬间肘角（画面估计）", "you": mm["elbow_angle_deg"], "ref": mr["elbow_angle_deg"], "unit": "°"})
    reviewed = bool(a.observation)
    result = dict(session_id=sid, player=a.name, date=datetime.now().strftime("%Y-%m-%d %H:%M"), stroke=a.stroke,
                  source=a.source, reference=a.ref_source,
                  observation=a.observation or draft_observation(mm, mr, a.handed),
                  drill=a.drill or "【待复核后填写】", retest=a.retest,
                  reviewed_by="球友复核" if reviewed else "AI 草稿，未复核",
                  metrics=metrics, consent=a.consent)
    (sdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    timing["total"] = round(time.time() - t_start, 1)
    info = dict(player_video=str(a.player), hit=hit, hit_source=hit_src, ref_video=str(a.ref), ref_hit=a.ref_hit,
                pose_frames={k: f"{v['pose_ok']}/{v['n']}" for k, v in clips.items()}, timing_s=timing)
    (sdir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(dict(session=str(sdir), **info, metrics=metrics), ensure_ascii=False, indent=2))


def cmd_hits(a):
    with tempfile.TemporaryDirectory() as tmp:
        c = hit_candidates(a.video, tmp)
    for k, (t, v) in enumerate(c): print(f"#{k}\t{t:.3f}s\t{v:.1f}dB")
    if not c: print("没有击球声候选（视频可能没有声音，或环境太吵）"); return
    if a.sheet and contact_sheet(a.video, c[: a.max], a.sheet): print("联系表:", a.sheet)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    h = sp.add_parser("hits"); h.add_argument("video"); h.add_argument("--sheet"); h.add_argument("--max", type=int, default=24)
    r = sp.add_parser("render")
    r.add_argument("--player", required=True); r.add_argument("--hit", type=float)
    r.add_argument("--ref", required=True); r.add_argument("--ref-hit", type=float, required=True)
    r.add_argument("--name", default="球友"); r.add_argument("--stroke", default="正手高远球")
    r.add_argument("--handed", default="right", choices=["right", "left"])
    r.add_argument("--ref-handed", default="right", choices=["right", "left"])
    r.add_argument("--ref-label", default="示范"); r.add_argument("--ref-source", default="自录示范")
    r.add_argument("--source", default="现场录制"); r.add_argument("--observation"); r.add_argument("--drill")
    r.add_argument("--retest", default="下次在相同机位再录一段同一动作。")
    r.add_argument("--consent", default="self", choices=["self", "screen", "public"])
    r.add_argument("--passes", type=int, default=3); r.add_argument("--sessions", default=str(ROOT / "sessions"))
    a = ap.parse_args()
    {"hits": cmd_hits, "render": cmd_render}[a.cmd](a)
    sys.stdout.flush(); os._exit(0)             # 跳过 MediaPipe 检测器的慢速 close()


if __name__ == "__main__":
    main()
