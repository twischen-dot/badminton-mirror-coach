"""流水线各步骤。每一步只读上一步的输出，写自己的目录；已完成的步骤会跳过。"""

import csv
import json
import math
import shutil
import subprocess
import tomllib
import wave

import cv2
import numpy as np

from common import (FFMPEG, FFPROBE, MODEL, CheckFailed, NeedsReview, log,
                    read_json, sh, stage_dir, write_json)

COURT_POINTS_HELP = """# 场地坐标约定（米）：只标近端（被分析球员）半场
#   x：从画面左侧的双打边线起算，0 → 6.10（单打边线在 0.46 和 5.64）
#   y：从球网起算，0 → 6.70（前发球线 1.98，双打后发球线 5.94，底线 6.70）
#   中线 x = 3.05（从前发球线到底线）
# 常用交点：
#   前发球线×左边线 (0, 1.98)   前发球线×中线 (3.05, 1.98)   前发球线×右边线 (6.10, 1.98)
#   双打后发球线×左边线 (0, 5.94)  ×中线 (3.05, 5.94)  ×右边线 (6.10, 5.94)
#   底线×左边线 (0, 6.70)  ×中线 (3.05, 6.70)  ×右边线 (6.10, 6.70)
# 至少 4 个点，最好 6 个以上，并且要同时包含靠网和靠底线的点。
"""


# ---------------------------------------------------------------- 00 质检
def s00_check(run, cfg, meta):
    out = stage_dir(run, "00-check")
    if FFPROBE:
        probe = json.loads(subprocess.run(
            [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(run / "input.mp4")],
            capture_output=True, text=True, check=True).stdout)
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        num, den = map(int, v["avg_frame_rate"].split("/"))
        info = {"width": int(v["width"]), "height": int(v["height"]), "fps": num / den if den else 0,
                "duration_s": float(probe["format"]["duration"]),
                "has_audio": any(s["codec_type"] == "audio" for s in probe["streams"])}
    else:  # 没有 ffprobe（例如 Linux 上只有 imageio-ffmpeg）
        cap = cv2.VideoCapture(str(run / "input.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps, "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0,
                "has_audio": "Audio:" in subprocess.run([FFMPEG, "-hide_banner", "-i", str(run / "input.mp4")],
                                                         capture_output=True, text=True).stderr}
    info["has_audio"] = info["has_audio"] or bool(meta.get("audio"))
    c = cfg["check"]
    errors, warnings = [], []
    if info["width"] < c["min_width"]:
        errors.append(f"分辨率宽 {info['width']} < {c['min_width']}")
    if info["fps"] < c["min_fps"]:
        errors.append(f"帧率 {info['fps']:.1f} < {c['min_fps']}")
    if not info["has_audio"]:
        errors.append("没有音轨：击球时刻检测需要声音")
    if info["fps"] < c["recommend_fps"]:
        warnings.append(f"帧率 {info['fps']:.0f}fps，低于建议的 {c['recommend_fps']}fps：挥拍瞬间手臂常识别不到")
    if info["duration_s"] > c["max_minutes"] * 60:
        warnings.append("视频过长，建议按局拆分")
    info.update(errors=errors, warnings=warnings)
    write_json(out / "check.json", info)
    for w in warnings:
        log(run, "  ⚠ " + w)
    if errors:
        raise CheckFailed("；".join(errors))


# ---------------------------------------------------------------- 01 预处理
def s01_prep(run, cfg, meta):
    out = stage_dir(run, "01-prep")
    src = run / meta["audio"] if meta.get("audio") else run / "input.mp4"
    sh(FFMPEG, "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "22050", out / "audio.wav")
    cap = cv2.VideoCapture(str(run / "input.mp4"))
    write_json(out / "video.json", {
        "fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})


# ---------------------------------------------------------------- 02 骨骼
CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
               (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
               (0, 11), (0, 12), (15, 19), (16, 20)]
COLORS = [(255, 220, 0), (0, 140, 255)]


def s02_pose(run, cfg, meta):
    import mediapipe as mp
    out = stage_dir(run, "02-pose")
    vid = read_json(run / "01-prep/video.json")
    pc = cfg["pose"]
    cap = cv2.VideoCapture(str(run / "input.mp4"))
    fps = vid["fps"]
    ow = min(pc["width"], vid["width"]) // 2 * 2
    oh = int(vid["height"] * ow / vid["width"]) // 2 * 2
    raw = out / "overlay_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=pc["num_poses"],
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    with open(out / "landmarks.csv", "w", newline="") as fh, \
            mp.tasks.vision.PoseLandmarker.create_from_options(opts) as lm:
        w = csv.writer(fh)
        w.writerow(["frame", "person", "landmark", "x", "y", "visibility"])
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (ow, oh))
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
            people = sorted(res.pose_landmarks or [], key=lambda p: -(p[23].y + p[24].y))  # 近的人在前
            for p, pts in enumerate(people):
                xy = [(int(q.x * ow), int(q.y * oh)) for q in pts]
                for a, b in CONNECTIONS:
                    if pts[a].visibility > pc["min_visibility"] and pts[b].visibility > pc["min_visibility"]:
                        cv2.line(frame, xy[a], xy[b], COLORS[p % 2], 3, cv2.LINE_AA)
                for k, q in enumerate(pts):
                    w.writerow([i, p, k, f"{q.x:.4f}", f"{q.y:.4f}", f"{q.visibility:.3f}"])
            cv2.putText(frame, f"{i / fps:6.1f}s", (14, oh - 14), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            i += 1
            if i % 600 == 0:
                log(run, f"  骨骼 {i}/{vid['frames']}")
    writer.release()
    sh(FFMPEG, "-loglevel", "error", "-y", "-i", raw, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", out / "overlay.mp4")
    raw.unlink()


# ---------------------------------------------------------------- 03 场地标定（复核点①）
def _detect_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 160)).astype(np.uint8) * 255
    green = cv2.dilate(cv2.inRange(hsv, (35, 50, 50), (95, 255, 255)), np.ones((25, 25), np.uint8))
    mask = cv2.bitwise_and(white, green)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, 80, minLineLength=100, maxLineGap=25)
    if segs is None:
        return []
    groups = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        a = math.atan2(y2 - y1, x2 - x1)
        a = a + math.pi if a < -math.pi / 2 else (a - math.pi if a > math.pi / 2 else a)
        r = -math.sin(a) * x1 + math.cos(a) * y1
        L = math.hypot(x2 - x1, y2 - y1)
        for g in groups:
            if abs(g["a"] - a) < 0.03 and abs(g["r"] - r) < 10:
                g["p"] += [(x1, y1), (x2, y2)]
                g["w"] += L
                break
        else:
            groups.append({"a": a, "r": r, "p": [(x1, y1), (x2, y2)], "w": L})
    lines = []
    for g in groups:
        if g["w"] > 250:
            vx, vy, x0, y0 = cv2.fitLine(np.float32(g["p"]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            lines.append((vx, vy, x0, y0))
    return lines


def s03_calib(run, cfg, meta):
    out = stage_dir(run, "03-calib")
    path = out / "calib.toml"
    if path.exists():
        with open(path, "rb") as fh:
            calib = tomllib.load(fh)
        if calib.get("confirmed"):
            if len(calib.get("points", [])) < 4:
                raise NeedsReview("① 场地标定：calib.toml 至少需要 4 个点")
            return
        raise NeedsReview(f"① 场地标定：请填写并确认 {path}")
    vid = read_json(run / "01-prep/video.json")
    ref = int(meta.get("calib_ref_frame", min(120, vid["frames"] - 1)))
    cap = cv2.VideoCapture(str(run / "input.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
    _, img = cap.read()
    cv2.imwrite(str(out / "ref.png"), img)
    lines = _detect_lines(img)
    vis = img.copy()
    cands = []
    for i, (vx, vy, x0, y0) in enumerate(lines):
        p1 = (int(x0 - vx * 3000), int(y0 - vy * 3000))
        p2 = (int(x0 + vx * 3000), int(y0 + vy * 3000))
        cv2.line(vis, p1, p2, (255, 0, 255), 1)
        for j in range(i + 1, len(lines)):
            ux, uy, u0x, u0y = lines[j]
            A = np.array([[vx, -ux], [vy, -uy]])
            if abs(np.linalg.det(A)) < 0.15:
                continue
            s, _ = np.linalg.solve(A, [u0x - x0, u0y - y0])
            px, py = x0 + s * vx, y0 + s * vy
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                cands.append((len(cands), round(float(px)), round(float(py))))
    for k, px, py in cands:
        cv2.circle(vis, (px, py), 7, (0, 255, 255), 2)
        cv2.putText(vis, f"P{k}", (px + 8, py - 8), 0, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out / "candidates.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    txt = [COURT_POINTS_HELP, f"ref_frame = {ref}", "confirmed = false",
           "# 自动找到的候选交点（画面像素），见 candidates.jpg：",
           *[f"#   P{k} = [{px}, {py}]" for k, px, py in cands],
           "# 把确认过的点按 [画面x, 画面y, 场地x, 场地y] 填进下面，然后把 confirmed 改成 true",
           "points = [", "]"]
    path.write_text("\n".join(txt) + "\n")
    raise NeedsReview(f"① 场地标定：已生成候选交点，请确认 {path}")


# ---------------------------------------------------------------- 04 场地坐标
def s04_court(run, cfg, meta):
    out = stage_dir(run, "04-court")
    with open(run / "03-calib/calib.toml", "rb") as fh:
        calib = tomllib.load(fh)
    P = np.float32(calib["points"])
    Hc, _ = cv2.findHomography(P[:, :2], P[:, 2:], 0)
    err = np.linalg.norm(cv2.perspectiveTransform(P[None, :, :2], Hc)[0] - P[:, 2:], axis=1)
    vid = read_json(run / "01-prep/video.json")
    W, H, fps = vid["width"], vid["height"], vid["fps"]
    cap = cv2.VideoCapture(str(run / "input.mp4"))
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bg = np.zeros((H, W), np.uint8)
    bg[: int(H * 0.65)] = 255  # 上方 65%：墙面、广告牌，作为静止背景

    def feats(f):
        return orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), bg)

    cap.set(cv2.CAP_PROP_POS_FRAMES, calib["ref_frame"])
    _, ref = cap.read()
    rk, rd = feats(ref)
    ank = {}
    for r in csv.DictReader(open(run / "02-pose/landmarks.csv")):
        if r["landmark"] in ("27", "28") and float(r["visibility"]) > cfg["pose"]["min_visibility"]:
            ank.setdefault(int(r["frame"]), {}).setdefault(int(r["person"]), {})[r["landmark"]] = (
                float(r["x"]) * W, float(r["y"]) * H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows, prev, last_ok, i = [], None, -10 ** 9, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        best = None
        k, d = feats(f)
        if d is not None and len(k) > 30:
            m = sorted(bf.match(d, rd), key=lambda x: x.distance)[:600]
            if len(m) >= 30:
                Ht, inl = cv2.findHomography(np.float32([k[x.queryIdx].pt for x in m]),
                                             np.float32([rk[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
                if Ht is not None and inl.sum() >= 25:
                    for a in ank.get(i, {}).values():
                        if "27" in a and "28" in a:
                            p = np.float32([[[(a["27"][0] + a["28"][0]) / 2, (a["27"][1] + a["28"][1]) / 2]]])
                            x, y = cv2.perspectiveTransform(cv2.perspectiveTransform(p, Ht), Hc)[0, 0]
                            if -0.8 < x < 6.9 and -0.3 < y < 7.6:  # 近端半场
                                dd = 0 if prev is None else math.hypot(x - prev[0], y - prev[1])
                                if best is None or dd < best[2]:
                                    best = (float(x), float(y), dd)
        # 连续性：允许的跳动随失联时间增长（人最快约 6 m/s），避免一次误判后连锁拒绝正常帧
        if best and (prev is None or best[2] < 1.5 + 6.0 * (i - last_ok) / fps):
            prev, last_ok = best[:2], i
            rows.append([i, f"{i / fps:.3f}", f"{best[0]:.3f}", f"{best[1]:.3f}", 1])
        else:
            rows.append([i, f"{i / fps:.3f}", "", "", 0])
        i += 1
    with open(out / "court.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t", "x_m", "y_m", "ok"])
        w.writerows(rows)
    write_json(out / "court.json", {"reproj_err_mean_m": float(err.mean()), "reproj_err_max_m": float(err.max()),
                                    "tracked_ratio": sum(r[4] for r in rows) / max(len(rows), 1)})
    log(run, f"  标定误差 {err.mean():.3f}m，定位成功 {sum(r[4] for r in rows)}/{len(rows)} 帧")


# ---------------------------------------------------------------- 05 事件检测
def s05_events(run, cfg, meta):
    out = stage_dir(run, "05-events")
    ec = cfg["events"]
    w = wave.open(str(run / "01-prep/audio.wav"))
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / sr) < ec["highpass_hz"]] = 0
    h = np.fft.irfft(X, len(x))
    hop = int(sr * 0.005)
    env = np.sqrt(np.mean(h[: len(h) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    db = 20 * np.log10(env + 1e-9)
    onset = db - np.convolve(db, np.ones(200) / 200, mode="same")
    np.save(out / "onset.npy", onset)
    half = int(ec["min_gap_s"] / 0.005)
    cands = [(round(i * 0.005, 3), round(float(onset[i]), 1)) for i in range(1, len(onset) - 1)
             if onset[i] > ec["onset_db"] and onset[i] == onset[max(0, i - half):i + half].max()]
    # 回合切分：强击球声间隔不超过 rally_gap_s
    strong = [t for t, v in cands if v >= ec["rally_db"]]
    rallies, cur = [], []
    for t in strong:
        if cur and t - cur[-1] > ec["rally_gap_s"]:
            rallies.append(cur)
            cur = []
        cur.append(t)
    if cur:
        rallies.append(cur)
    rallies = [(r[0] - 0.5, r[-1] + 0.6) for r in rallies if len(r) >= ec["rally_min_hits"]]
    with open(out / "candidates.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["k", "t", "audio_db", "rally"])
        for k, (t, v) in enumerate(cands):
            rid = next((i + 1 for i, (a, b) in enumerate(rallies) if a <= t <= b), 0)
            wr.writerow([k, t, v, rid])
    write_json(out / "rallies.json", [{"id": i + 1, "start": a, "end": b} for i, (a, b) in enumerate(rallies)])
    _review_strips(run, out, [(k, t) for k, (t, v) in enumerate(cands)
                              if any(a <= t <= b for a, b in rallies)])
    log(run, f"  候选击球 {len(cands)} 个，回合 {len(rallies)} 个")


def _review_strips(run, out, cands):
    """每个候选：近端球员放大图，t-0.13 / t / t+0.13 三帧，供逐拍复核。"""
    vid = read_json(run / "01-prep/video.json")
    W, H, fps = vid["width"], vid["height"], vid["fps"]
    hips = {}
    for r in csv.DictReader(open(run / "02-pose/landmarks.csv")):
        if r["person"] == "0" and r["landmark"] in ("23", "24"):
            hips.setdefault(int(r["frame"]), []).append((float(r["x"]) * W, float(r["y"]) * H))
    cap = cv2.VideoCapture(str(run / "input.mp4"))
    sdir = out / "strips"
    sdir.mkdir(exist_ok=True)
    s = int(H * 0.48)
    for k, t in cands:
        f0 = int(round(t * fps))
        c = next((hips[f] for d in range(15) for f in (f0 - d, f0 + d) if len(hips.get(f, [])) == 2), None)
        cx, cy = (np.mean(c, axis=0) if c else (W / 2, H * 0.65))
        x0 = int(np.clip(cx - s / 2, 0, W - s))
        y0 = int(np.clip(cy - s * 0.55, 0, H - s))
        tiles = []
        for d in (-0.13, 0, 0.13):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((t + d) * fps))))
            ok, f = cap.read()
            tiles.append(cv2.resize(f[y0:y0 + s, x0:x0 + s], (240, 240)) if ok else np.zeros((240, 240, 3), np.uint8))
        cv2.imwrite(str(sdir / f"{k:04d}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- 06 逐拍复核（复核点②）
LABELS_HELP = "label: me=本方击球 / opp=对手击球 / none=不是击球 / ?=未判断"


def s06_review(run, cfg, meta):
    out = stage_dir(run, "06-review")
    labels = out / "labels.csv"
    if (out / "DONE").exists() and labels.exists():
        return
    if not labels.exists():
        with open(run / "05-events/candidates.csv") as fh, open(labels, "w", newline="") as oh:
            w = csv.writer(oh)
            w.writerow(["k", "t", "rally", "label", "stroke", "note"])
            for r in csv.DictReader(fh):
                if r["rally"] != "0":
                    w.writerow([r["k"], r["t"], r["rally"], "?", "", ""])
    raise NeedsReview(f"② 逐拍复核：请在 {labels} 标注每个候选（{LABELS_HELP}），完成后创建 06-review/DONE")


# ---------------------------------------------------------------- 07 指标与图表
def _smooth(v, k):
    v = v.copy()
    idx = np.arange(len(v))
    good = ~np.isnan(v)
    if good.sum() < 2:
        return v
    v[~good] = np.interp(idx[~good], idx[good], v[good])
    pad = np.pad(v, k // 2, mode="edge")
    med = np.array([np.median(pad[i:i + k]) for i in range(len(v))])
    return np.convolve(np.pad(med, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def s07_metrics(run, cfg, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from pathlib import Path as _P
    import os
    font = os.environ.get("BADMINTON_FONT")
    font = font if font and _P(font).exists() else None
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()

    out = stage_dir(run, "07-metrics")
    mc = cfg["metrics"]
    fps = read_json(run / "01-prep/video.json")["fps"]
    T, X, Y = [], [], []
    for r in csv.DictReader(open(run / "04-court/court.csv")):
        T.append(float(r["t"]))
        X.append(float(r["x_m"]) if r["ok"] == "1" else np.nan)
        Y.append(float(r["y_m"]) if r["ok"] == "1" else np.nan)
    T, X, Y = map(np.array, (T, X, Y))
    Xs, Ys = _smooth(X, mc["smooth_frames"]), _smooth(Y, mc["smooth_frames"])
    speed = np.hypot(np.gradient(Xs), np.gradient(Ys)) * fps
    labels = list(csv.DictReader(open(run / "06-review/labels.csv")))
    rallies = read_json(run / "05-events/rallies.json")
    onset = np.load(run / "05-events/onset.npy")
    cand = [(float(r["t"]), float(r["audio_db"])) for r in csv.DictReader(open(run / "05-events/candidates.csv"))]

    inr = np.zeros(len(T), bool)
    for r in rallies:
        inr |= (T >= r["start"]) & (T <= r["end"])
    base = np.array([np.nanmedian(Xs[inr]), np.nanmedian(Ys[inr])])

    def at(t):
        # 击球时刻的位置：用原始坐标前后 3 帧的中位数。平滑轨迹会滞后，快速移动时可偏差近 1 米
        i = int(np.clip(round(t * fps), 0, len(T) - 1))
        xs, ys = X[max(0, i - 3):i + 4], Y[max(0, i - 3):i + 4]
        if np.sum(~np.isnan(xs)) >= 3:
            return float(np.nanmedian(xs)), float(np.nanmedian(ys))
        return float(Xs[i]), float(Ys[i])

    rows = []
    for rally in rallies:
        mine = sorted((float(l["t"]), l) for l in labels if l["label"] == "me" and l["rally"] == str(rally["id"]))
        opp_l = sorted(float(l["t"]) for l in labels if l["label"] == "opp" and l["rally"] == str(rally["id"]))
        for j, (t, l) in enumerate(mine):
            x, y = at(t)
            dist = math.hypot(x - base[0], y - base[1])
            zone = "前场" if y < mc["front_y_m"] else ("中场" if y < mc["rear_y_m"] else "后场")
            prev_me = mine[j - 1][0] if j else None
            # 对手击球：优先用人工标注；没有就取两次本方击球之间最强的击球声
            if prev_me is not None:
                lab = [o for o in opp_l if prev_me < o < t]
                if lab:
                    prev_opp = lab[-1]
                else:
                    c = [(ct, v) for ct, v in cand if prev_me + 0.3 < ct < t - 0.3]
                    prev_opp = max(c, key=lambda p: p[1])[0] if c else None
            else:
                prev_opp = None
            avail = (t - prev_opp) if (prev_opp and t - prev_me <= mc["avail_max_gap_s"]) else None
            nxt = mine[j + 1][0] if j + 1 < len(mine) else t + 2.5
            rec = None
            if dist >= mc["recover_radius_m"]:
                rec = float("nan")
                for i in range(int(t * fps), min(int(nxt * fps), len(T))):
                    if math.hypot(Xs[i] - base[0], Ys[i] - base[1]) < mc["recover_radius_m"]:
                        rec = T[i] - t
                        break
            rows.append(dict(rally=rally["id"], t=t, stroke=l["stroke"], note=l["note"], zone=zone,
                             x=x, y=y, dist=dist, recover=rec, avail=avail))
    with open(out / "shots.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rally", "t", "stroke", "zone", "x_m", "y_m", "dist_from_base_m", "recover_s",
                    "time_available_s", "note"])
        for r in rows:
            w.writerow([r["rally"], r["t"], r["stroke"], r["zone"], f"{r['x']:.2f}", f"{r['y']:.2f}",
                        f"{r['dist']:.2f}", "" if r["recover"] is None else f"{r['recover']:.2f}",
                        "" if r["avail"] is None else f"{r['avail']:.2f}", r["note"]])

    def med(key, sel=lambda r: True):
        v = [r[key] for r in rows if r[key] is not None and not np.isnan(r[key]) and sel(r)]
        return (round(float(np.median(v)), 2) if v else None, len(v))

    dist_range = {}
    for k in (7, 15, 31):
        xs2, ys2 = _smooth(X, k), _smooth(Y, k)
        dist_range[k] = round(float(np.sum(np.hypot(np.diff(xs2[inr]), np.diff(ys2[inr])))), 1)
    segs = []
    for rally in rallies:
        a = rally["start"]
        while a < rally["end"] - 3:
            b = min(a + 9, rally["end"])
            m = (T >= a) & (T < b)
            d = np.hypot(Xs[m] - base[0], Ys[m] - base[1])
            segs.append({"rally": rally["id"], "from": round(a, 1), "to": round(b, 1),
                         "p90_dist_m": round(float(np.percentile(d, 90)), 2),
                         "mean_speed_mps": round(float(np.mean(speed[m])), 2)})
            a = b
    zones = {z: {"n": sum(r["zone"] == z for r in rows), "dist_median": med("dist", lambda r, z=z: r["zone"] == z),
                 "recover_median": med("recover", lambda r, z=z: r["zone"] == z),
                 "avail_median": med("avail", lambda r, z=z: r["zone"] == z)} for z in ("前场", "中场", "后场")}
    rear = [r for r in rows if r["zone"] == "后场"]
    summary = {
        "rallies": rallies, "rally_seconds": round(sum(r["end"] - r["start"] for r in rallies), 1),
        "tracked_ratio_in_rally": round(float(np.mean(~np.isnan(X[inr]))), 2),
        "base_m": {"x": round(float(base[0]), 2), "y": round(float(base[1]), 2)},
        "my_shots_confirmed": len(rows),
        "zones": zones, "recover_median_all": med("recover"), "avail_median_all": med("avail"),
        "rear_left_right": {"left(x<3.05)": sum(r["x"] < 3.05 for r in rear), "right": sum(r["x"] >= 3.05 for r in rear)},
        "total_distance_m_by_smoothing": dist_range, "segments_9s": segs,
        "disabled_metrics": cfg["reliability"]["disabled"],
    }
    write_json(out / "summary.json", summary)

    # ---- 图 ----
    C = {"前场": "#d9480f", "中场": "#1971c2", "后场": "#2f9e44"}

    def court(ax):
        ax.add_patch(plt.Rectangle((0, 0), 6.10, 6.70, fill=False, lw=1.5, color="#333"))
        for xv in (0.46, 5.64):
            ax.plot([xv, xv], [0, 6.70], color="#999", lw=0.8)
        for yv in (1.98, 5.94):
            ax.plot([0, 6.10], [yv, yv], color="#999", lw=0.8)
        ax.plot([3.05, 3.05], [1.98, 6.70], color="#999", lw=0.8)
        ax.plot([-0.3, 6.4], [0, 0], color="#333", lw=3)
        ax.text(3.05, -0.25, "球网", ha="center", va="top", fontsize=9)
        ax.set_xlim(-0.8, 6.9)
        ax.set_ylim(7.3, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5.6))
    court(axs[0])
    axs[0].plot(np.where(inr, Xs, np.nan), np.where(inr, Ys, np.nan), color="#868e96", lw=0.8, alpha=0.7)
    for r in rows:
        axs[0].scatter(r["x"], r["y"], s=60, color=C[r["zone"]], edgecolor="white", zorder=3)
    axs[0].scatter(*base, marker="x", s=120, color="black", zorder=4)
    axs[0].legend(handles=[Patch(color=c, label=z) for z, c in C.items()], loc="lower right", fontsize=8, frameon=False)
    axs[0].set_title("跑位轨迹与击球点（× = 中心位置）", fontsize=11)
    court(axs[1])
    hh, xe, ye = np.histogram2d(Xs[inr], Ys[inr], bins=[np.linspace(-0.5, 6.6, 15), np.linspace(-0.3, 7.2, 16)])
    axs[1].imshow(hh.T, extent=[xe[0], xe[-1], ye[-1], ye[0]], cmap="Oranges", alpha=0.85, aspect="auto", zorder=0)
    axs[1].set_title("停留时间热区", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_court.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(T, np.where(inr, np.hypot(Xs - base[0], Ys - base[1]), np.nan), color="#adb5bd", lw=1)
    for r in rows:
        ax.bar(r["t"], r["dist"], width=0.5, color=C[r["zone"]])
        if r["avail"] is not None and r["avail"] < 0.9:
            ax.text(r["t"], r["dist"] + 0.12, f"{r['avail']:.1f}s", ha="center", fontsize=7, color="#c92a2a")
    for rr in rallies:
        ax.axvspan(rr["start"], rr["end"], color="#f1f3f5", zorder=0)
    ax.legend(handles=[plt.Line2D([], [], color="#adb5bd", label="离中心距离（连续）")] +
              [Patch(color=c, label=z) for z, c in C.items()], fontsize=8, ncol=4, frameon=False, loc="upper left")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("击球时离中心（米）")
    ax.set_title("每拍击球时离中心多远；红字 = 准备时间不足 0.9 秒；灰底 = 回合", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_timeline.png", dpi=130)
    plt.close(fig)
    log(run, f"  本方确认击球 {len(rows)} 拍，回位中位数 {summary['recover_median_all'][0]}s")


# ---------------------------------------------------------------- 08 报告素材
def s08_report_input(run, cfg, meta):
    out = stage_dir(run, "08-report")
    for f in ("fig_court.png", "fig_timeline.png"):
        shutil.copy(run / "07-metrics" / f, out / f)
    shots = list(csv.DictReader(open(run / "07-metrics/shots.csv")))
    write_json(out / "report_input.json", {
        "meta": meta, "check": read_json(run / "00-check/check.json"),
        "calibration": read_json(run / "04-court/court.json"),
        "summary": read_json(run / "07-metrics/summary.json"), "shots": shots,
        "template": "参考 docs/method.md 的报告规则，人工撰写分析报告与训练计划",
        "rules": ["每条结论标注依据（数据 / 画面）和置信度（高 / 中 / 低）",
                  "disabled_metrics 中的指标不得出现在报告里",
                  "总跑动距离只报范围（见 total_distance_m_by_smoothing）",
                  "看不出得失分时明确写出"]})
    log(run, f"  报告素材已就绪：{out}/report_input.json")


STAGES = [("00", "质检", s00_check, "00-check/check.json"),
          ("01", "预处理", s01_prep, "01-prep/video.json"),
          ("02", "骨骼", s02_pose, "02-pose/overlay.mp4"),
          ("03", "场地标定", s03_calib, None),
          ("04", "场地坐标", s04_court, "04-court/court.json"),
          ("05", "事件检测", s05_events, "05-events/rallies.json"),
          ("06", "逐拍复核", s06_review, None),
          ("07", "指标与图表", s07_metrics, "07-metrics/summary.json"),
          ("08", "报告素材", s08_report_input, "08-report/report_input.json")]
