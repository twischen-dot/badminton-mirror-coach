"""对镜陪练样片：触球对齐、慢放、骨骼叠加与往复对比。
用法：python compare/render_compare.py --config examples/compare.example.toml --help
"""
import argparse
import tomllib
from pathlib import Path
import subprocess, json, numpy as np, cv2, mediapipe as mp
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

FF = "ffmpeg"
W, H, FPS = 1920, 1080, 30
PRE, POST, SLOW = 1.0, 0.5, 4          # 击球前 1.0 秒、后 0.5 秒，慢放 4 倍
FONT = None

CLIPS = {}
pose = None
CONN = None
out = None
frame_count = 0
check_times = {}

@lru_cache(None)
def font(sz): return ImageFont.truetype(FONT, sz)

def read_frames(path, t0, dur, fps=FPS, vf="scale=1920:1080", wh=(1920, 1080)):
    cmd = [FF, "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", path,
           "-vf", f"fps={fps},{vf}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, wh[1], wh[0], 3)

NPRE, NPOST = int(PRE * FPS), int(POST * FPS)   # 输出时间轴：击球前/后各多少帧（未慢放）


def prep(c):
    pose.reset()
    if "fullbody" in c:
        fr = read_frames(c["path"], 0.8, 1.52, 25, "crop=360:420:400:300", (360,420))
    elif "win" in c:   # 已慢放的源：按击球点分两段线性映射到统一时间轴
        a, h, b = c["win"]; x0, y0, x1, y1 = c["roi"]
        src = read_frames(c["path"], a, b - a, c["src_fps"], c["vf"], (x1 - x0, y1 - y0))
        hi = int(round((h - a) * c["src_fps"]))
        idx = [int(round(hi * k / NPRE)) for k in range(NPRE)] + \
              [int(round(hi + (len(src) - 1 - hi) * k / NPOST)) for k in range(NPOST)]
        fr = src[idx]
    else:
        fr = read_frames(c["path"], c["hit"] - PRE, PRE + POST)
    x0, y0, x1, y1 = c["roi"]
    lms = []
    for f in fr:
        r = pose.process(np.ascontiguousarray(f[y0:y1, x0:x1]))
        if r.pose_landmarks:
            lms.append([(x0 + l.x * (x1 - x0), y0 + l.y * (y1 - y0), l.visibility) for l in r.pose_landmarks.landmark])
        else:
            lms.append(None)
    pts = np.array([p[:2] for L in lms if L for p in L if p[2] > 0.5])
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    span = max(np.percentile(pts[:, 0], 98) - np.percentile(pts[:, 0], 2),
               np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2)) * 1.45
    fh, fw = fr.shape[1:3]
    s = int(min(max(span, 420), fh, fw))
    bx = int(np.clip(cx - s / 2, 0, fw - s)); by = int(np.clip(cy - s / 2 - s * 0.05, 0, fh - s))
    box = (0, 0, 360, 420) if "fullbody" in c else (bx, by, s, s)
    hit_i = 19 if "fullbody" in c else NPRE
    c.update(frames=fr, lms=lms, box=box, hit_i=hit_i)
    print(c["label"], "frames", len(fr), "pose ok", sum(L is not None for L in lms), "box", c["box"])

def panel(c, i, size, hit_mark=False):
    bx, by, bw, bh = c["box"]
    pw, ph = (round(size*bw/bh), size) if "fullbody" in c else (size,size)
    img = Image.fromarray(c["frames"][i][by:by + bh, bx:bx + bw]).resize((pw, ph), Image.LANCZOS)
    d = ImageDraw.Draw(img); k = size / bh
    L = c["lms"][i]
    if L:
        for a, b in CONN:
            if L[a][2] > 0.5 and L[b][2] > 0.5:
                d.line([((L[a][0] - bx) * k, (L[a][1] - by) * k), ((L[b][0] - bx) * k, (L[b][1] - by) * k)],
                       fill=(0, 255, 200), width=max(2, size // 300))
    if hit_mark:
        d.rectangle([0, 0, pw - 1, ph - 1], outline=(255, 210, 0), width=8)
        d.text((pw // 2, ph - 40), "击球瞬间", font=font(44), fill=(255, 210, 0), anchor="mm")
    return img

def canvas(title=None, sub=None):
    im = Image.new("RGB", (W, H), (18, 20, 24)); d = ImageDraw.Draw(im)
    if title: d.text((60, 40), title, font=font(46), fill="white")
    if sub: d.text((60, 102), sub, font=font(30), fill=(170, 175, 185))
    d.text((W - 60, H - 40), "AI 羽毛球对镜陪练 · 样片", font=font(26), fill=(120, 125, 135), anchor="rs")
    return im

def seq(c):
    """常速素材慢放四倍；全身素材保留25fps源帧，输出120fps等效时间。"""
    n = len(c["frames"])
    for i in range(n):
        base = round((i+1)*FPS*4/25)-round(i*FPS*4/25) if "fullbody" in c else SLOW
        reps = int(0.8 * FPS) if i == c["hit_i"] else base
        yield i, i == c["hit_i"], reps

def emit(im, k=1):
    global frame_count
    frame_count += k
    b = im.tobytes()
    for _ in range(k): out.stdin.write(b)

def main():
    global FF, FONT, CLIPS, pose, CONN, out, frame_count, check_times
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='素材 TOML；素材相对路径以此文件所在目录为基准')
    parser.add_argument('--font', type=Path, help='支持中文的字体文件；覆盖配置 font')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='编码器命令或路径')
    parser.add_argument('--output', type=Path, default=Path('comparison.mp4'))
    parser.add_argument('--metadata', type=Path, default=Path('render_metadata.json'))
    a = parser.parse_args()
    with a.config.open('rb') as f:
        settings = tomllib.load(f)
    base = a.config.resolve().parent
    for name in ('am', 'pro', 'full'):
        settings[name]['path'] = str(base / settings[name]['path'])
    FF = a.ffmpeg
    FONT = str(a.font) if a.font else str(base / settings['font'])
    CLIPS = {name: dict(settings[name]) for name in ('am', 'pro')}
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    CONN = mp.solutions.pose.POSE_CONNECTIONS
    out = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", "-threads", "4",
                            "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    frame_count = 0
    check_times = {}

    for c in CLIPS.values(): prep(c)
    am, pro = CLIPS["am"], CLIPS["pro"]

    # 1 片头
    im = canvas(); d = ImageDraw.Draw(im)
    d.text((W // 2, 430), "打 3 分钟，看 1 分钟", font=font(96), fill="white", anchor="mm")
    d.text((W // 2, 560), "把你的动作和高手的同一动作放在一起慢放", font=font(44), fill=(200, 205, 215), anchor="mm")
    d.text((W // 2, 660), "击球瞬间对齐 · 慢镜头 · AI 骨骼识别", font=font(32), fill=(140, 145, 155), anchor="mm")
    emit(im, 3 * FPS)

    # 2 往复 3 次：本人 → 高手
    for rnd in range(1, 4):
        for c in (am, pro):
            for i, hm, reps in seq(c):
                im = canvas(f"{c['label']}", f"{c['sub']}　·　第 {rnd}/3 遍")
                im.paste(panel(c, i, 900, hm), ((W - 900) // 2, 160))
                if rnd == 1 and c is pro and hm: check_times["pro"] = (frame_count + 12)/FPS
                emit(im, reps)

    # 3 并排同步
    sz = 860
    for i, hm, reps in seq(am):
        im = canvas("并排对比", "同一动作 · 击球瞬间对齐"); d = ImageDraw.Draw(im)
        for j, c in enumerate((am, pro)):
            x = 80 + j * (sz + 100)
            im.paste(panel(c, i, sz, hm), (x, 170))
            d.text((x + sz // 2, 150), c["label"], font=font(36), fill="white", anchor="mb")
        if hm: check_times["side"] = (frame_count + 12)/FPS
        emit(im, reps)

    # 4 全身动作：起跳与步法
    full = dict(settings["full"])
    full["fullbody"] = True
    prep(full)
    for rnd in range(1,3):
        for i, hm, reps in seq(full):
            im = canvas(full["label"], full["sub"])
            d = ImageDraw.Draw(im)
            d.text((1450, 400), "全身动作：起跳与步法", font=font(34), fill="white", anchor="mm")
            d.text((1450, 465), f"慢放 4 倍　·　第 {rnd}/2 遍", font=font(30), fill=(170,175,185), anchor="mm")
            body = panel(full, i, 880, hm)
            im.paste(body, (480,160))
            if rnd == 1 and hm: check_times["full"] = (frame_count + 12)/FPS
            emit(im, reps)

    # 5 片尾
    im = canvas("素材说明"); d = ImageDraw.Draw(im)
    lines = settings["credits"]["lines"]
    for k, t in enumerate(lines):
        d.text((120, 300 + k * 80), t, font=font(28), fill=(220, 222, 228))
    emit(im, 5 * FPS)

    out.stdin.close()
    if out.wait() != 0: raise RuntimeError("ffmpeg encoding failed")
    with open(a.metadata, "w") as f:
        json.dump(dict(frames=frame_count, duration=frame_count/FPS, check_times=check_times, fullbody=dict(source_start=0.8,source_end=2.32,hit=1.56,crop=[400,300,360,420],pose_frames=sum(L is not None for L in full["lms"]),total_source_frames=len(full["frames"]))),f,indent=2)
    print("done")


if __name__ == "__main__":
    main()
