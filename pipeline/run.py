"""羽毛球视频分析流水线总入口。

  新建一次分析：  run.py new <视频> --name <名字> [--audio <音频>]
  运行 / 续跑：    run.py run <名字> [--from 04] [--to 07]
  查看进度：      run.py status <名字>

每次分析一个目录 runs/<名字>/，各步骤输出在 00-check/ … 08-report/。
遇到人工复核点（① 场地标定、② 逐拍复核）会停下；复核完再 run 即可从停下处继续。
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CheckFailed, NeedsReview, load_config, load_meta, log, read_json, write_json  # noqa: E402
from stages import STAGES  # noqa: E402

META_TEMPLATE = """# 本次分析的基本信息（必填项标 *）
player = "{player}"        # * 被分析的球员
handed = "right"            # * 持拍手：right / left
side = "near"               # 被分析球员在近端（靠摄像机）半场；目前只支持 near
jersey = ""                 # 球衣颜色，帮助复核时辨认
opponent = ""
date = "{date}"
camera = ""                 # 机位描述，例如 "本方半场后方高处"
score = ""                  # 比分或得失分记录，没有就留空
audio = "{audio}"           # 视频本身没有声音时，另给一个音频文件（相对本目录）
# calib_ref_frame = 120     # 场地标定用的参考帧，默认第 120 帧
notes = ""
"""


def cmd_new(a):
    run = ROOT / "runs" / a.name
    if run.exists():
        sys.exit(f"已存在：{run}")
    run.mkdir(parents=True)
    src = Path(a.video).expanduser().resolve()
    (run / "input.mp4").symlink_to(src)
    audio = ""
    if a.audio:
        asrc = Path(a.audio).expanduser().resolve()
        shutil.copy(asrc, run / ("audio" + asrc.suffix))
        audio = "audio" + asrc.suffix
    (run / "meta.toml").write_text(META_TEMPLATE.format(player=a.player or "", date=time.strftime("%Y-%m-%d"),
                                                        audio=audio))
    print(f"已创建 {run}\n下一步：检查 meta.toml，然后 run.py run {a.name}")


def cmd_run(a):
    run = ROOT / "runs" / a.name
    cfg, meta = load_config(), load_meta(run)
    status = read_json(run / "status.json", {})
    for code, name, fn, done_marker in STAGES:
        if a.from_ and code < a.from_:
            continue
        if a.to and code > a.to:
            break
        if not a.from_ and done_marker and (run / done_marker).exists():
            continue
        log(run, f"[{code}] {name} …")
        t0 = time.time()
        try:
            fn(run, cfg, meta)
        except NeedsReview as e:
            status[code] = {"state": "needs_review", "msg": str(e)}
            write_json(run / "status.json", status)
            log(run, f"⏸ 停在复核点 {e}")
            log(run, f"   复核网页：.venv/bin/python pipeline/review_server.py {a.name}   然后打开 http://localhost:8765")
            return
        except CheckFailed as e:
            status[code] = {"state": "failed", "msg": str(e)}
            write_json(run / "status.json", status)
            log(run, f"✗ 输入不合格，退回：{e}")
            sys.exit(2)
        status[code] = {"state": "done", "seconds": round(time.time() - t0, 1)}
        write_json(run / "status.json", status)
        log(run, f"    完成，用时 {time.time() - t0:.0f}s")
    log(run, "✓ 流水线完成")


def cmd_status(a):
    run = ROOT / "runs" / a.name
    status = read_json(run / "status.json", {})
    for code, name, _, _ in STAGES:
        s = status.get(code, {})
        print(f"[{code}] {name:8s} {s.get('state', '-'):13s} {s.get('msg', '')}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("video")
    n.add_argument("--name", required=True)
    n.add_argument("--audio")
    n.add_argument("--player")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--from", dest="from_")
    r.add_argument("--to")
    s = sub.add_parser("status")
    s.add_argument("name")
    a = p.parse_args()
    {"new": cmd_new, "run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
