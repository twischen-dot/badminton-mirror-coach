"""流水线公共工具：路径、配置、状态、日志。"""

import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = Path(__file__).resolve().parent
MODEL = ROOT / "pose_landmarker_heavy.task"


def _tool(name):
    """按 环境变量 → PATH 的顺序找工具；ffmpeg 最后退回 imageio-ffmpeg 自带的版本。"""
    import os
    import shutil
    if os.environ.get("BADMINTON_" + name.upper()):
        return os.environ["BADMINTON_" + name.upper()]
    if shutil.which(name):
        return shutil.which(name)
    if name == "ffmpeg":
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    return None


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")  # 可能为 None：质检改用 OpenCV + ffmpeg


class NeedsReview(Exception):
    """流水线停在人工复核点。"""


class CheckFailed(Exception):
    """输入不合格，退回。"""


def load_config():
    with open(PIPE / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def load_meta(run):
    with open(run / "meta.toml", "rb") as fh:
        return tomllib.load(fh)


def stage_dir(run, name):
    d = run / name
    d.mkdir(exist_ok=True)
    return d


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sh(*args):
    subprocess.run([str(a) for a in args], check=True)


def log(run, msg):
    print(msg, flush=True)
    with open(run / "pipeline.log", "a") as fh:
        fh.write(msg + "\n")
