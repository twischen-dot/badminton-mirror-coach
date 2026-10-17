#!/bin/bash
# 在 Air 上无损剪出一小段（不重新编码，几乎不占 Air 资源），再拉回 Mini。
# 用法：live/fetch_clip.sh <Air 上的视频路径> <开始秒> [时长秒，默认 8] [Mini 输出目录]
# 整段 4K 视频从 Air 传到 Mini 很慢（300MB 用了 712 秒），几秒的片段只有几 MB。
set -euo pipefail
SRC="$1"; START="$2"; DUR="${3:-8}"; OUT="${4:-$HOME/aidol-work/clips}"
AIR_FF=/Users/twis/aidol-show/resultpage/.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1
NAME="$(basename "${SRC%.*}")_$(printf '%.0f' "$START")s.mp4"
mkdir -p "$OUT"; t0=$(date +%s)
ssh air "mkdir -p ~/aidol-show/clips && $AIR_FF -v error -y -ss $START -t $DUR -i '$SRC' -c copy -map 0:v:0 -map '0:a:0?' ~/aidol-show/clips/'$NAME'"
rsync -a "air:aidol-show/clips/$NAME" "$OUT/"
echo "片段: $OUT/$NAME  ($(du -h "$OUT/$NAME" | cut -f1)，用时 $(( $(date +%s) - t0 )) 秒)"
