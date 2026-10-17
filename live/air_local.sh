#!/bin/bash
# 备用方案：网络太差、Mini 连不上时，直接在现场 Air 上分析和出片（不经过 Mini）。
# 在 Air 上运行：
#   ~/aidol-show/analyze/repo/live/air_local.sh hits VIDEO [--sheet out.jpg]
#   ~/aidol-show/analyze/repo/live/air_local.sh render --player VIDEO --hit T --ref REF --ref-hit T ... [--play]
# render 完成后自动生成手机页、确保结果页服务在跑、打印扫码地址。
# 环境：~/aidol-show/analyze/.venv（Python 3.12 + mediapipe 1.1.0），Air 的 Homebrew ffmpeg 坏了，改用 imageio-ffmpeg 自带的。
set -euo pipefail
A=~/aidol-show/analyze; SHOW=~/aidol-show; PORT=${YUYAN_PORT:-8080}
export YUYAN_MODEL=$A/pose_landmarker_heavy.task
export BADMINTON_FFMPEG=$SHOW/resultpage/.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1
PY="$A/.venv/bin/python $A/repo/live/yuyan.py"
cmd=${1:?用法: air_local.sh hits|render ...}; shift
if [ "$cmd" = hits ]; then exec $PY hits "$@"; fi
PLAY=0; args=()
for x in "$@"; do [ "$x" = --play ] && PLAY=1 || args+=("$x"); done
t0=$(date +%s)
$PY render "${args[@]}" --sessions $SHOW/sessions > /dev/null
SID=$(ls -t $SHOW/sessions | head -1)
cd $SHOW/resultpage
.venv/bin/python build.py ../sessions ../site > /dev/null
if ! /usr/bin/pgrep -f 'serve.py --site ../site' > /dev/null; then
  nohup .venv/bin/python serve.py --site ../site --port $PORT > ../serve.log 2>&1 &
  sleep 1
fi
IP=$(/usr/sbin/ipconfig getifaddr en0 || true); URL=http://$IP:$PORT/$SID/
.venv/bin/python qr.py "$URL" ../site/$SID/qr.png > /dev/null
echo "会话: $SID"; echo "扫码地址: $URL"; echo "二维码: ~/aidol-show/site/$SID/qr.png"
echo "Air 本机出片+发布耗时: $(( $(date +%s) - t0 )) 秒"
[ $PLAY = 1 ] && /usr/bin/open -a 'QuickTime Player' $SHOW/sessions/$SID/compare.mp4 || true
