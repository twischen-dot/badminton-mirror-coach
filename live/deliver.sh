#!/bin/sh
# 把 Mini 上生成的一次结果送到现场 Air：同步 → 生成手机页 → 确保服务在跑 → 打印扫码地址并生成二维码。
# 用法：live/deliver.sh <session 目录> [--play]   （--play 会在 Air 上用 QuickTime 打开对比视频）
set -e
S=${1:?用法: live/deliver.sh <session 目录> [--play]}; SID=$(basename "$S"); PORT=${YUYAN_PORT:-8080}
t0=$(date +%s)
/usr/bin/rsync -a "$S/" "air:aidol-show/sessions/$SID/"
/usr/bin/ssh air "set -e; cd ~/aidol-show/resultpage
  .venv/bin/python build.py ../sessions ../site >/dev/null
  if ! /usr/bin/pgrep -f 'serve.py --site ../site' >/dev/null; then
    nohup .venv/bin/python serve.py --site ../site --port $PORT > ../serve.log 2>&1 &
    sleep 1
  fi
  IP=\$(/usr/sbin/ipconfig getifaddr en0 || true)
  URL=http://\$IP:$PORT/$SID/
  .venv/bin/python qr.py \"\$URL\" ../site/$SID/qr.png >/dev/null
  echo \"扫码地址: \$URL\"; echo \"二维码: ~/aidol-show/site/$SID/qr.png\"
  [ \"$2\" = --play ] && /usr/bin/open -a 'QuickTime Player' ~/aidol-show/sessions/$SID/compare.mp4 || true"
echo "送达耗时: $(( $(date +%s) - t0 )) 秒"
