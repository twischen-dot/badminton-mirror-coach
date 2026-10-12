"""两个人工复核点的网页：① 场地标定确认  ② 逐拍复核。

  python pipeline/review_server.py <分析名字> [--port 8765] [--host localhost]

默认只监听本机；浏览器打开 http://localhost:8765。
"""

import argparse
import base64
import csv
import json
import re
import sys
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_json  # noqa: E402
from stages import COURT_POINTS_HELP, STAGES  # noqa: E402

STD_POINTS = [
    ("网线×左双打边线（左网柱）", 0, 0), ("网线×右双打边线（右网柱）", 6.10, 0),
    ("前发球线×左双打边线", 0, 1.98), ("前发球线×左单打边线", 0.46, 1.98), ("前发球线×中线", 3.05, 1.98),
    ("前发球线×右单打边线", 5.64, 1.98), ("前发球线×右双打边线", 6.10, 1.98),
    ("双打后发球线×左双打边线", 0, 5.94), ("双打后发球线×左单打边线", 0.46, 5.94), ("双打后发球线×中线", 3.05, 5.94),
    ("双打后发球线×右单打边线", 5.64, 5.94), ("双打后发球线×右双打边线", 6.10, 5.94),
    ("底线×左双打边线", 0, 6.70), ("底线×左单打边线", 0.46, 6.70), ("底线×中线", 3.05, 6.70),
    ("底线×右单打边线", 5.64, 6.70), ("底线×右双打边线", 6.10, 6.70),
]
COURT_LINES = [((0, 0), (6.10, 0)), ((0, 1.98), (6.10, 1.98)), ((0, 5.94), (6.10, 5.94)), ((0, 6.70), (6.10, 6.70)),
               ((0, 0), (0, 6.70)), ((0.46, 0), (0.46, 6.70)), ((5.64, 0), (5.64, 6.70)), ((6.10, 0), (6.10, 6.70)),
               ((3.05, 1.98), (3.05, 6.70))]
LABELS = ["me", "opp", "none", "?"]
STROKES = ["", "高远", "杀球", "吊球", "劈吊", "网前", "挑球", "推球", "平抽", "抽挡", "接杀", "被动救球", "发球"]


def calib_points(run):
    p = run / "03-calib/calib.toml"
    if not p.exists():
        return {"ref_frame": 0, "confirmed": False, "points": [], "candidates": []}
    txt = p.read_text()
    with open(p, "rb") as fh:
        c = tomllib.load(fh)
    c["candidates"] = [[int(a), int(b)] for a, b in re.findall(r"#\s+P\d+ = \[(\d+), (\d+)\]", txt)]
    return c


def project_overlay(run, points):
    P = np.float32(points)
    H, _ = cv2.findHomography(P[:, 2:], P[:, :2], 0)       # 场地 → 画面
    back = cv2.perspectiveTransform(P[None, :, 2:], H)[0]
    err_px = np.linalg.norm(back - P[:, :2], axis=1)
    Hi, _ = cv2.findHomography(P[:, :2], P[:, 2:], 0)
    err_m = np.linalg.norm(cv2.perspectiveTransform(P[None, :, :2], Hi)[0] - P[:, 2:], axis=1)
    img = cv2.imread(str(run / "03-calib/ref.png"))
    for a, b in COURT_LINES:
        q = cv2.perspectiveTransform(np.float32([np.linspace(a, b, 40)]), H)[0].astype(int)
        cv2.polylines(img, [q], False, (255, 0, 255), 2, cv2.LINE_AA)
    for x, y, *_ in points:
        cv2.circle(img, (int(x), int(y)), 9, (0, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"img": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
            "err_px_mean": float(err_px.mean()), "err_m_mean": float(err_m.mean()), "err_m_max": float(err_m.max())}


def write_calib(run, ref_frame, points):
    c = calib_points(run)
    lines = [COURT_POINTS_HELP, f"ref_frame = {ref_frame}", "confirmed = true  # 由复核网页确认",
             "# 自动找到的候选交点（画面像素）：", *[f"#   P{i} = [{a}, {b}]" for i, (a, b) in enumerate(c["candidates"])],
             "points = [", *[f"  [{x:.0f}, {y:.0f}, {cx:.2f}, {cy:.2f}]," for x, y, cx, cy in points], "]"]
    (run / "03-calib/calib.toml").write_text("\n".join(lines) + "\n")


def read_labels(run):
    p = run / "06-review/labels.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def write_labels(run, rows):
    with open(run / "06-review/labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "t", "rally", "label", "stroke", "note"])
        w.writeheader()
        w.writerows(rows)


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--bg:#fafaf7;--fg:#222;--mut:#777;--line:#ddd;--acc:#1971c2;--me:#2f9e44;--opp:#d9480f;--none:#868e96}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1b;--fg:#eee;--mut:#999;--line:#333}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"PingFang SC",sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
header a{{color:var(--acc);text-decoration:none}} main{{padding:16px;max-width:1400px;margin:auto}}
button{{font:inherit;padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}}
button.on{{color:#fff;border-color:transparent}} .mut{{color:var(--mut)}} table{{border-collapse:collapse}}
td,th{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}}
</style></head><body><header><b>羽毛球分析复核 · {run}</b><a href="/">进度</a><a href="/calib">① 场地标定</a><a href="/review">② 逐拍复核</a></header>
<main>{body}</main></body></html>"""

INDEX = """<h2>流水线进度</h2><table>{rows}</table>
<p class="mut">复核完成后，在 本机 上运行：<code>.venv/bin/python pipeline/run.py run {run}</code></p>"""

CALIB = """<h2>① 场地标定</h2>
<p class="mut">在图上点击一个场地线交点（靠近黄色候选圈时会自动吸附），再在右侧选择它是哪个标准点。至少 4 个点，最好同时包含靠网和靠底线的点。
画面外的点可以直接输入坐标。点"预览"会把整个半场的标准场地线（紫色）投影回画面：紫线和真实白线重合，才算标定正确。</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">
<div style="flex:3;min-width:320px;position:relative"><img id="img" src="/calib/ref.png" style="width:100%;cursor:crosshair">
<svg id="ov" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none"></svg></div>
<div style="flex:2;min-width:300px"><table id="pts"><tr><th>画面 x</th><th>画面 y</th><th>场地点</th><th></th></tr></table>
<p><button onclick="addRow(0,0)">＋手动添加</button> <button onclick="preview()">预览</button> <button onclick="save()" id="sv">保存并确认</button></p>
<p id="msg" class="mut"></p></div></div>
<p><img id="pv" style="width:100%;display:none"></p>
<script>
const STD={std}; const C={calib};
const img=document.getElementById('img'), ov=document.getElementById('ov'), tb=document.getElementById('pts');
function opts(sel){{return STD.map((s,i)=>`<option value="${{i}}" ${{i==sel?'selected':''}}>${{s[0]}} (${{s[1]}}, ${{s[2]}})</option>`).join('')}}
function addRow(x,y,sel){{const tr=tb.insertRow();tr.innerHTML=`<td><input size=5 value="${{Math.round(x)}}"></td><td><input size=5 value="${{Math.round(y)}}"></td><td><select>${{opts(sel??-1)}}<option value=-1 ${{sel==null?'selected':''}}>— 选择 —</option></select></td><td><button>删</button></td>`;
 tr.querySelector('button').onclick=()=>{{tr.remove();draw()}}; tr.querySelectorAll('input').forEach(i=>i.oninput=draw); draw()}}
function rows(){{return [...tb.rows].slice(1).map(r=>{{const i=r.querySelectorAll('input'),s=+r.querySelector('select').value;return [+i[0].value,+i[1].value,s]}})}}
function draw(){{const W=img.naturalWidth,H=img.naturalHeight; if(!W)return; ov.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);
 const u=W/500; ov.innerHTML=C.candidates.map(p=>`<circle cx="${{p[0]}}" cy="${{p[1]}}" r="${{5*u}}" fill="none" stroke="yellow" stroke-width="${{1.5*u}}" opacity="0.85" />`).join('')+
 rows().map((p,i)=>`<circle cx="${{p[0]}}" cy="${{p[1]}}" r="${{6*u}}" fill="none" stroke="#ff00ff" stroke-width="${{2*u}}" /><text x="${{p[0]+8*u}}" y="${{p[1]-6*u}}" fill="#ff00ff" stroke="#000" stroke-width="${{0.4*u}}" font-size="${{14*u}}" font-weight="bold">${{i+1}}</text>`).join('')}}
img.onload=()=>{{C.points.forEach(p=>{{const s=STD.findIndex(q=>Math.abs(q[1]-p[2])<.01&&Math.abs(q[2]-p[3])<.01);addRow(p[0],p[1],s<0?null:s)}});draw()}};
img.onclick=e=>{{const r=img.getBoundingClientRect(),k=img.naturalWidth/r.width;let x=(e.clientX-r.left)*k,y=(e.clientY-r.top)*k;
 let best=null;C.candidates.forEach(p=>{{const d=Math.hypot(p[0]-x,p[1]-y);if(d<25&&(!best||d<best[0]))best=[d,p]}}); if(best){{x=best[1][0];y=best[1][1]}} addRow(x,y)}};
function payload(){{const r=rows();if(r.some(p=>p[2]<0))throw '有点还没选择对应的场地点';if(r.length<4)throw '至少需要 4 个点';
 return {{ref_frame:C.ref_frame,points:r.map(p=>[p[0],p[1],STD[p[2]][1],STD[p[2]][2]])}}}}
async function post(u){{let b;try{{b=payload()}}catch(e){{msg.textContent=e;return}}
 const r=await fetch(u,{{method:'POST',body:JSON.stringify(b)}});return r.json()}}
async function preview(){{const j=await post('/api/calib/preview');if(!j)return;pv.src=j.img;pv.style.display='block';
 msg.textContent=`重投影误差：平均 ${{j.err_m_mean.toFixed(3)}} 米，最大 ${{j.err_m_max.toFixed(3)}} 米（${{j.err_px_mean.toFixed(1)}} 像素）。请检查下方紫线是否和白线重合。`}}
async function save(){{const j=await post('/api/calib/save');if(j)msg.textContent=j.ok?'已保存并确认。回到 本机 重新运行流水线即可继续。':j.error}}
</script>"""

REVIEW = """<h2>② 逐拍复核</h2>
<p class="mut">每行是一个候选击球：放大的近端球员，左中右三帧为击球声前 0.13 秒、击球声时刻、后 0.13 秒。
快捷键：<b>1</b> 我（近端） <b>2</b> 对手 <b>3</b> 不是击球 <b>0</b> 未判断 · <b>J/K</b> 下一个/上一个 · <b>P</b> 播放前后 1 秒。每次改动自动保存。
提示：同一回合里双方击球应该交替出现。</p>
<p id="cnt"></p><p><button onclick="done()">全部复核完成</button> <span id="msg" class="mut"></span></p>
<video id="vid" src="/video" preload="none" style="position:fixed;right:16px;bottom:16px;width:420px;max-width:45vw;display:none;border:2px solid var(--acc)" muted playsinline></video>
<div id="list"></div>
<script>
let R={labels}; const ST={strokes}; let cur=0;
const COL={{me:'var(--me)',opp:'var(--opp)',none:'var(--none)'}}, NAME={{me:'我',opp:'对手',none:'不是','?':'?'}};
function render(){{list.innerHTML=R.map((r,i)=>`<div id="r${{i}}" style="display:flex;gap:12px;align-items:center;padding:8px;border-left:4px solid ${{COL[r.label]||'transparent'}};${{i==cur?'background:rgba(25,113,194,.08)':''}}">
<div style="width:120px"><b>#${{r.k}}</b><br>${{(+r.t).toFixed(2)}}s<br><span class="mut">回合 ${{r.rally}}</span><br>${{warn(i)}}</div>
<img src="/strip/${{String(r.k).padStart(4,'0')}}.jpg" style="height:160px;max-width:60vw" loading="lazy" onclick="sel(${{i}})">
<div>${{['me','opp','none','?'].map(l=>`<button class="${{r.label==l?'on':''}}" style="${{r.label==l?'background:'+(COL[l]||'#999'):''}}" onclick="setL(${{i}},'${{l}}')">${{NAME[l]}}</button>`).join(' ')}}
<br><select onchange="setF(${{i}},'stroke',this.value)">${{(ST.includes(r.stroke)?ST:[...ST,r.stroke]).map(s=>`<option ${{s==r.stroke?'selected':''}}>${{s}}</option>`).join('')}}</select>
<input placeholder="备注" value="${{r.note||''}}" onchange="setF(${{i}},'note',this.value)" size=18> <button onclick="play(${{i}})">▶</button></div></div>`).join('');
 const c=R.reduce((a,r)=>(a[r.label]=(a[r.label]||0)+1,a),{{}}); cnt.textContent=`共 ${{R.length}} 个候选：我 ${{c.me||0}} · 对手 ${{c.opp||0}} · 不是 ${{c.none||0}} · 未判断 ${{c['?']||0}}`}}
function warn(i){{const r=R[i]; if(r.label!='me'&&r.label!='opp')return '';
 for(let j=i-1;j>=0;j--){{const p=R[j];if(p.rally!=r.rally)break;if(p.label=='me'||p.label=='opp'){{return p.label==r.label?'<span style="color:#c92a2a">⚠ 与上一拍同一人</span>':''}}}} return ''}}
async function save(i){{await fetch('/api/label',{{method:'POST',body:JSON.stringify(R[i])}})}}
function setL(i,l){{R[i].label=l;save(i);sel(i)}} function setF(i,f,v){{R[i][f]=v;save(i)}}
function sel(i){{cur=Math.max(0,Math.min(R.length-1,i));render();document.getElementById('r'+cur).scrollIntoView({{block:'center'}})}}
function play(i){{const t=+R[i].t;vid.style.display='block';vid.currentTime=Math.max(0,t-1);vid.play();clearTimeout(window.pt);window.pt=setTimeout(()=>vid.pause(),2000)}}
document.onkeydown=e=>{{if(e.target.tagName=='INPUT')return;const m={{'1':'me','2':'opp','3':'none','0':'?'}};
 if(m[e.key]){{setL(cur,m[e.key]);sel(cur+1)}} else if(e.key=='j')sel(cur+1); else if(e.key=='k')sel(cur-1); else if(e.key=='p')play(cur)}};
async function done(){{const n=R.filter(r=>r.label=='?').length;if(n&&!confirm(`还有 ${{n}} 个未判断，确定完成吗？未判断的不会参与统计。`))return;
 const j=await (await fetch('/api/review/done',{{method:'POST'}})).json();msg.textContent=j.ok?'已标记完成。回到 本机 重新运行流水线即可继续。':j.error}}
render();
</script>"""


def make_handler(run):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def page(self, title, body):
            self.send(200, PAGE.format(title=title, run=run.name, body=body))

        def file(self, path, ctype):
            if not path.exists():
                return self.send(404, "not found", "text/plain")
            size = path.stat().st_size
            rng = self.headers.get("Range")
            if rng:  # Safari 播放视频需要 Range 支持
                a, b = rng.replace("bytes=", "").split("-")
                a = int(a)
                b = int(b) if b else min(size - 1, a + 4 * 1024 * 1024)
                with open(path, "rb") as fh:
                    fh.seek(a)
                    data = fh.read(b - a + 1)
                return self.send(206, data, ctype, {"Content-Range": f"bytes {a}-{b}/{size}", "Accept-Ranges": "bytes"})
            self.send(200, path.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                st = read_json(run / "status.json", {})
                rows = "".join(f"<tr><td>[{c}]</td><td>{n}</td><td>{st.get(c, {}).get('state', '-')}</td>"
                               f"<td class=mut>{st.get(c, {}).get('msg', '')}</td></tr>" for c, n, *_ in STAGES)
                return self.page("进度", INDEX.format(rows=rows, run=run.name))
            if p == "/calib":
                if not (run / "03-calib/ref.png").exists():
                    return self.page("场地标定", "<p>还没有生成参考帧，请先运行流水线到第 03 步。</p>")
                return self.page("场地标定", CALIB.format(std=json.dumps(STD_POINTS, ensure_ascii=False),
                                                       calib=json.dumps(calib_points(run))))
            if p == "/calib/ref.png":
                return self.file(run / "03-calib/ref.png", "image/png")
            if p == "/review":
                labels = read_labels(run)
                if not labels:
                    return self.page("逐拍复核", "<p>还没有候选击球，请先运行流水线到第 06 步。</p>")
                return self.page("逐拍复核", REVIEW.format(labels=json.dumps(labels, ensure_ascii=False),
                                                       strokes=json.dumps(STROKES, ensure_ascii=False)))
            m = re.fullmatch(r"/strip/(\d{4})\.jpg", p)
            if m:
                return self.file(run / "05-events/strips" / f"{m.group(1)}.jpg", "image/jpeg")
            if p == "/video":
                return self.file((run / "input.mp4").resolve(), "video/mp4")
            self.send(404, "not found", "text/plain")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            data = json.loads(body) if body else {}
            try:
                if self.path == "/api/calib/preview":
                    return self.send(200, json.dumps(project_overlay(run, data["points"])), "application/json")
                if self.path == "/api/calib/save":
                    res = project_overlay(run, data["points"])
                    write_calib(run, data["ref_frame"], data["points"])
                    return self.send(200, json.dumps({"ok": True, "err_m_mean": res["err_m_mean"]}), "application/json")
                if self.path == "/api/label":
                    rows = read_labels(run)
                    for r in rows:
                        if r["k"] == str(data["k"]):
                            r["label"] = data["label"] if data["label"] in LABELS else "?"
                            r["stroke"] = data.get("stroke", "")
                            r["note"] = data.get("note", "").replace("\n", " ")
                    write_labels(run, rows)
                    return self.send(200, '{"ok":true}', "application/json")
                if self.path == "/api/review/done":
                    (run / "06-review/DONE").write_text("由复核网页标记完成\n")
                    return self.send(200, '{"ok":true}', "application/json")
            except Exception as e:  # 把错误显示在页面上
                return self.send(200, json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False), "application/json")
            self.send(404, "not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="localhost")
    a = ap.parse_args()
    run = ROOT / "runs" / a.name
    if not run.exists():
        sys.exit(f"没有这个分析：{run}")
    print(f"复核网页：http://{a.host}:{a.port}   （Ctrl+C 退出）")
    ThreadingHTTPServer((a.host, a.port), make_handler(run)).serve_forever()


if __name__ == "__main__":
    main()
