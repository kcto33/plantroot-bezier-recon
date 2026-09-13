# -*- coding: utf-8 -*-
"""
web_server.py
纯标准库 Web 服务器(http.server), 不依赖 Flask。

两个输入入口:
  POST /api/from_matlab  接收 {file: <base64/.m>}  -> 生成模型/根表/重建图
  POST /api/from_images  接收 {img: [base64*24], pose: <base64或用默认>} -> 标色三角化/空间表/贝塞尔图
  GET  /results/<path>   服务生成的图片和表文件
  GET  /                 主页(前端)
"""
import os
import io
import sys
import json
import uuid
import base64
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def resource_path(rel):
    """解析随程序分发的资源路径(兼容 PyInstaller 冻结模式)。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# 生成的结果放在这个目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def set_result_dir(path):
    """重定向结果目录(桌面版指向用户可写目录)。"""
    global RESULT_DIR
    RESULT_DIR = path
    os.makedirs(RESULT_DIR, exist_ok=True)


def save_b64_to_file(b64, path):
    """保存 base64 字符串为二进制文件。返回保存的路径。"""
    if isinstance(b64, str) and b64.startswith('data:'):
        # data URL 形式: data:...;base64,xxxx
        b64 = b64.split(',', 1)[1]
    data = base64.b64decode(b64)
    with open(path, 'wb') as f:
        f.write(data)
    return path


# ---- 后台任务管理: POST 立即返回 job_id, 前端轮询 /api/job/<id> 取进度 ----
JOBS = {}
JOBS_LOCK = threading.Lock()


def _new_job():
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {'status': 'running', 'stage': '准备中', 'done': 0, 'total': 0,
                     'urls': {}, 'result': None, 'error': None}
    return jid


def _job_progress_cb(jid):
    def cb(stage, done, total):
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if j:
                j.update(stage=stage, done=done, total=total)
    return cb


def _run_job(jid, fn):
    def target():
        try:
            res = fn()
            with JOBS_LOCK:
                JOBS[jid]['status'] = 'done'
                JOBS[jid]['result'] = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            with JOBS_LOCK:
                JOBS[jid]['status'] = 'error'
                JOBS[jid]['error'] = str(e)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


class Handler(BaseHTTPRequestHandler):
    # ---- 静态资源(前端) ----
    def do_GET(self):
        # 去掉查询参数(?xxx=yyy), 只按路径路由
        self.path = self.path.split('?', 1)[0]
        if self.path.startswith('/api/job/'):
            jid = self.path[len('/api/job/'):]
            with JOBS_LOCK:
                j = JOBS.get(jid)
                if j:
                    self._json(200, dict(j))
                else:
                    self._json(404, {"error": "任务不存在"})
            return
        if self.path == '/' or self.path == '/index.html':
            self._serve_file(resource_path("templates/index.html"), "text/html; charset=utf-8")
        elif self.path.startswith('/api/zip_views/'):
            # 把该 job 的 24 张图 + pose.json 打包成 ZIP 下载
            job = os.path.basename(self.path[len('/api/zip_views/'):])
            self._serve_zip(job)
        elif self.path.startswith('/results/'):
            rel = self.path[len('/results/'):]
            # 防止路径穿越
            safe = os.path.normpath(rel)
            full = os.path.join(RESULT_DIR, safe)
            if os.path.isfile(full):
                if full.endswith('.csv'):
                    self._serve_file(full, "text/csv; charset=utf-8")
                elif full.endswith('.sql'):
                    self._serve_file(full, "text/plain; charset=utf-8")
                elif full.endswith('.json'):
                    self._serve_file(full, "application/json; charset=utf-8")
                elif full.endswith('.png'):
                    self._serve_file(full, "image/png")
                elif full.endswith('.npy'):
                    self._serve_file(full, "application/octet-stream")
                else:
                    self._serve_file(full, "application/octet-stream")
            else:
                self._json(404, {"error": "文件不存在"})
        else:
            self._json(404, {"error": "未知路由"})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode('utf-8')) if body else {}
        except Exception as e:
            self._json(400, {"error": f"请求解析失败: {e}"})
            return

        if self.path == '/api/from_matlab':
            self._handle_matlab(payload)
        elif self.path == '/api/from_images':
            self._handle_images(payload)
        elif self.path == '/api/gen_views':
            self._handle_genviews(payload)
        else:
            self._json(404, {"error": "未知接口"})

    # ---- 入口1: matlab ----
    def _handle_matlab(self, payload):
        try:
            m_b64 = payload.get('file')
            if not m_b64:
                raise ValueError("缺少 matlab 文件(file)")
            jid = _new_job()
            outdir = os.path.join(RESULT_DIR, f"job_matlab_{jid[:8]}")
            os.makedirs(outdir, exist_ok=True)
            m_path = save_b64_to_file(m_b64, os.path.join(outdir, "model.m"))
            with JOBS_LOCK:
                JOBS[jid]['urls'] = {}

            def work():
                from web_core_matlab import run_matlab_pipeline
                res = run_matlab_pipeline(m_path, outdir,
                                          progress_cb=_job_progress_cb(jid))
                base = os.path.basename(outdir)
                return {
                    "ok": True,
                    "stats": {"曲线数": res['curves'], "节点数": res['nodes'],
                              "参数": res['params']},
                    "table_csv": f"/results/{base}/root_table.csv",
                    "table_sql": f"/results/{base}/root_table.sql",
                    "image": f"/results/{base}/model_rebuilt_3d.png",
                    "model3d": f"/results/{base}/model3d.json",
                }

            _run_job(jid, work)
            self._json(200, {"ok": True, "job_id": jid, "urls": {}})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ---- 入口2: 图片 ----
    def _handle_images(self, payload):
        try:
            imgs = payload.get('images')   # list of base64
            if not imgs or len(imgs) == 0:
                raise ValueError("缺少图片(images)")
            jid = _new_job()
            outdir = os.path.join(RESULT_DIR, f"job_images_{jid[:8]}")
            os.makedirs(outdir, exist_ok=True)

            # 存所有图 -> imgdir
            imgdir = os.path.join(outdir, "input_imgs")
            os.makedirs(imgdir, exist_ok=True)
            for i, b64 in enumerate(imgs):
                save_b64_to_file(b64, os.path.join(imgdir, f"cam{i}.png"))

            # 可选 pose(位姿文件), 默认用已知相机
            pose = None
            pose_b64 = payload.get('pose')
            if pose_b64:
                pose_data = base64.b64decode(
                    pose_b64.split(',', 1)[1] if pose_b64.startswith('data:') else pose_b64)
                pose = json.loads(pose_data.decode('utf-8'))

            with JOBS_LOCK:
                JOBS[jid]['urls'] = {}

            def work():
                from web_core_images import run_images_pipeline
                res = run_images_pipeline(imgdir, outdir, pose=pose,
                                          progress_cb=_job_progress_cb(jid))
                base = os.path.basename(outdir)
                return {
                    "ok": True,
                    "stats": res['stats'],
                    "points_table": f"/results/{base}/reconstructed_points.csv",
                    "control_table": f"/results/{base}/control_points.csv",
                    "root_table": f"/results/{base}/root_table.csv",
                    "root_sql": f"/results/{base}/root_table.sql",
                    "image": f"/results/{base}/rebuilt_from_table.png",
                    "compare": f"/results/{base}/rebuilt_vs_orig_cam0.png",
                    "model3d": f"/results/{base}/model3d.json",
                }

            _run_job(jid, work)
            self._json(200, {"ok": True, "job_id": jid, "urls": {}})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ---- 生成 24图+位姿 (matlab -> 24图+pose), 逐张落盘供流式展示 ----
    def _handle_genviews(self, payload):
        try:
            m_b64 = payload.get('file')
            if not m_b64:
                raise ValueError("缺少 matlab 文件(file)")
            jid = _new_job()
            outdir = os.path.join(RESULT_DIR, f"job_genviews_{jid[:8]}")
            os.makedirs(outdir, exist_ok=True)
            m_path = save_b64_to_file(m_b64, os.path.join(outdir, "model.m"))
            base = os.path.basename(outdir)
            urls = {"view_dir": f"/results/{base}/views",
                    "pose_file": f"/results/{base}/pose.json",
                    "zip_url": f"/api/zip_views/{base}"}
            with JOBS_LOCK:
                JOBS[jid]['urls'] = urls

            def work():
                from web_core_genviews import generate_views_from_matlab
                res = generate_views_from_matlab(m_path, outdir,
                                                 progress_cb=_job_progress_cb(jid))
                return {
                    "ok": True,
                    "stats": {"曲线数": res['curves'], "节点数": res['nodes'],
                              "视角数": res['n_views'], "每度": f"{res['step_deg']:g}°/张"},
                    "view_dir": urls['view_dir'],
                    "pose_file": urls['pose_file'],
                    "zip_url": urls['zip_url'],
                }

            _run_job(jid, work)
            self._json(200, {"ok": True, "job_id": jid,
                             "n_views": 24, "urls": urls})
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ---- 工具 ----
    def _serve_file(self, path, content_type):
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")  # 页面更新后浏览器不会拿到旧缓存
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._json(404, {"error": "文件不存在"})

    def _serve_zip(self, job):
        """把结果目录 results/<job>/views/*.png + pose.json 打包成 ZIP 下载。"""
        import io, zipfile
        jobdir = os.path.join(RESULT_DIR, job)
        views_dir = os.path.join(jobdir, "views")
        pose_path = os.path.join(jobdir, "pose.json")
        if not os.path.isdir(views_dir):
            self._json(404, {"error": "该任务没有视图文件, 请先运行模块①生成"})
            return
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            # pose.json
            if os.path.isfile(pose_path):
                z.write(pose_path, "pose.json")
            # 24 张图
            for i in range(24):
                png = os.path.join(views_dir, f"cam{i}.png")
                if os.path.isfile(png):
                    z.write(png, f"cam{i}.png")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{job}_views.zip"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # 静默


def start_server(host="127.0.0.1", port=8090):
    """启动服务器并返回实例(桌面版用随机端口时传 port=0)。"""
    srv = ThreadingHTTPServer((host, port), Handler)
    return srv


def main():
    host, port = "127.0.0.1", 8090
    srv = start_server(host, port)
    print(f"Web 服务已启动: http://{host}:{port}")
    print("入口1: /api/from_matlab  ;  入口2: /api/from_images")
    srv.serve_forever()


if __name__ == "__main__":
    main()
