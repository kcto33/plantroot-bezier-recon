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
import json
import base64
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 生成的结果放在这个目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
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


class Handler(BaseHTTPRequestHandler):
    # ---- 静态资源(前端) ----
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_file("templates/index.html", "text/html; charset=utf-8")
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
            outdir = os.path.join(RESULT_DIR, "job_matlab")
            os.makedirs(outdir, exist_ok=True)
            m_path = save_b64_to_file(m_b64, os.path.join(outdir, "model.m"))

            # 延迟导入, 避免服务器启动即计算
            from web_core_matlab import run_matlab_pipeline
            res = run_matlab_pipeline(m_path, outdir)

            self._json(200, {
                "ok": True,
                "stats": {"曲线数": res['curves'], "节点数": res['nodes'],
                          "参数": res['params']},
                "table_csv": "/results/job_matlab/root_table.csv",
                "table_sql": "/results/job_matlab/root_table.sql",
                "image": "/results/job_matlab/model_rebuilt_3d.png",
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ---- 入口2: 图片 ----
    def _handle_images(self, payload):
        try:
            imgs = payload.get('images')   # list of base64
            if not imgs or len(imgs) == 0:
                raise ValueError("缺少图片(images)")
            outdir = os.path.join(RESULT_DIR, "job_images")
            os.makedirs(outdir, exist_ok=True)

            # 存24张图 -> imgdir
            imgdir = os.path.join(outdir, "input_imgs")
            os.makedirs(imgdir, exist_ok=True)
            for i, b64 in enumerate(imgs):
                save_b64_to_file(b64, os.path.join(imgdir, f"cam{i}.png"))

            # 可选 pose(位姿文件), 默认用已知相机
            pose = None
            pose_b64 = payload.get('pose')
            if pose_b64:
                import io
                pose_data = base64.b64decode(
                    pose_b64.split(',', 1)[1] if pose_b64.startswith('data:') else pose_b64)
                pose = json.loads(pose_data.decode('utf-8'))

            from web_core_images import run_images_pipeline
            res = run_images_pipeline(imgdir, outdir, pose=pose)

            self._json(200, {
                "ok": True,
                "stats": res['stats'],
                "points_table": "/results/job_images/reconstructed_points.csv",
                "control_table": "/results/job_images/control_points.csv",
                "image": "/results/job_images/rebuilt_from_table.png",
                "compare": "/results/job_images/rebuilt_vs_orig_cam0.png",
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            self._json(500, {"error": str(e)})

    # ---- 生成 24图+位姿 (matlab -> 24图+pose) ----
    def _handle_genviews(self, payload):
        try:
            m_b64 = payload.get('file')
            if not m_b64:
                raise ValueError("缺少 matlab 文件(file)")
            outdir = os.path.join(RESULT_DIR, "job_genviews")
            os.makedirs(outdir, exist_ok=True)
            m_path = save_b64_to_file(m_b64, os.path.join(outdir, "model.m"))

            from web_core_genviews import generate_views_from_matlab
            res = generate_views_from_matlab(m_path, outdir)

            self._json(200, {
                "ok": True,
                "stats": {"曲线数": res['curves'], "节点数": res['nodes'],
                          "视角数": res['n_views'], "每度": f"{res['step_deg']:g}°/张"},
                "view_dir": "/results/job_genviews/views",
                "pose_file": "/results/job_genviews/pose.json",
                "zip_url": "/api/zip_views/job_genviews",
            })
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


def main():
    host, port = "127.0.0.1", 8090
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Web 服务已启动: http://{host}:{port}")
    print("入口1: /api/from_matlab  ;  入口2: /api/from_images")
    srv.serve_forever()


if __name__ == "__main__":
    main()
