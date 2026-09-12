# -*- coding: utf-8 -*-
"""
desktop_app.py
桌面版入口: pywebview 原生窗口 + 内嵌 web_server。

结果目录重定向到 %LOCALAPPDATA%\\PlantRootRecon\\results,
日志写入 %LOCALAPPDATA%\\PlantRootRecon\\app.log。
直接 `python desktop_app.py` 即可开发调试, 打包见 plantroot.spec。
"""
import os
import sys
import base64
import logging


class Api:
    """暴露给前端 JS (window.pywebview.api) 的桥接接口。

    WebView2 内嵌窗口不支持 Blob 锚点下载, 所以所有文件保存
    统一走原生"另存为"对话框。
    注意: 这里不能把 window 存成公开属性 —— pywebview 启动时会
    递归遍历 js_api 的所有公开属性, 会把整个 WinForms 控件树展开,
    造成无限递归错误风暴(卡死/内存暴涨)。窗口统一用 webview.windows[0]。
    """

    def save_file(self, name, data_b64):
        """前端调用: 弹保存对话框, 把 base64 数据写到用户选的位置。"""
        import webview
        if data_b64.startswith("data:"):
            data_b64 = data_b64.split(",", 1)[1]
        content = base64.b64decode(data_b64)
        path = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG, save_filename=name)
        if not path:
            return "cancelled"
        if isinstance(path, (list, tuple)):
            path = path[0]
        with open(path, "wb") as f:
            f.write(content)
        logging.info("已保存: %s", path)
        return "saved"


def app_data_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "PlantRootRecon")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    if getattr(sys, "frozen", False):
        # windowed 模式下没有 stdout/stderr, 把报错也接进日志(逐行落盘)
        class _FlushOnWrite:
            def __init__(self, path):
                self._fh = open(path, "a", encoding="utf-8")

            def write(self, s):
                self._fh.write(s)
                self._fh.flush()

            def flush(self):
                self._fh.flush()

        sys.stdout = sys.stderr = _FlushOnWrite(log_path)
    return log_path


def main():
    data_dir = app_data_dir()
    log_path = setup_logging(data_dir)
    logging.info("启动桌面版, 数据目录: %s", data_dir)

    import web_server
    web_server.set_result_dir(os.path.join(data_dir, "results"))

    # 随机空闲端口, 避免 8090 被占用
    srv = web_server.start_server(port=0)
    host, port = srv.server_address[:2]
    logging.info("Web 服务已启动: http://%s:%s", host, port)

    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    import webview
    api = Api()
    window = webview.create_window(
        "根系三维重建",
        f"http://{host}:{port}/",
        js_api=api,
        width=1280,
        height=900,
        min_size=(960, 640),
    )
    try:
        webview.start()
    finally:
        logging.info("窗口关闭, 停止服务")
        srv.shutdown()
        logging.info("日志文件: %s", log_path)


if __name__ == "__main__":
    main()
