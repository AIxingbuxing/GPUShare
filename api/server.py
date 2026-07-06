"""Web API 服务器

Flask + Flask-SocketIO，提供 REST API 与 WebSocket 实时推送。
"""
from __future__ import annotations

import os
import sys
import time
import json
import base64
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.node import init_node, get_node
from core import crypto

logger = logging.getLogger(__name__)

# 静态文件目录
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"


def create_app(listen_port: int = 9000,
               bootstrap_nodes: list[tuple[str, int]] | None = None,
               web_port: int = 5000) -> tuple[Flask, SocketIO]:
    """创建 Flask 应用"""
    app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
    app.config["SECRET_KEY"] = crypto.random_nonce_hex(16)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        ping_timeout=60, ping_interval=25)

    # 初始化节点
    node = init_node(listen_port=listen_port, bootstrap_nodes=bootstrap_nodes)

    # ---------- 页面 ----------

    @app.route("/")
    def index():
        return send_from_directory(str(TEMPLATE_DIR), "index.html")

    @app.route("/static/<path:path>")
    def static_file(path):
        return send_from_directory(str(STATIC_DIR), path)

    # ---------- 身份 API ----------

    @app.route("/api/identity/status", methods=["GET"])
    def identity_status():
        return jsonify({
            "ok": True,
            "data": {
                "logged_in": node.is_logged_in,
                "address": node.identity.address if node.identity else None,
                "email": node.identity.email if node.identity else None,
                "nickname": node.identity.nickname if node.identity else None,
            },
        })

    @app.route("/api/identity/register", methods=["POST"])
    def identity_register():
        data = request.get_json() or {}
        try:
            result = node.register(
                email=data["email"],
                nickname=data.get("nickname", ""),
                password=data.get("password", ""),
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            logger.exception("注册失败")
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/identity/login", methods=["POST"])
    def identity_login():
        data = request.get_json() or {}
        try:
            result = node.login(data["email"], data.get("password", ""))
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/identity/recover", methods=["POST"])
    def identity_recover():
        data = request.get_json() or {}
        try:
            result = node.recover(
                mnemonic=data["mnemonic"],
                email=data["email"],
                new_password=data.get("password", ""),
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/identity/logout", methods=["POST"])
    def identity_logout():
        node.logout()
        return jsonify({"ok": True})

    # ---------- GPU API ----------

    @app.route("/api/gpu/info", methods=["GET"])
    def gpu_info():
        return jsonify({"ok": True, "data": node.get_gpu_info()})

    @app.route("/api/gpu/devices", methods=["GET"])
    def gpu_devices():
        return jsonify({"ok": True, "data": node.get_devices()})

    @app.route("/api/gpu/benchmark", methods=["POST"])
    def gpu_benchmark():
        data = request.get_json() or {}
        duration = data.get("duration", 5)
        result = node.run_benchmark(duration)
        return jsonify({"ok": True, "data": result})

    # ---------- 余额 API ----------

    @app.route("/api/balance", methods=["GET"])
    def balance():
        return jsonify({"ok": True, "data": node.get_balance()})

    @app.route("/api/wallet/deposit", methods=["POST"])
    def wallet_deposit():
        data = request.get_json() or {}
        try:
            result = node.deposit(float(data["amount"]), data.get("channel", "wechat"))
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/wallet/transfer", methods=["POST"])
    def wallet_transfer():
        data = request.get_json() or {}
        try:
            result = node.transfer(data["to_addr"], float(data["amount"]))
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/wallet/stake", methods=["POST"])
    def wallet_stake():
        data = request.get_json() or {}
        try:
            result = node.stake(float(data["amount"]))
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/wallet/unstake", methods=["POST"])
    def wallet_unstake():
        data = request.get_json() or {}
        try:
            result = node.unstake(float(data["amount"]))
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ---------- 贡献端 API ----------

    @app.route("/api/contributor/stats", methods=["GET"])
    def contributor_stats():
        return jsonify({"ok": True, "data": node.get_contributor_stats()})

    @app.route("/api/contributor/history", methods=["GET"])
    def contributor_history():
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "data": node.get_contributor_history(limit)})

    @app.route("/api/remote/enable", methods=["POST"])
    def remote_enable():
        try:
            result = node.enable_remote()
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/remote/disable", methods=["POST"])
    def remote_disable():
        try:
            result = node.disable_remote()
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ---------- 账单 API ----------

    @app.route("/api/transactions", methods=["GET"])
    def transactions():
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "data": node.get_tx_history(limit)})

    @app.route("/api/settlements", methods=["GET"])
    def settlements():
        return jsonify({"ok": True, "data": node.get_settlements()})

    # ---------- 任务 API ----------

    @app.route("/api/tasks/submit", methods=["POST"])
    def tasks_submit():
        data = request.get_json() or {}
        try:
            input_data_b64 = data.get("input_data")
            input_data = base64.b64decode(input_data_b64) if input_data_b64 else None
            result = node.submit_task(
                task_type=data["task_type"],
                input_data=input_data,
                input_spec=data.get("input_spec"),
                use_local=data.get("use_local", True),
                use_remote=data.get("use_remote", True),
                local_utilization_limit=data.get("local_utilization_limit"),
                chunk_count=int(data.get("chunk_count", 1)),
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            logger.exception("任务提交失败")
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/tasks", methods=["GET"])
    def tasks_list():
        # 优先使用智能调度器的任务列表
        scheduler = node.smart_scheduler or node.task_scheduler
        if not scheduler:
            return jsonify({"ok": True, "data": []})
        tasks = scheduler.list_tasks()
        return jsonify({"ok": True, "data": [t.to_dict() for t in tasks]})

    # ---------- 设置 API ----------

    @app.route("/api/settings", methods=["GET"])
    def settings_get():
        return jsonify({"ok": True, "data": node.settings})

    @app.route("/api/settings", methods=["POST"])
    def settings_update():
        data = request.get_json() or {}
        result = node.update_settings(**data)
        return jsonify({"ok": True, "data": result})

    @app.route("/api/sharing/start", methods=["POST"])
    def sharing_start():
        try:
            result = node.start_sharing()
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sharing/stop", methods=["POST"])
    def sharing_stop():
        result = node.stop_sharing()
        return jsonify({"ok": True, "data": result})

    # ---------- 节点 API ----------

    @app.route("/api/peers", methods=["GET"])
    def peers():
        return jsonify({"ok": True, "data": node.get_peers()})

    @app.route("/api/peers/sharing", methods=["GET"])
    def peers_sharing():
        return jsonify({"ok": True, "data": node.get_sharing_peers()})

    @app.route("/api/stats", methods=["GET"])
    def stats():
        return jsonify({"ok": True, "data": node.get_stats()})

    @app.route("/api/system", methods=["GET"])
    def system_info():
        return jsonify({"ok": True, "data": node.get_system_info()})

    # ---------- 计算代理 API（供外部软件调用） ----------

    @app.route("/v1/chat/completions", methods=["POST"])
    def openai_chat_completions():
        """OpenAI 兼容推理 API

        本地大模型软件配置 base_url 指向 GPUShare 即可使用：
        base_url = "http://127.0.0.1:5000/v1"

        GPUShare 自动调度本地 GPU / 远程 GPU 执行推理。
        """
        data = request.get_json() or {}
        try:
            model = data.get("model", "llama2")
            messages = data.get("messages", [])
            result = node.compute_inference(
                model=model,
                messages=messages,
                source="openai_api",
                stream=data.get("stream", False),
                temperature=data.get("temperature", 0.7),
            )
            # 转换为 OpenAI 兼容响应格式
            inner = result.get("result", {})
            message = inner.get("message", {})
            return jsonify({
                "id": result.get("request_id", ""),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.get("content", ""),
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "_gpushare": {
                    "schedule": result.get("schedule", "unknown"),
                    "reason": result.get("reason", ""),
                    "latency_ms": result.get("latency_ms", 0),
                },
            })
        except Exception as e:
            logger.exception("推理 API 失败")
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

    @app.route("/api/compute/inference", methods=["POST"])
    def compute_inference():
        """推理计算 API（非 OpenAI 格式的原生接口）"""
        data = request.get_json() or {}
        try:
            result = node.compute_inference(
                model=data.get("model", "default"),
                messages=data.get("messages", []),
                source=data.get("source", "sdk"),
                **{k: v for k, v in data.items() if k not in ("model", "messages", "source")},
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/compute/tensor", methods=["POST"])
    def compute_tensor():
        """张量计算 API"""
        data = request.get_json() or {}
        try:
            result = node.compute_tensor(
                operation=data.get("operation", "matmul"),
                data={k: v for k, v in data.items() if k != "operation"},
                source=data.get("source", "sdk"),
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/compute/stats", methods=["GET"])
    def compute_stats():
        """获取计算代理统计"""
        return jsonify({"ok": True, "data": node.get_compute_stats()})

    @app.route("/api/compute/logs", methods=["GET"])
    def compute_logs():
        """获取调度日志"""
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "data": node.get_compute_logs(limit)})

    @app.route("/api/compute/requests", methods=["GET"])
    def compute_requests():
        """获取最近计算请求"""
        limit = int(request.args.get("limit", 20))
        return jsonify({"ok": True, "data": node.get_compute_requests(limit)})

    # ---------- C2C API ----------

    @app.route("/api/orders", methods=["GET"])
    def orders_list():
        return jsonify({"ok": True, "data": node.get_orders()})

    @app.route("/api/orders/place", methods=["POST"])
    def orders_place():
        data = request.get_json() or {}
        try:
            result = node.place_order(
                order_type=data["type"],
                price=float(data["price"]),
                amount=float(data["amount"]),
            )
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/orders/cancel", methods=["POST"])
    def orders_cancel():
        data = request.get_json() or {}
        try:
            result = node.cancel_order(data["order_id"])
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ---------- WebSocket ----------

    @socketio.on("connect")
    def ws_connect():
        emit("hello", {"msg": "connected", "time": time.time()})

    @socketio.on("subscribe")
    def ws_subscribe(data):
        emit("subscribed", {"channel": data.get("channel", "all")})

    # 周期性推送状态
    def push_status():
        while True:
            try:
                time.sleep(3)
                socketio.emit("status", {
                    "balance": node.get_balance(),
                    "gpu": node.get_gpu_info(),
                    "peers": len(node.get_peers()),
                    "stats": node.get_stats(),
                    "time": time.time(),
                })
            except Exception as e:
                logger.debug(f"推送状态异常: {e}")

    threading.Thread(target=push_status, daemon=True, name="ws-push").start()

    return app, socketio


def run_server(listen_port: int = 9000,
               bootstrap_nodes: list[tuple[str, int]] | None = None,
               web_port: int = 5000):
    """启动 Web 服务器"""
    app, socketio = create_app(listen_port, bootstrap_nodes, web_port)
    logger.info(f"启动 Web 服务器: http://127.0.0.1:{web_port}")
    socketio.run(app, host=config.WEB_CONFIG["host"], port=web_port,
                 debug=False, allow_unsafe_werkzeug=True)
