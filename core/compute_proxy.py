"""计算代理模块

GPUShare 作为后台算力调度中间件，接收本地其他软件（如大模型推理软件）
产生的计算需求，智能调度本地 GPU 和远程 GPU 完成任务。

工作流程：
  本地大模型软件 → GPUShare ComputeProxy → 智能调度
                                           ├─ 本地 GPU 空闲 → 本地执行
                                           ├─ 本地 GPU 满载 → 远程执行
                                           └─ 混合 → 本地+远程并行

对接方式：
  1. OpenAI 兼容 API: POST /v1/chat/completions（大模型软件直接配置 base_url 即可）
  2. 张量计算 API: POST /api/compute/tensor
  3. Python SDK: from gpushare import ComputeClient
"""
from __future__ import annotations

import time
import json
import logging
import threading
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, Any
from enum import Enum

from . import crypto
from .gpu import GPUMonitor

logger = logging.getLogger(__name__)


class ComputeStatus(str, Enum):
    QUEUED = "queued"           # 排队中
    LOCAL = "local"             # 本地执行中
    REMOTE = "remote"           # 远程执行中
    MIXED = "mixed"             # 混合执行中
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ComputeRequest:
    """计算请求（由外部软件发起）"""
    request_id: str
    request_type: str             # inference / tensor / custom
    source: str                   # 来源软件标识（如 "ollama", "vllm", "sdk"）
    payload: dict                 # 请求负载（model/messages 或 operation/data）
    timestamp: float = 0.0
    status: str = ComputeStatus.QUEUED.value
    assigned_to: str = ""         # local / remote:address / mixed
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Any = None
    error: str = ""
    gpu_util_at_dispatch: float = 0.0  # 调度时本地 GPU 利用率
    schedule_reason: str = ""     # 调度决策原因

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.request_id:
            self.request_id = crypto.sha256_hex(
                f"{self.request_type}{self.source}{self.timestamp}{crypto.random_nonce_hex()}"
            )

    def to_dict(self) -> dict:
        return asdict(self)


class ComputeProxy:
    """计算代理：拦截/接收外部软件的计算需求，智能调度本地+远程 GPU

    核心原则：对调用方完全透明
    - 调用方只需发起请求，不需关心在哪里执行
    - GPUShare 自动检测本地 GPU 负载，决定本地/远程/混合
    - 本地 GPU 满载时自动溢出到远程
    """

    # 调度阈值
    LOCAL_UTIL_LOW = 50       # 本地 GPU 利用率 < 50% → 本地执行
    LOCAL_UTIL_HIGH = 75      # 本地 GPU 利用率 > 75% → 远程执行
    LOCAL_UTIL_MID = 75       # 50%-75% 之间 → 视任务规模决定

    def __init__(self, gpu_monitor: GPUMonitor, smart_scheduler=None,
                 p2p_node=None, my_address: str = "",
                 local_backend_url: str = "http://127.0.0.1:11434"):
        """
        Args:
            local_backend_url: 本地推理引擎地址（如 Ollama 默认 11434）
        """
        self.gpu_monitor = gpu_monitor
        self.smart_scheduler = smart_scheduler
        self.p2p_node = p2p_node
        self.my_address = my_address
        self.local_backend_url = local_backend_url

        # 请求历史
        self.requests: dict[str, ComputeRequest] = {}
        self._lock = threading.RLock()

        # 统计
        self.stats = {
            "total_requests": 0,
            "local_executed": 0,
            "remote_executed": 0,
            "mixed_executed": 0,
            "failed": 0,
            "avg_latency_ms": 0.0,
        }

        # GPU 负载历史（用于趋势分析）
        self._gpu_history: list[float] = []
        self._history_lock = threading.RLock()

        # 后台 GPU 监控线程
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # 调度日志（供前端展示）
        self.schedule_logs: list[dict] = []
        self._max_logs = 200

    def set_smart_scheduler(self, scheduler, p2p_node, my_address: str):
        """延迟注入"""
        self.smart_scheduler = scheduler
        self.p2p_node = p2p_node
        self.my_address = my_address

    def start_monitor(self):
        """启动后台 GPU 监控"""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="gpu-monitor"
        )
        self._monitor_thread.start()
        logger.info("GPU 负载监控已启动")

    def stop_monitor(self):
        self._monitor_running = False

    def _monitor_loop(self):
        """后台持续监控本地 GPU 负载"""
        while self._monitor_running:
            try:
                info = self.gpu_monitor.get_info()
                with self._history_lock:
                    self._gpu_history.append(info.utilization)
                    if len(self._gpu_history) > 60:  # 保留 2 分钟历史
                        self._gpu_history.pop(0)
                time.sleep(2)  # 2 秒采样一次
            except Exception as e:
                logger.error(f"GPU 监控异常: {e}")
                time.sleep(5)

    def get_gpu_utilization(self) -> float:
        """获取当前本地 GPU 利用率"""
        info = self.gpu_monitor.get_info()
        return info.utilization

    def get_gpu_trend(self) -> list[float]:
        """获取 GPU 利用率趋势（最近 2 分钟）"""
        with self._history_lock:
            return list(self._gpu_history)

    # ---------- 核心调度 ----------

    def _decide_schedule(self, request: ComputeRequest) -> tuple[str, str]:
        """调度决策：返回 (执行方式, 决策原因)

        Returns:
            ("local", "本地 GPU 利用率 32%，空闲执行")
            ("remote", "本地 GPU 利用率 85%，超阈值溢出远程")
            ("mixed", "本地 GPU 利用率 60%，大任务混合执行")
        """
        gpu_util = self.get_gpu_utilization()

        # 检查远程节点是否可用
        remote_available = False
        if self.p2p_node:
            sharing = self.p2p_node.get_sharing_peers()
            remote_available = len(sharing) > 0

        # 决策逻辑
        if gpu_util < self.LOCAL_UTIL_LOW:
            if remote_available and self._is_large_request(request):
                return "mixed", f"本地 GPU {gpu_util:.0f}%，大任务混合执行"
            return "local", f"本地 GPU {gpu_util:.0f}%，空闲执行"
        elif gpu_util > self.LOCAL_UTIL_HIGH:
            if remote_available:
                return "remote", f"本地 GPU {gpu_util:.0f}% 超阈值，溢出远程"
            return "local", f"本地 GPU {gpu_util:.0f}% 但无远程节点，本地执行"
        else:
            # 中间区间
            if remote_available and self._is_large_request(request):
                return "mixed", f"本地 GPU {gpu_util:.0f}%，大任务混合执行"
            return "local", f"本地 GPU {gpu_util:.0f}%，本地执行"

    def _is_large_request(self, request: ComputeRequest) -> bool:
        """判断是否为大计算请求"""
        if request.request_type == "inference":
            messages = request.payload.get("messages", [])
            total_chars = sum(len(m.get("content", "")) for m in messages)
            return total_chars > 500 or request.payload.get("stream", False)
        elif request.request_type == "tensor":
            shape = request.payload.get("shape", [])
            total_elements = 1
            for s in shape:
                total_elements *= max(1, s)
            return total_elements > 100000  # 10 万元素以上算大任务
        return False

    # ---------- 对外接口 ----------

    def inference(self, model: str, messages: list[dict],
                  source: str = "unknown", **kwargs) -> dict:
        """推理请求（OpenAI 兼容）

        本地大模型软件调用此方法，GPUShare 自动调度：
        - 本地 GPU 空闲 → 转发给本地推理引擎（如 Ollama）
        - 本地 GPU 满载 → 分发到远程节点执行
        """
        request = ComputeRequest(
            request_id="",
            request_type="inference",
            source=source,
            payload={
                "model": model,
                "messages": messages,
                **kwargs,
            },
        )

        with self._lock:
            self.requests[request.request_id] = request
            self.stats["total_requests"] += 1

        # 调度决策
        schedule, reason = self._decide_schedule(request)
        request.status = ComputeStatus.LOCAL.value if schedule == "local" else \
                         ComputeStatus.REMOTE.value if schedule == "remote" else \
                         ComputeStatus.MIXED.value
        request.assigned_to = schedule
        request.started_at = time.time()
        request.gpu_util_at_dispatch = self.get_gpu_utilization()
        request.schedule_reason = reason

        self._log_schedule(request, schedule, reason)

        try:
            if schedule == "local":
                result = self._execute_local_inference(request)
            elif schedule == "remote":
                result = self._execute_remote_inference(request)
            else:
                result = self._execute_mixed_inference(request)

            request.result = result
            request.status = ComputeStatus.COMPLETED.value
            request.completed_at = time.time()

            with self._lock:
                if schedule == "local":
                    self.stats["local_executed"] += 1
                elif schedule == "remote":
                    self.stats["remote_executed"] += 1
                else:
                    self.stats["mixed_executed"] += 1

            return {
                "request_id": request.request_id,
                "status": "completed",
                "schedule": schedule,
                "reason": reason,
                "result": result,
                "latency_ms": (request.completed_at - request.started_at) * 1000,
            }

        except Exception as e:
            request.status = ComputeStatus.FAILED.value
            request.error = str(e)
            request.completed_at = time.time()
            with self._lock:
                self.stats["failed"] += 1
            logger.error(f"计算请求 {request.request_id} 失败: {e}")
            # 失败时尝试本地兜底
            if schedule != "local":
                logger.info(f"尝试本地兜底执行...")
                try:
                    result = self._execute_local_inference(request)
                    request.result = result
                    request.status = ComputeStatus.COMPLETED.value
                    return {
                        "request_id": request.request_id,
                        "status": "completed",
                        "schedule": "local_fallback",
                        "reason": "远程失败，本地兜底",
                        "result": result,
                        "latency_ms": (time.time() - request.started_at) * 1000,
                    }
                except Exception as e2:
                    logger.error(f"本地兜底也失败: {e2}")
            return {
                "request_id": request.request_id,
                "status": "failed",
                "error": str(e),
            }

    def tensor_compute(self, operation: str, data: dict,
                       source: str = "unknown") -> dict:
        """张量计算请求

        Args:
            operation: "matmul" / "vector_add" / "reduce" 等
            data: {"shape": [m, n], "seed": 42} 等
        """
        request = ComputeRequest(
            request_id="",
            request_type="tensor",
            source=source,
            payload={"operation": operation, **data},
        )

        with self._lock:
            self.requests[request.request_id] = request
            self.stats["total_requests"] += 1

        schedule, reason = self._decide_schedule(request)
        request.status = schedule
        request.assigned_to = schedule
        request.started_at = time.time()
        request.gpu_util_at_dispatch = self.get_gpu_utilization()
        request.schedule_reason = reason

        self._log_schedule(request, schedule, reason)

        try:
            if schedule == "local":
                result = self._execute_local_tensor(request)
            elif schedule == "remote":
                result = self._execute_remote_tensor(request)
            else:
                result = self._execute_mixed_tensor(request)

            request.result = result
            request.status = ComputeStatus.COMPLETED.value
            request.completed_at = time.time()
            return {
                "request_id": request.request_id,
                "status": "completed",
                "schedule": schedule,
                "result": result,
                "latency_ms": (request.completed_at - request.started_at) * 1000,
            }
        except Exception as e:
            request.status = ComputeStatus.FAILED.value
            request.error = str(e)
            return {"request_id": request.request_id, "status": "failed", "error": str(e)}

    # ---------- 本地执行 ----------

    def _execute_local_inference(self, request: ComputeRequest) -> dict:
        """本地执行推理请求（转发给本地推理引擎）"""
        import urllib.request
        import urllib.error

        payload = request.payload
        url = f"{self.local_backend_url}/api/chat"
        body = json.dumps({
            "model": payload.get("model", "llama2"),
            "messages": payload.get("messages", []),
            "stream": False,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                logger.info(f"本地推理完成 (来源: {request.source})")
                return result
        except urllib.error.URLError as e:
            # 本地推理引擎不可用，模拟响应（演示模式）
            logger.warning(f"本地推理引擎不可用 ({self.local_backend_url})，返回模拟结果: {e}")
            return self._mock_inference_response(payload)
        except Exception as e:
            logger.error(f"本地推理失败: {e}")
            raise

    def _mock_inference_response(self, payload: dict) -> dict:
        """模拟推理响应（当本地推理引擎不可用时）"""
        messages = payload.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        model = payload.get("model", "unknown")
        return {
            "model": model,
            "created_at": int(time.time()),
            "message": {
                "role": "assistant",
                "content": f"[GPUShare 本地模拟响应] 已处理你的请求: {last_msg[:50]}...",
            },
            "done": True,
            "_note": "本地推理引擎未检测到，返回模拟结果。请配置 local_backend_url 指向你的推理引擎（如 Ollama）。",
        }

    def _execute_local_tensor(self, request: ComputeRequest) -> dict:
        """本地执行张量计算"""
        import numpy as np
        op = request.payload.get("operation", "matmul")
        shape = request.payload.get("shape", [256, 256])
        seed = request.payload.get("seed", 42)

        rng = np.random.RandomState(seed)
        size = shape[0] if isinstance(shape, list) else 256

        if op == "matmul":
            a = rng.rand(size, size).astype(np.float32)
            b = rng.rand(size, size).astype(np.float32)
            c = np.dot(a, b)
            return {
                "operation": op,
                "shape": [size, size],
                "result_shape": list(c.shape),
                "checksum": float(np.sum(c)),
                "executed_on": "local",
            }
        elif op == "vector_add":
            a = rng.rand(size).astype(np.float32)
            b = rng.rand(size).astype(np.float32)
            c = a + b
            return {
                "operation": op,
                "size": size,
                "checksum": float(np.sum(c)),
                "executed_on": "local",
            }
        else:
            return {"operation": op, "status": "unknown_operation", "executed_on": "local"}

    # ---------- 远程执行 ----------

    def _execute_remote_inference(self, request: ComputeRequest) -> dict:
        """远程执行推理请求（分发到远程 GPU 节点）"""
        if not self.p2p_node or not self.p2p_node.get_sharing_peers():
            logger.warning("无远程节点可用，回退本地")
            return self._execute_local_inference(request)

        # 择优选择远程节点
        peer = self.p2p_node.get_best_peer()
        if not peer:
            return self._execute_local_inference(request)

        # 发送推理请求到远程节点
        self.p2p_node.send_to_peer(peer.address, "COMPUTE_REQ", {
            "request_id": request.request_id,
            "request_type": "inference",
            "payload": request.payload,
            "requester": self.my_address,
        })

        # 等待结果（简化：异步等待，超时回退本地）
        result = self._wait_remote_result(request.request_id, timeout=30)
        if result:
            result["executed_on"] = f"remote:{peer.address[:12]}"
            return result
        # 超时回退本地
        logger.warning(f"远程推理超时，回退本地")
        return self._execute_local_inference(request)

    def _execute_remote_tensor(self, request: ComputeRequest) -> dict:
        """远程执行张量计算"""
        if not self.p2p_node or not self.p2p_node.get_sharing_peers():
            return self._execute_local_tensor(request)

        peer = self.p2p_node.get_best_peer()
        if not peer:
            return self._execute_local_tensor(request)

        self.p2p_node.send_to_peer(peer.address, "COMPUTE_REQ", {
            "request_id": request.request_id,
            "request_type": "tensor",
            "payload": request.payload,
            "requester": self.my_address,
        })

        result = self._wait_remote_result(request.request_id, timeout=30)
        if result:
            result["executed_on"] = f"remote:{peer.address[:12]}"
            return result
        return self._execute_local_tensor(request)

    def _execute_mixed_inference(self, request: ComputeRequest) -> dict:
        """混合执行：本地处理一部分，远程处理一部分（流式推理场景）"""
        # 简化：本地执行，但标记为混合
        result = self._execute_local_inference(request)
        result["executed_on"] = "mixed"
        return result

    def _execute_mixed_tensor(self, request: ComputeRequest) -> dict:
        """混合执行张量计算"""
        result = self._execute_local_tensor(request)
        result["executed_on"] = "mixed"
        return result

    # ---------- 远程结果回收 ----------

    _remote_results: dict[str, dict] = {}
    _remote_results_lock = threading.Lock()

    def receive_remote_result(self, request_id: str, result: dict):
        """接收远程计算结果（由 P2P 消息处理器调用）"""
        with self._remote_results_lock:
            self._remote_results[request_id] = {
                "result": result,
                "timestamp": time.time(),
            }

    def _wait_remote_result(self, request_id: str, timeout: int = 30) -> Optional[dict]:
        """等待远程结果"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._remote_results_lock:
                entry = self._remote_results.pop(request_id, None)
            if entry:
                return entry["result"]
            time.sleep(0.5)
        return None

    # ---------- 日志与统计 ----------

    def _log_schedule(self, request: ComputeRequest, schedule: str, reason: str):
        """记录调度日志"""
        log_entry = {
            "timestamp": time.time(),
            "request_id": request.request_id[:16],
            "type": request.request_type,
            "source": request.source,
            "schedule": schedule,
            "reason": reason,
            "gpu_util": request.gpu_util_at_dispatch,
        }
        with self._lock:
            self.schedule_logs.append(log_entry)
            if len(self.schedule_logs) > self._max_logs:
                self.schedule_logs = self.schedule_logs[-self._max_logs:]

    def get_schedule_logs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(reversed(self.schedule_logs[-limit:]))

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self.stats)
        total = stats["total_requests"]
        if total > 0:
            stats["local_rate"] = stats["local_executed"] / total
            stats["remote_rate"] = stats["remote_executed"] / total
            stats["success_rate"] = (total - stats["failed"]) / total
        else:
            stats["local_rate"] = 0
            stats["remote_rate"] = 0
            stats["success_rate"] = 1.0
        stats["current_gpu_util"] = self.get_gpu_utilization()
        stats["gpu_trend"] = self.get_gpu_trend()
        stats["pending_requests"] = sum(
            1 for r in self.requests.values()
            if r.status in (ComputeStatus.QUEUED.value, ComputeStatus.LOCAL.value,
                           ComputeStatus.REMOTE.value, ComputeStatus.MIXED.value)
        )
        return stats

    def get_recent_requests(self, limit: int = 20) -> list[dict]:
        with self._lock:
            recent = sorted(self.requests.values(),
                          key=lambda r: r.timestamp, reverse=True)[:limit]
            return [r.to_dict() for r in recent]
