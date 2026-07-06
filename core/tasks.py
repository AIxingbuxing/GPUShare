"""任务引擎模块

任务类型：图像模糊、矩阵乘法、哈希基准、ML 推理。
支持任务拆分、分布式分发、冗余执行（2 节点）、断点续算、结果哈希验证。
"""
from __future__ import annotations

import io
import json
import time
import hashlib
import logging
import threading
import base64
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable
from enum import Enum

import numpy as np

from . import crypto

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubTaskStatus(str, Enum):
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SubTask:
    """子任务"""
    sub_id: str
    parent_id: str
    index: int                    # 分片索引
    task_type: str
    input_data: bytes             # 输入数据（已序列化）
    input_hash: str               # 输入哈希
    assigned_to: str = ""         # 贡献节点 address
    redundant_node: str = ""      # 冗余节点 address
    status: str = SubTaskStatus.ASSIGNED.value
    result_data: bytes = b""
    result_hash: str = ""
    redundant_result_hash: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    tops_measured: float = 0.0
    duration_sec: float = 0.0
    error: str = ""
    # 容错相关
    retry_count: int = 0          # 已重试次数
    max_retries: int = 3          # 最大重试次数
    last_checkpoint: float = 0.0  # 最近 checkpoint 时间
    checkpoint_data: bytes = b""  # 断点续算数据
    dispatch_history: list = field(default_factory=list)  # 曾分配过的节点列表

    def to_dict(self) -> dict:
        d = asdict(self)
        d["input_data"] = base64.b64encode(self.input_data).decode()
        d["result_data"] = base64.b64encode(self.result_data).decode()
        d["checkpoint_data"] = base64.b64encode(self.checkpoint_data).decode()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubTask":
        d = dict(d)
        d["input_data"] = base64.b64decode(d.get("input_data", ""))
        d["result_data"] = base64.b64decode(d.get("result_data", ""))
        d["checkpoint_data"] = base64.b64decode(d.get("checkpoint_data", ""))
        return cls(**d)


@dataclass
class Task:
    """主任务"""
    task_id: str
    task_type: str
    requester: str                # 需求方 address
    input_spec: dict              # 任务规格（不含大数据，仅参数）
    total_chunks: int = 1
    subtasks: list = field(default_factory=list)  # list[dict]
    status: str = TaskStatus.PENDING.value
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    use_local: bool = True
    use_remote: bool = True
    local_utilization_limit: int = 80
    estimated_tpt: float = 0.0
    actual_tpt: float = 0.0
    result_data: bytes = b""
    error: str = ""

    def __post_init__(self):
        if not self.task_id:
            self.task_id = crypto.sha256_hex(
                f"{self.tasker()}{self.task_type}{time.time()}{crypto.random_nonce_hex()}"
            )
        if not self.created_at:
            self.created_at = time.time()

    def tasker(self):
        return self.requester

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result_data"] = base64.b64encode(self.result_data).decode()
        return d


# ---------- 任务执行器 ----------

class TaskExecutor:
    """任务执行器（贡献端）"""

    def __init__(self, gpu_monitor):
        self.gpu_monitor = gpu_monitor
        self.running_tasks: dict[str, SubTask] = {}
        self._lock = threading.RLock()

    def execute(self, subtask: SubTask, utilization_limit: int = 80,
                progress_cb: Optional[Callable[[float, float], None]] = None) -> SubTask:
        """执行子任务"""
        with self._lock:
            self.running_tasks[subtask.sub_id] = subtask

        subtask.status = SubTaskStatus.RUNNING.value
        subtask.started_at = time.time()

        try:
            if subtask.task_type == "image_blur":
                result = self._exec_image_blur(subtask, utilization_limit, progress_cb)
            elif subtask.task_type == "matrix_mult":
                result = self._exec_matrix_mult(subtask, utilization_limit, progress_cb)
            elif subtask.task_type == "hash_benchmark":
                result = self._exec_hash_benchmark(subtask, utilization_limit, progress_cb)
            elif subtask.task_type == "ml_inference":
                result = self._exec_ml_inference(subtask, utilization_limit, progress_cb)
            else:
                raise ValueError(f"未知任务类型: {subtask.task_type}")

            subtask.result_data = result
            subtask.result_hash = crypto.sha256_hex(result)
            subtask.status = SubTaskStatus.COMPLETED.value
            subtask.completed_at = time.time()
            subtask.duration_sec = subtask.completed_at - subtask.started_at
            # 实测 TOPS（基于结果数据量与耗时）
            subtask.tops_measured = self._measure_tops(subtask)
            logger.info(f"子任务 {subtask.sub_id} 完成，耗时 {subtask.duration_sec:.2f}s, "
                        f"实测 {subtask.tops_measured:.4f} TOPS")
        except Exception as e:
            subtask.status = SubTaskStatus.FAILED.value
            subtask.error = str(e)
            logger.error(f"子任务 {subtask.sub_id} 失败: {e}")
        finally:
            with self._lock:
                self.running_tasks.pop(subtask.sub_id, None)

        return subtask

    def _exec_image_blur(self, subtask: SubTask, util_limit: int,
                         progress_cb) -> bytes:
        """图像模糊处理"""
        from PIL import Image, ImageFilter
        img = Image.open(io.BytesIO(subtask.input_data))
        # 模拟 GPU 加速：多次模糊以体现算力消耗
        radius = 5
        iterations = max(1, int(util_limit / 10))
        for i in range(iterations):
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            if progress_cb:
                progress_cb((i + 1) / iterations, 0)
            time.sleep(0.1)  # 模拟计算时间
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    def _exec_matrix_mult(self, subtask: SubTask, util_limit: int,
                          progress_cb) -> bytes:
        """矩阵乘法"""
        spec = json.loads(subtask.input_data.decode("utf-8"))
        size = spec.get("size", 512)
        seed = spec.get("seed", 42)
        rng = np.random.RandomState(seed)
        a = rng.rand(size, size).astype(np.float32)
        b = rng.rand(size, size).astype(np.float32)
        iterations = max(1, int(util_limit / 20))
        c = None
        for i in range(iterations):
            c = np.dot(a, b)
            a = c  # 链式乘法
            if progress_cb:
                progress_cb((i + 1) / iterations, 0)
        return c.astype(np.float32).tobytes()

    def _exec_hash_benchmark(self, subtask: SubTask, util_limit: int,
                             progress_cb) -> bytes:
        """哈希基准测试"""
        spec = json.loads(subtask.input_data.decode("utf-8"))
        data_size = spec.get("data_size", 1024)  # KB
        iterations = spec.get("iterations", 1000)
        data = b"x" * (data_size * 1024)
        result = b""
        for i in range(iterations):
            result = hashlib.sha256(data + str(i).encode()).digest()
            if i % max(1, iterations // 10) == 0 and progress_cb:
                progress_cb((i + 1) / iterations, 0)
        return result

    def _exec_ml_inference(self, subtask: SubTask, util_limit: int,
                           progress_cb) -> bytes:
        """ML 推理（简化版：矩阵前向传播）"""
        spec = json.loads(subtask.input_data.decode("utf-8"))
        size = spec.get("size", 256)
        layers = spec.get("layers", 3)
        seed = spec.get("seed", 42)
        rng = np.random.RandomState(seed)
        x = rng.rand(size, size).astype(np.float32)
        # 模拟多层前向
        for i in range(layers):
            w = rng.rand(size, size).astype(np.float32) * 0.1
            x = np.tanh(np.dot(x, w))
            if progress_cb:
                progress_cb((i + 1) / layers, 0)
            time.sleep(0.05)
        return x.astype(np.float32).tobytes()

    def _measure_tops(self, subtask: SubTask) -> float:
        """估算本次任务实测 TOPS"""
        # 简化：基于结果数据量与耗时
        data_size = len(subtask.result_data)
        if subtask.duration_sec <= 0:
            return 0.0
        # 粗略估算：每字节结果对应约 1e6 次运算
        ops = data_size * 1e6
        return (ops / subtask.duration_sec) / 1e12


# ---------- 任务调度器（需求端） ----------

class TaskScheduler:
    """任务调度器"""

    def __init__(self, gpu_monitor, executor: TaskExecutor):
        self.gpu_monitor = gpu_monitor
        self.executor = executor
        self.tasks: dict[str, Task] = {}
        self._lock = threading.RLock()

    def create_task(self, task_type: str, requester: str,
                    input_data: bytes, input_spec: dict,
                    use_local: bool = True, use_remote: bool = True,
                    local_utilization_limit: int = 80,
                    chunk_count: int = 1) -> Task:
        """创建任务并拆分子任务"""
        task = Task(
            task_id="",
            task_type=task_type,
            requester=requester,
            input_spec=input_spec,
            total_chunks=chunk_count,
            use_local=use_local,
            use_remote=use_remote,
            local_utilization_limit=local_utilization_limit,
        )
        # 拆分子任务
        chunks = self._split_input(input_data, chunk_count)
        for i, chunk in enumerate(chunks):
            sub = SubTask(
                sub_id=crypto.sha256_hex(f"{task.task_id}-{i}"),
                parent_id=task.task_id,
                index=i,
                task_type=task_type,
                input_data=chunk,
                input_hash=crypto.sha256_hex(chunk),
            )
            task.subtasks.append(sub.to_dict())
        with self._lock:
            self.tasks[task.task_id] = task
        logger.info(f"任务创建: {task.task_id}, 类型={task_type}, 子任务={len(chunks)}")
        return task

    def _split_input(self, data: bytes, chunks: int) -> list[bytes]:
        """拆分输入数据"""
        if chunks <= 1:
            return [data]
        chunk_size = max(1, len(data) // chunks)
        result = []
        for i in range(chunks):
            start = i * chunk_size
            end = start + chunk_size if i < chunks - 1 else len(data)
            result.append(data[start:end])
        return result

    def execute_local(self, task: Task) -> bool:
        """本地执行所有子任务"""
        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()
        results = []
        try:
            for sub_dict in task.subtasks:
                sub = SubTask.from_dict(sub_dict)
                sub = self.executor.execute(sub, task.local_utilization_limit)
                sub_dict_updated = sub.to_dict()
                # 更新到 task
                idx = next(i for i, s in enumerate(task.subtasks) if s["sub_id"] == sub.sub_id)
                task.subtasks[idx] = sub_dict_updated
                if sub.status != SubTaskStatus.COMPLETED.value:
                    task.status = TaskStatus.FAILED.value
                    task.error = f"子任务 {sub.sub_id} 失败: {sub.error}"
                    return False
                results.append(sub.result_data)
            task.result_data = b"".join(results)
            task.status = TaskStatus.COMPLETED.value
            task.completed_at = time.time()
            # 计算实际 TPT
            total_tops = sum(s.get("tops_measured", 0) for s in task.subtasks)
            total_min = (task.completed_at - task.started_at) / 60
            task.actual_tpt = total_tops * total_min
            return True
        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error = str(e)
            return False

    def aggregate_results(self, task: Task, subtask_results: dict[str, bytes]) -> bytes:
        """聚合远程子任务结果"""
        ordered = []
        for sub_dict in sorted(task.subtasks, key=lambda s: s["index"]):
            sub_id = sub_dict["sub_id"]
            if sub_id in subtask_results:
                ordered.append(subtask_results[sub_id])
        return b"".join(ordered)

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return list(self.tasks.values())


# ---------- 任务输入构造工具 ----------

def make_image_blur_input(image_bytes: bytes) -> bytes:
    """构造图像模糊任务输入"""
    return image_bytes


def make_matrix_mult_input(size: int = 512, seed: int = 42) -> bytes:
    """构造矩阵乘法任务输入"""
    return json.dumps({"size": size, "seed": seed}).encode("utf-8")


def make_hash_benchmark_input(data_size_kb: int = 1024, iterations: int = 1000) -> bytes:
    """构造哈希基准任务输入"""
    return json.dumps({"data_size": data_size_kb, "iterations": iterations}).encode("utf-8")


def make_ml_inference_input(size: int = 256, layers: int = 3, seed: int = 42) -> bytes:
    """构造 ML 推理任务输入"""
    return json.dumps({"size": size, "layers": layers, "seed": seed}).encode("utf-8")


# 预置任务输入生成器
TASK_INPUT_BUILDERS = {
    "image_blur": make_image_blur_input,
    "matrix_mult": make_matrix_mult_input,
    "hash_benchmark": make_hash_benchmark_input,
    "ml_inference": make_ml_inference_input,
}


# ---------- 任务规模评估 ----------

# 任务类型 → 估算每 MB 输入所需的 TOPS（用于调度决策）
TASK_COMPLEXITY = {
    "image_blur": 0.5,       # 较轻
    "matrix_mult": 5.0,      # 较重
    "hash_benchmark": 2.0,   # 中等
    "ml_inference": 3.0,     # 中等偏重
}

# 本地 GPU 利用率溢出阈值：超过则自动溢出到远程
LOCAL_OVERFLOW_THRESHOLD = 75  # %
# 远程子任务超时秒数
REMOTE_SUBTASK_TIMEOUT = 60
# 远程子任务最大重试次数
REMOTE_MAX_RETRIES = 3
# checkpoint 间隔秒数
CHECKPOINT_INTERVAL = 10


def estimate_task_tops(task_type: str, input_size_bytes: int) -> float:
    """估算任务所需 TOPS"""
    complexity = TASK_COMPLEXITY.get(task_type, 1.0)
    size_mb = max(0.001, input_size_bytes / (1024 * 1024))
    return complexity * size_mb


# ---------- 智能调度器（无感调用 + 容错） ----------

class SmartScheduler:
    """智能调度器：对需求方完全透明

    调度策略：
    1. 评估任务规模 vs 本地算力
    2. 本地优先：小任务直接本地执行
    3. 溢出策略：本地 GPU 利用率超阈值 → 剩余子任务自动分发到远程
    4. 回退策略：本地失败 → 自动回退到远程
    5. 容错策略：远程节点超时/掉线 → 自动重分发到其他节点
    6. 全程需求方只需 submit_task，无需任何手动操作
    """

    def __init__(self, gpu_monitor, executor: TaskExecutor,
                 p2p_node=None, my_address: str = "",
                 local_utilization_limit: int = 80):
        self.gpu_monitor = gpu_monitor
        self.executor = executor
        self.p2p_node = p2p_node
        self.my_address = my_address
        self.local_utilization_limit = local_utilization_limit
        self.tasks: dict[str, Task] = {}
        self._lock = threading.RLock()

        # 远程子任务结果收集器
        # sub_id -> {"result": SubTask, "received": Event, "contributor": str}
        self._pending_remote: dict[str, dict] = {}
        self._remote_lock = threading.RLock()

        # watchdog 线程
        self._watchdog_running = False
        self._watchdog_thread: Optional[threading.Thread] = None

    def set_p2p(self, p2p_node, my_address: str):
        """延迟注入 P2P 节点"""
        self.p2p_node = p2p_node
        self.my_address = my_address

    def start_watchdog(self):
        """启动看门狗线程，监控远程子任务超时"""
        if self._watchdog_running:
            return
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="task-watchdog"
        )
        self._watchdog_thread.start()
        logger.info("任务看门狗已启动")

    def stop_watchdog(self):
        self._watchdog_running = False

    def _watchdog_loop(self):
        """看门狗循环：检测远程子任务超时并重分发"""
        while self._watchdog_running:
            try:
                time.sleep(5)
                self._check_timeouts()
            except Exception as e:
                logger.error(f"看门狗异常: {e}")

    def _check_timeouts(self):
        """检查所有待回收的远程子任务是否超时"""
        now = time.time()
        with self._remote_lock:
            timed_out = []
            for sub_id, info in list(self._pending_remote.items()):
                sub = info.get("sub")
                if not sub:
                    continue
                if sub.status != SubTaskStatus.RUNNING.value:
                    continue
                if sub.started_at and (now - sub.started_at) > REMOTE_SUBTASK_TIMEOUT:
                    timed_out.append(sub_id)
        for sub_id in timed_out:
            logger.warning(f"远程子任务 {sub_id} 超时，触发重分发")
            self._handle_subtask_timeout(sub_id)

    def _handle_subtask_timeout(self, sub_id: str):
        """处理子任务超时：标记超时 → 自动重分发"""
        with self._remote_lock:
            info = self._pending_remote.get(sub_id)
            if not info:
                return
            sub = info["sub"]
            old_node = sub.assigned_to
            sub.status = SubTaskStatus.TIMEOUT.value
            sub.error = f"节点 {old_node[:12]} 超时"
            sub.dispatch_history.append(old_node)
            sub.retry_count += 1

        # 找到父任务并更新子任务状态
        task = self.get_task(sub.parent_id) if sub.parent_id else None
        if not task:
            return

        # 更新 task 中的 subtask 记录
        for i, sd in enumerate(task.subtasks):
            if sd["sub_id"] == sub_id:
                task.subtasks[i] = sub.to_dict()
                break

        # 尝试重分发
        if sub.retry_count <= sub.max_retries:
            logger.info(f"子任务 {sub_id} 第 {sub.retry_count} 次重分发")
            # 优先回退到本地
            if self._try_local_fallback(sub, task):
                return
            # 否则找其他远程节点
            self._redispatch_to_other_node(sub, exclude=old_node)
        else:
            logger.error(f"子任务 {sub_id} 已达最大重试次数 {sub.max_retries}，放弃")
            sub.status = SubTaskStatus.FAILED.value
            sub.error = f"超过最大重试次数 {sub.max_retries}"

    def _try_local_fallback(self, sub: SubTask, task: Task) -> bool:
        """尝试本地回退执行"""
        try:
            # 重置子任务状态
            sub.status = SubTaskStatus.RUNNING.value
            sub.started_at = time.time()
            sub.error = ""
            sub.assigned_to = self.my_address
            result = self.executor.execute(sub, self.local_utilization_limit)
            if result.status == SubTaskStatus.COMPLETED.value:
                self._collect_remote_result(result)
                logger.info(f"子任务 {sub.sub_id} 本地回退成功")
                return True
        except Exception as e:
            logger.error(f"本地回退失败: {e}")
        return False

    def _redispatch_to_other_node(self, sub: SubTask, exclude: str = ""):
        """重分发到其他远程节点"""
        if not self.p2p_node:
            return
        # 找一个未排除的在线共享节点
        candidates = [
            p for p in self.p2p_node.get_sharing_peers()
            if p.address != exclude and p.address not in sub.dispatch_history
        ]
        if not candidates:
            # 所有节点都试过了，放宽限制
            candidates = [
                p for p in self.p2p_node.get_sharing_peers()
                if p.address != exclude
            ]
        if not candidates:
            logger.error(f"无可用远程节点可重分发子任务 {sub.sub_id}")
            return
        # 择优：按信誉+延迟排序
        candidates.sort(key=lambda p: (-p.reputation, p.latency_ms))
        peer = candidates[0]
        sub.status = SubTaskStatus.RUNNING.value
        sub.started_at = time.time()
        sub.assigned_to = peer.address
        sub.error = ""
        self.p2p_node.send_to_peer(peer.address, "TASK_ASSIGN", {
            "subtask": sub.to_dict(),
            "requester": self.my_address,
            "callback_port": self.p2p_node.listen_port,
            "retry": sub.retry_count,
        })
        logger.info(f"子任务 {sub.sub_id} 已重分发到 {peer.address[:12]}")

    # ---------- 核心调度入口 ----------

    def submit(self, task_type: str, requester: str,
               input_data: bytes, input_spec: dict | None = None,
               chunk_count: int = 1,
               local_utilization_limit: int | None = None,
               force_local: bool = False,
               force_remote: bool = False) -> Task:
        """无感提交：需求方只需提供任务类型和数据，调度器全自动决策

        Args:
            force_local: 强制仅本地（调试用）
            force_remote: 强制仅远程（调试用）
        """
        if local_utilization_limit is not None:
            self.local_utilization_limit = local_utilization_limit

        task = Task(
            task_id="",
            task_type=task_type,
            requester=requester,
            input_spec=input_spec or {"type": task_type},
            total_chunks=chunk_count,
            use_local=not force_remote,
            use_remote=not force_local,
            local_utilization_limit=self.local_utilization_limit,
        )

        # 拆分子任务
        chunks = self._split_input(input_data, chunk_count)
        for i, chunk in enumerate(chunks):
            sub = SubTask(
                sub_id=crypto.sha256_hex(f"{task.task_id}-{i}"),
                parent_id=task.task_id,
                index=i,
                task_type=task_type,
                input_data=chunk,
                input_hash=crypto.sha256_hex(chunk),
            )
            task.subtasks.append(sub.to_dict())

        with self._lock:
            self.tasks[task.task_id] = task

        # 估算任务规模
        est_tops = estimate_task_tops(task_type, len(input_data))
        task.estimated_tpt = est_tops
        logger.info(f"任务创建: {task.task_id}, 类型={task_type}, "
                    f"估算={est_tops:.2f} TOPS, 子任务={len(chunks)}")

        # 自动调度执行
        self._auto_schedule(task)
        return task

    def _auto_schedule(self, task: Task):
        """自动调度：本地优先 → 溢出远程 → 容错补位"""
        task.status = TaskStatus.RUNNING.value
        task.started_at = time.time()

        # 第一步：评估本地 GPU 是否可用
        gpu_info = self.gpu_monitor.get_info()
        local_available = (
            task.use_local
            and gpu_info.available
            and self.local_utilization_limit > 0
        )

        # 第二步：检查远程节点是否可用
        remote_available = False
        if task.use_remote and self.p2p_node:
            sharing = self.p2p_node.get_sharing_peers()
            remote_available = len(sharing) > 0

        # 决策：本地优先，但大任务或本地满载时溢出
        est_tops = task.estimated_tpt
        local_capacity = gpu_info.estimated_tops * (self.local_utilization_limit / 100.0)
        # 如果估算需求超过本地容量 1.5 倍，且有远程节点 → 混合模式
        use_mixed = (
            local_available
            and remote_available
            and est_tops > local_capacity * 1.5
            and len(task.subtasks) > 1
        )

        if use_mixed:
            logger.info(f"任务 {task.task_id} 采用混合模式：本地+远程")
            self._execute_mixed(task)
        elif local_available:
            logger.info(f"任务 {task.task_id} 本地执行")
            self._execute_local_with_overflow(task)
        elif remote_available:
            logger.info(f"任务 {task.task_id} 远程执行")
            self._execute_remote(task)
        else:
            # 无可用算力，最后尝试本地（CPU 模式也行）
            logger.warning(f"任务 {task.task_id} 无远程节点，回退本地")
            self._execute_local_with_overflow(task)

    def _execute_local_with_overflow(self, task: Task):
        """本地执行，负载溢出时自动拉远程"""
        results = []
        for i, sub_dict in enumerate(task.subtasks):
            sub = SubTask.from_dict(sub_dict)

            # 检查本地 GPU 当前负载
            gpu_info = self.gpu_monitor.get_info()
            if (task.use_remote and self.p2p_node
                    and gpu_info.utilization > LOCAL_OVERFLOW_THRESHOLD
                    and i < len(task.subtasks) - 1):
                # 本地负载过高，剩余子任务分发到远程
                logger.info(f"本地负载 {gpu_info.utilization:.0f}% 超阈值，"
                            f"子任务 {sub.sub_id} 溢出到远程")
                self._dispatch_subtask_remote(sub, task)
                results.append(None)  # 占位，远程结果异步回收
                continue

            # 本地执行
            sub.assigned_to = self.my_address
            sub = self.executor.execute(sub, task.local_utilization_limit)
            task.subtasks[i] = sub.to_dict()

            if sub.status != SubTaskStatus.COMPLETED.value:
                # 本地失败，尝试远程回退
                if task.use_remote and self.p2p_node:
                    logger.warning(f"本地子任务 {sub.sub_id} 失败，回退远程")
                    sub.status = SubTaskStatus.ASSIGNED.value
                    sub.error = ""
                    sub.retry_count = 0
                    self._dispatch_subtask_remote(sub, task)
                    results.append(None)
                else:
                    task.status = TaskStatus.FAILED.value
                    task.error = f"子任务 {sub.sub_id} 失败: {sub.error}"
                    return
            else:
                results.append(sub.result_data)

        # 检查是否有远程子任务待回收
        has_remote = any(r is None for r in results)
        if has_remote:
            self._wait_remote_completion(task, results)
        else:
            task.result_data = b"".join(results)
            task.status = TaskStatus.COMPLETED.value
            task.completed_at = time.time()
            self._calc_actual_tpt(task)

    def _execute_mixed(self, task: Task):
        """混合模式：部分本地 + 部分远程"""
        mid = len(task.subtasks) // 2
        results = [None] * len(task.subtasks)

        # 前半本地
        for i in range(mid):
            sub = SubTask.from_dict(task.subtasks[i])
            sub.assigned_to = self.my_address
            sub = self.executor.execute(sub, task.local_utilization_limit)
            task.subtasks[i] = sub.to_dict()
            if sub.status == SubTaskStatus.COMPLETED.value:
                results[i] = sub.result_data
            else:
                # 本地失败 → 远程补位
                sub.status = SubTaskStatus.ASSIGNED.value
                sub.error = ""
                self._dispatch_subtask_remote(sub, task)

        # 后半远程
        for i in range(mid, len(task.subtasks)):
            sub = SubTask.from_dict(task.subtasks[i])
            self._dispatch_subtask_remote(sub, task)

        self._wait_remote_completion(task, results)

    def _execute_remote(self, task: Task):
        """纯远程执行"""
        results = [None] * len(task.subtasks)
        for i, sub_dict in enumerate(task.subtasks):
            sub = SubTask.from_dict(sub_dict)
            self._dispatch_subtask_remote(sub, task)
        self._wait_remote_completion(task, results)

    def _dispatch_subtask_remote(self, sub: SubTask, task: Task):
        """分发子任务到远程节点（择优 + 注册超时监控）"""
        if not self.p2p_node:
            sub.status = SubTaskStatus.FAILED.value
            sub.error = "P2P 未启动"
            return

        sharing = self.p2p_node.get_sharing_peers()
        if not sharing:
            # 无远程节点，本地兜底
            sub.assigned_to = self.my_address
            sub = self.executor.execute(sub, self.local_utilization_limit)
            self._collect_remote_result(sub)
            return

        # 择优：信誉高 + 延迟低 + 未在 dispatch_history 中
        available = [p for p in sharing if p.address not in sub.dispatch_history]
        if not available:
            available = sharing
        available.sort(key=lambda p: (-p.reputation, p.latency_ms))
        peer = available[0]

        sub.assigned_to = peer.address
        sub.status = SubTaskStatus.RUNNING.value
        sub.started_at = time.time()

        # 注册到待回收列表
        with self._remote_lock:
            self._pending_remote[sub.sub_id] = {
                "sub": sub,
                "task_id": task.task_id,
                "received": threading.Event(),
            }

        # 发送任务
        self.p2p_node.send_to_peer(peer.address, "TASK_ASSIGN", {
            "subtask": sub.to_dict(),
            "requester": self.my_address,
            "callback_port": self.p2p_node.listen_port,
        })
        logger.info(f"子任务 {sub.sub_id} 已分发到 {peer.address[:12]}")

    def _wait_remote_completion(self, task: Task, results: list):
        """等待所有远程子任务完成（带超时）"""
        total_timeout = REMOTE_SUBTASK_TIMEOUT * (REMOTE_MAX_RETRIES + 1)
        deadline = time.time() + total_timeout

        while time.time() < deadline:
            # 检查所有子任务状态
            all_done = True
            for i, sub_dict in enumerate(task.subtasks):
                sub = SubTask.from_dict(sub_dict)
                if sub.status in (SubTaskStatus.COMPLETED.value,
                                  SubTaskStatus.FAILED.value):
                    if results[i] is None and sub.status == SubTaskStatus.COMPLETED.value:
                        results[i] = sub.result_data
                else:
                    all_done = False
                    # 同步 task 中的最新状态
                    task.subtasks[i] = sub.to_dict()

            if all_done:
                break
            time.sleep(1)

        # 检查最终状态
        failed_subs = []
        for i, sub_dict in enumerate(task.subtasks):
            sub = SubTask.from_dict(sub_dict)
            if sub.status == SubTaskStatus.COMPLETED.value:
                if results[i] is None:
                    results[i] = sub.result_data
            else:
                failed_subs.append(sub.sub_id)

        if failed_subs:
            # 有子任务最终失败，尝试本地兜底
            for i, sub_dict in enumerate(task.subtasks):
                sub = SubTask.from_dict(sub_dict)
                if sub.status != SubTaskStatus.COMPLETED.value and results[i] is None:
                    logger.warning(f"子任务 {sub.sub_id} 最终失败，本地兜底")
                    sub.assigned_to = self.my_address
                    sub.status = SubTaskStatus.RUNNING.value
                    sub.error = ""
                    result = self.executor.execute(sub, self.local_utilization_limit)
                    task.subtasks[i] = result.to_dict()
                    if result.status == SubTaskStatus.COMPLETED.value:
                        results[i] = result.result_data
                    else:
                        task.status = TaskStatus.FAILED.value
                        task.error = f"子任务 {sub.sub_id} 最终失败"
                        return

        # 全部完成
        task.result_data = b"".join(r for r in results if r is not None)
        task.status = TaskStatus.COMPLETED.value
        task.completed_at = time.time()
        self._calc_actual_tpt(task)
        logger.info(f"任务 {task.task_id} 全部完成")

    def _calc_actual_tpt(self, task: Task):
        """计算实际 TPT 消耗"""
        total_tops = sum(
            SubTask.from_dict(s).tops_measured
            for s in task.subtasks
        )
        total_min = (task.completed_at - task.started_at) / 60
        task.actual_tpt = total_tops * total_min

    # ---------- 远程结果回收 ----------

    def collect_result(self, sub: SubTask, contributor: str):
        """收集远程子任务结果（由 P2P 消息处理器调用）"""
        with self._remote_lock:
            info = self._pending_remote.get(sub.sub_id)
            if info:
                info["sub"] = sub
                info["received"].set()

        # 更新到父任务
        task = self.get_task(sub.parent_id) if sub.parent_id else None
        if task:
            for i, sd in enumerate(task.subtasks):
                if sd["sub_id"] == sub.sub_id:
                    task.subtasks[i] = sub.to_dict()
                    break

        # 清理待回收
        with self._remote_lock:
            self._pending_remote.pop(sub.sub_id, None)

        logger.info(f"远程子任务 {sub.sub_id} 结果已回收 (贡献者 {contributor[:12]})")

    def _collect_remote_result(self, sub: SubTask):
        """内部调用：直接收集结果"""
        with self._remote_lock:
            info = self._pending_remote.get(sub.sub_id)
            if info:
                info["sub"] = sub
                info["received"].set()
            self._pending_remote.pop(sub.sub_id, None)

        task = self.get_task(sub.parent_id) if sub.parent_id else None
        if task:
            for i, sd in enumerate(task.subtasks):
                if sd["sub_id"] == sub.sub_id:
                    task.subtasks[i] = sub.to_dict()
                    break

    # ---------- 工具方法 ----------

    def _split_input(self, data: bytes, chunks: int) -> list[bytes]:
        """拆分输入数据

        智能拆分策略：
        - JSON 输入：不拆分数据本身，而是为每个分片生成不同的 seed（保证并行计算不重复）
        - 二进制输入（图像等）：不拆分（拆分会破坏文件格式），每个分片获得完整副本，
          执行器内部根据 chunk_index 使用不同参数（如不同的模糊 radius）
        """
        if chunks <= 1:
            return [data]
        # 尝试解析为 JSON
        try:
            spec = json.loads(data.decode("utf-8"))
            # JSON 输入：每个分片获得完整 spec，但 seed 不同
            result = []
            base_seed = spec.get("seed", 42)
            for i in range(chunks):
                chunk_spec = dict(spec)
                chunk_spec["seed"] = base_seed + i * 1000
                chunk_spec["chunk_index"] = i
                chunk_spec["chunk_total"] = chunks
                result.append(json.dumps(chunk_spec).encode("utf-8"))
            return result
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 二进制输入：不拆分（拆分会破坏文件格式），每个分片获得完整副本
            # 执行器会根据子任务的 index 使用不同参数
            return [data for _ in range(chunks)]

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return list(self.tasks.values())

    def get_pending_remote_count(self) -> int:
        with self._remote_lock:
            return len(self._pending_remote)
