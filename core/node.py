"""节点协调器

整合 identity / ledger / p2p / gpu / tasks / settlement / witness 所有模块。
对外提供统一 API，被 web api 调用。
"""
from __future__ import annotations

import os
import time
import json
import base64
import logging
import threading
from pathlib import Path
from typing import Optional

from . import crypto
from .identity import Identity, IdentityManager
from .ledger import Ledger, Transaction, TxType
from .p2p import P2PNode, PeerInfo
from .gpu import GPUMonitor, get_monitor
from .tasks import (
    TaskScheduler, TaskExecutor, Task, SubTask, TaskStatus, SubTaskStatus,
    SmartScheduler, estimate_task_tops,
    make_image_blur_input, make_matrix_mult_input, make_hash_benchmark_input,
    make_ml_inference_input, TASK_INPUT_BUILDERS,
)
from .compute_proxy import ComputeProxy, ComputeRequest, ComputeStatus
from .settlement import SettlementEngine, Settlement, SettlementStatus
from .witness import WitnessNetwork, WitnessClient

import config

logger = logging.getLogger(__name__)


class Node:
    """节点协调器（单例）"""

    _instance: Optional["Node"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, listen_port: int = 9000,
                 bootstrap_nodes: list[tuple[str, int]] | None = None):
        if self._initialized:
            return
        self._initialized = True

        self.listen_port = listen_port
        self.bootstrap_nodes = bootstrap_nodes or []

        # 身份
        self.identity_manager = IdentityManager(config.IDENTITY_FILE)
        self.identity: Optional[Identity] = None

        # GPU 监控
        self.gpu_monitor = get_monitor()

        # 账本（地址待身份加载后更新）
        self.ledger: Optional[Ledger] = None

        # P2P 节点（待身份加载后初始化）
        self.p2p: Optional[P2PNode] = None

        # 任务执行器与调度器
        self.task_executor: Optional[TaskExecutor] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.smart_scheduler: Optional[SmartScheduler] = None  # 无感调度器
        self.compute_proxy: Optional[ComputeProxy] = None     # 计算代理（外部软件入口）

        # 结算引擎
        self.settlement_engine: Optional[SettlementEngine] = None

        # 见证网络
        self.witness_network: Optional[WitnessNetwork] = None
        self.witness_client: Optional[WitnessClient] = None

        # 用户设置
        self.settings = {
            "local_gpu_enabled": True,
            "local_utilization_limit": config.GPU_CONFIG["default_utilization_limit"],
            "vram_reserve_mb": config.GPU_CONFIG["default_vram_reserve_mb"],
            "remote_enabled": False,          # 使用远程算力（需求端）
            "sharing_enabled": False,          # 共享算力（贡献端），与 remote_enabled 互斥
            "sharing_utilization_limit": config.GPU_CONFIG["default_utilization_limit"],
            "mode": "balanced",  # speed / balanced / economy
        }

        # 共享状态
        self._sharing_thread: Optional[threading.Thread] = None
        self._sharing_running = False

        # 接收到的远程任务（贡献端）
        self.received_subtasks: dict[str, SubTask] = {}
        # 贡献端任务历史（完成/失败的远程任务）
        self.contributor_history: list[dict] = []
        self._contributor_history_lock = threading.RLock()
        # 贡献端统计
        self.contributor_stats = {
            "tasks_received": 0,       # 接收的任务数
            "tasks_completed": 0,      # 完成的任务数
            "tasks_failed": 0,         # 失败的任务数
            "total_tpt_earned": 0.0,   # 累计收益
            "total_compute_time": 0.0, # 累计计算时长（秒）
            "sharing_started_at": 0.0, # 共享开启时间
        }

        # 统计
        self.stats = {
            "tasks_completed": 0,
            "tpt_earned": 0.0,
            "tpt_consumed": 0.0,
            "uptime_start": time.time(),
        }

    # ---------- 身份 ----------

    @property
    def is_logged_in(self) -> bool:
        return self.identity is not None

    def register(self, email: str, nickname: str = "", password: str = "") -> dict:
        """注册新身份"""
        identity, mnemonic = self.identity_manager.register(email, nickname, password)
        self.identity = identity
        self._post_login_init()
        # 新用户福利
        self._grant_new_user_bonus()
        return {
            "address": identity.address,
            "mnemonic": mnemonic,
            "email": identity.email,
            "nickname": identity.nickname,
        }

    def login(self, email: str, password: str) -> dict:
        """密码登录"""
        identity = self.identity_manager.login_with_password(email, password)
        self.identity = identity
        self._post_login_init()
        return {
            "address": identity.address,
            "email": identity.email,
            "nickname": identity.nickname,
        }

    def recover(self, mnemonic: str, email: str, new_password: str) -> dict:
        """助记词恢复"""
        identity = self.identity_manager.recover_from_mnemonic(mnemonic, email, new_password)
        self.identity = identity
        self._post_login_init()
        return {
            "address": identity.address,
            "email": identity.email,
            "nickname": identity.nickname,
        }

    def _post_login_init(self):
        """登录后初始化各模块"""
        if not self.identity:
            return
        addr = self.identity.address

        # 账本
        self.ledger = Ledger(config.LEDGER_DB, my_address=addr)

        # P2P 节点
        self.p2p = P2PNode(
            listen_host=config.P2P_CONFIG["listen_host"],
            listen_port=self.listen_port,
            signing_kp=self.identity.signing_keypair,
            encryption_kp=self.identity.encryption_keypair,
            my_address=addr,
        )
        # 注册消息处理器
        self.p2p.on("TASK_ASSIGN", self._handle_task_assign)
        self.p2p.on("TASK_RESULT_REQ", self._handle_result_request)
        self.p2p.on("COMPUTE_REQ", self._handle_compute_req)        # 远程计算请求
        self.p2p.on("COMPUTE_RESULT", self._handle_compute_result)  # 远程计算结果
        self.p2p.on("WITNESS_REQ", self._handle_witness_req)
        self.p2p.on("NODE_INFO", self._handle_node_info)
        self.p2p.on("TX_BROADCAST", self._handle_tx_broadcast)
        self.p2p.on("DISCOVER", self._handle_discover)

        # 任务执行器与调度器
        self.task_executor = TaskExecutor(self.gpu_monitor)
        self.task_scheduler = TaskScheduler(self.gpu_monitor, self.task_executor)
        # 智能无感调度器
        self.smart_scheduler = SmartScheduler(
            gpu_monitor=self.gpu_monitor,
            executor=self.task_executor,
            p2p_node=self.p2p,
            my_address=addr,
            local_utilization_limit=self.settings["local_utilization_limit"],
        )
        self.smart_scheduler.start_watchdog()

        # 计算代理（外部软件的算力调度入口）
        self.compute_proxy = ComputeProxy(
            gpu_monitor=self.gpu_monitor,
            smart_scheduler=self.smart_scheduler,
            p2p_node=self.p2p,
            my_address=addr,
            local_backend_url=self.settings.get("local_backend_url", "http://127.0.0.1:11434"),
        )
        self.compute_proxy.start_monitor()

        # 结算引擎
        self.settlement_engine = SettlementEngine(self.ledger, self.identity.signing_keypair)
        self.settlement_engine.start()

        # 见证网络
        self.witness_network = WitnessNetwork(addr, self.identity.signing_keypair)
        self.witness_client = WitnessClient(self.p2p, self.witness_network)

        # 启动 P2P
        self.p2p.start(self.bootstrap_nodes)

    def _grant_new_user_bonus(self):
        """新用户福利"""
        if not self.identity or not self.ledger:
            return
        bonus = config.ECONOMICS["new_user_bonus"]
        bonus_tx = Transaction(
            tx_id="",
            type=TxType.NEW_USER_BONUS.value,
            from_addr="SYSTEM",
            to_addr=self.identity.address,
            amount=float(bonus),
            payload={"reason": "new_user_bonus"},
        )
        # 系统交易无需签名（用系统密钥，这里用本节点签名代替）
        bonus_tx.sign_by(self.identity.signing_keypair)
        self.ledger.add_pending_tx(bonus_tx)
        # 立即出块（_apply_tx_to_balance 会自动给用户加余额）
        self.ledger.produce_block(self.identity.address, self.identity.signing_keypair)
        logger.info(f"已发放新用户福利 {bonus} TPT")

    def logout(self):
        """登出"""
        if self._sharing_running:
            self.stop_sharing()
        if self.compute_proxy:
            self.compute_proxy.stop_monitor()
        if self.smart_scheduler:
            self.smart_scheduler.stop_watchdog()
        if self.p2p:
            self.p2p.stop()
        if self.settlement_engine:
            self.settlement_engine.stop()
        self.identity_manager.logout()
        self.identity = None
        self.ledger = None
        self.p2p = None

    # ---------- GPU 信息 ----------

    def get_gpu_info(self) -> dict:
        """获取本机 GPU 信息"""
        info = self.gpu_monitor.get_info()
        return info.to_dict()

    def get_devices(self) -> list[dict]:
        return [d.to_dict() for d in self.gpu_monitor.list_devices()]

    def run_benchmark(self, duration: int = 5) -> dict:
        """运行基准测试"""
        tops = self.gpu_monitor.benchmark(duration)
        return {"measured_tops": tops, "duration_sec": duration}

    # ---------- 余额与账单 ----------

    def get_balance(self) -> dict:
        if not self.ledger:
            return {"balance": 0, "staked": 0}
        bal, staked = self.ledger.get_my_balance()
        return {"balance": bal, "staked": staked}

    def get_tx_history(self, limit: int = 50) -> list[dict]:
        if not self.ledger or not self.identity:
            return []
        return self.ledger.get_tx_history(self.identity.address, limit)

    # ---------- 充值 ----------

    def deposit(self, amount: float, channel: str = "wechat") -> dict:
        """模拟充值（法币充值 0 手续费）"""
        if not self.identity or not self.ledger:
            raise RuntimeError("未登录")
        # 生成订单 ID
        order_id = crypto.sha256_hex(f"deposit{time.time()}{crypto.random_nonce_hex()}")
        # 充值交易（SYSTEM -> 用户）
        tx = Transaction(
            tx_id="",
            type=TxType.DEPOSIT.value,
            from_addr="SYSTEM",
            to_addr=self.identity.address,
            amount=amount,
            payload={"order_id": order_id, "channel": channel, "fiat_amount": amount},
        )
        tx.sign_by(self.identity.signing_keypair)
        self.ledger.add_pending_tx(tx)
        # 出块后 _apply_tx_to_balance 会自动给用户加余额
        self.ledger.produce_block(self.identity.address, self.identity.signing_keypair)
        logger.info(f"充值成功: {amount} TPT, 渠道 {channel}")
        return {"order_id": order_id, "amount": amount, "status": "confirmed"}

    # ---------- 转账 ----------

    def transfer(self, to_addr: str, amount: float) -> dict:
        """点对点转账（2% 手续费）"""
        if not self.identity or not self.ledger:
            raise RuntimeError("未登录")
        bal, _ = self.ledger.get_my_balance()
        fee = amount * config.ECONOMICS["fee_rate"]
        if bal < amount + fee:
            raise RuntimeError(f"余额不足: 需要 {amount + fee}, 当前 {bal}")
        tx = Transaction(
            tx_id="",
            type=TxType.TRANSFER.value,
            from_addr=self.identity.address,
            to_addr=to_addr,
            amount=amount,
            fee=fee,
            payload={"fee_split": {
                "witness": amount * config.ECONOMICS["witness_pool_rate"],
                "ops": amount * config.ECONOMICS["ops_pool_rate"],
                "revenue": amount * config.ECONOMICS["revenue_pool_rate"],
            }},
        )
        tx.sign_by(self.identity.signing_keypair)
        self.ledger.add_pending_tx(tx)
        self.ledger.produce_block(self.identity.address, self.identity.signing_keypair)
        # 广播
        if self.p2p:
            self.p2p.broadcast("TX_BROADCAST", tx.to_dict())
        logger.info(f"转账成功: {amount} TPT -> {to_addr}, 手续费 {fee}")
        return {"tx_id": tx.tx_id, "amount": amount, "fee": fee}

    # ---------- 质押 ----------

    def stake(self, amount: float) -> dict:
        """质押 TPT"""
        if not self.identity or not self.ledger:
            raise RuntimeError("未登录")
        bal, staked = self.ledger.get_my_balance()
        if bal < amount:
            raise RuntimeError("余额不足")
        tx = Transaction(
            tx_id="",
            type=TxType.STAKE.value,
            from_addr=self.identity.address,
            to_addr="SYSTEM_STAKE",
            amount=amount,
            payload={"stake_to": self.identity.address},
        )
        tx.sign_by(self.identity.signing_keypair)
        self.ledger.add_pending_tx(tx)
        # 出块后 _apply_tx_to_balance 会自动从用户余额扣除 amount
        self.ledger.produce_block(self.identity.address, self.identity.signing_keypair)
        # 仅更新质押列（balance 已由出块扣除，不要重复扣）
        self.ledger.update_staked(self.identity.address, amount)
        # 更新身份质押（用当前会话密码加密，不覆盖原始密码）
        self.identity_manager.update(staked=staked + amount)
        logger.info(f"质押成功: {amount} TPT")
        return {"tx_id": tx.tx_id, "staked": staked + amount}

    def unstake(self, amount: float) -> dict:
        """解除质押（提取质押的 TPT 回余额）"""
        if not self.identity or not self.ledger:
            raise RuntimeError("未登录")
        bal, staked = self.ledger.get_my_balance()
        if staked < amount:
            raise RuntimeError(f"质押不足: 当前 {staked}, 需要 {amount}")
        tx = Transaction(
            tx_id="",
            type=TxType.UNSTAKE.value,
            from_addr="SYSTEM_STAKE",
            to_addr=self.identity.address,
            amount=amount,
            payload={"unstake_to": self.identity.address},
        )
        tx.sign_by(self.identity.signing_keypair)
        self.ledger.add_pending_tx(tx)
        self.ledger.produce_block(self.identity.address, self.identity.signing_keypair)
        self.ledger.update_staked(self.identity.address, -amount)
        self.identity_manager.update(staked=staked - amount)
        logger.info(f"解除质押成功: {amount} TPT")
        return {"tx_id": tx.tx_id, "staked": staked - amount}

    # ---------- 任务发起（需求端，无感调用） ----------

    def submit_task(self, task_type: str, input_data: bytes | None = None,
                    input_spec: dict | None = None,
                    use_local: bool = True, use_remote: bool = True,
                    local_utilization_limit: int | None = None,
                    chunk_count: int = 1) -> dict:
        """提交算力任务（无感调度）

        需求方只需提供任务类型和数据，调度器自动决策：
        - 本地优先执行
        - 本地负载溢出 → 自动拉远程
        - 本地失败 → 自动回退远程
        - 远程节点掉线 → 自动重分发到其他节点
        - 所有远程节点失败 → 本地兜底
        """
        if not self.identity or not self.smart_scheduler:
            raise RuntimeError("未登录")

        if local_utilization_limit is None:
            local_utilization_limit = self.settings["local_utilization_limit"]

        # 构造输入数据
        if input_data is None:
            builder = TASK_INPUT_BUILDERS.get(task_type)
            if builder:
                if task_type == "image_blur":
                    from PIL import Image
                    import io
                    img = Image.new("RGB", (256, 256), color=(123, 200, 50))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    input_data = buf.getvalue()
                else:
                    input_data = builder()
            else:
                input_data = b""

        if input_spec is None:
            input_spec = {"type": task_type, "chunk_count": chunk_count}

        # 同步最新设置到调度器
        self.smart_scheduler.local_utilization_limit = local_utilization_limit
        self.smart_scheduler.set_p2p(self.p2p, self.identity.address)

        # 无感调度执行
        task = self.smart_scheduler.submit(
            task_type=task_type,
            requester=self.identity.address,
            input_data=input_data,
            input_spec=input_spec,
            chunk_count=chunk_count,
            local_utilization_limit=local_utilization_limit,
            force_local=not use_remote,      # use_remote=False 时强制本地
            force_remote=not use_local,      # use_local=False 时强制远程
        )

        self.stats["tasks_completed"] += 1

        # 统计远程执行情况
        remote_count = sum(
            1 for s in task.subtasks
            if SubTask.from_dict(s).assigned_to
            and SubTask.from_dict(s).assigned_to != self.identity.address
        )

        return {
            "task_id": task.task_id,
            "status": task.status,
            "local_executed": remote_count == 0,
            "remote_count": remote_count,
            "actual_tpt": task.actual_tpt,
            "duration_sec": (task.completed_at - task.started_at) if task.completed_at else 0,
            "error": task.error,
            "schedule_mode": self._describe_schedule_mode(task),
        }

    def _describe_schedule_mode(self, task: Task) -> str:
        """描述实际使用的调度模式"""
        subs = [SubTask.from_dict(s) for s in task.subtasks]
        local = sum(1 for s in subs if s.assigned_to == self.identity.address)
        remote = sum(1 for s in subs if s.assigned_to and s.assigned_to != self.identity.address)
        if remote == 0:
            return "local"
        if local == 0:
            return "remote"
        return "mixed"

    def _dispatch_remote_task(self, task: Task) -> dict:
        """分发任务到远程节点"""
        if not self.p2p:
            return {"task_id": task.task_id, "status": "failed", "error": "P2P 未启动"}
        sharing_peers = self.p2p.get_sharing_peers()
        if not sharing_peers:
            # 无远程节点，回退本地
            success = self.task_scheduler.execute_local(task)
            return {
                "task_id": task.task_id,
                "status": "completed" if success else "failed",
                "local_executed": True,
                "fallback": True,
                "error": task.error if not success else "",
            }
        # 分发到第一个共享节点（简化：单节点执行）
        peer = sharing_peers[0]
        for sub_dict in task.subtasks:
            sub = SubTask.from_dict(sub_dict)
            sub.assigned_to = peer.address
            self.p2p.send_to_peer(peer.address, "TASK_ASSIGN", {
                "subtask": sub.to_dict(),
                "requester": self.identity.address,
                "callback_port": self.listen_port,
            })
        task.status = TaskStatus.DISPATCHED.value
        task.started_at = time.time()
        logger.info(f"任务 {task.task_id} 已分发到远程节点 {peer.address}")
        return {
            "task_id": task.task_id,
            "status": "dispatched",
            "assigned_to": peer.address,
            "subtask_count": len(task.subtasks),
        }

    # ---------- 贡献端：接收并执行任务 ----------

    def _handle_task_assign(self, msg, addr):
        """处理远程任务分配（贡献端：自动接取）"""
        if not self.settings["sharing_enabled"] or not self.task_executor:
            # 拒绝任务，通知需求方
            if self.p2p and msg.sender_addr:
                self.p2p.send_to_peer(msg.sender_addr, "TASK_RESULT_REQ", {
                    "subtask": {
                        **msg.payload.get("subtask", {}),
                        "status": SubTaskStatus.FAILED.value,
                        "error": "贡献端未开启共享",
                    },
                    "contributor": self.identity.address if self.identity else "",
                })
            return
        sub_dict = msg.payload.get("subtask")
        if not sub_dict:
            return
        try:
            sub = SubTask.from_dict(sub_dict)
            self.received_subtasks[sub.sub_id] = sub
            # 更新贡献端统计
            with self._contributor_history_lock:
                self.contributor_stats["tasks_received"] += 1
            # 记录节点开始执行任务
            if self.p2p and msg.sender_addr:
                self.p2p.record_task_assigned(self.identity.address)
            # 在后台执行（自动接取，无需手动确认）
            threading.Thread(
                target=self._execute_remote_subtask,
                args=(sub, msg.sender_addr),
                daemon=True,
            ).start()
            logger.info(f"已自动接取远程任务 {sub.sub_id[:16]} (来自 {msg.sender_addr[:12]})")
        except Exception as e:
            logger.error(f"处理任务分配失败: {e}")

    def _execute_remote_subtask(self, sub: SubTask, requester_addr: str):
        """执行远程子任务（贡献端：自动执行并反馈结果）"""
        try:
            # 按贡献者设定的 GPU 使用率上限执行
            sub = self.task_executor.execute(
                sub, self.settings["sharing_utilization_limit"]
            )
            # 返回结果给需求方
            if self.p2p:
                self.p2p.send_to_peer(requester_addr, "TASK_RESULT_REQ", {
                    "subtask": sub.to_dict(),
                    "contributor": self.identity.address,
                })
            # 更新贡献端统计和历史
            with self._contributor_history_lock:
                if sub.status == SubTaskStatus.COMPLETED.value:
                    self.contributor_stats["tasks_completed"] += 1
                    self.contributor_stats["total_compute_time"] += sub.duration_sec
                else:
                    self.contributor_stats["tasks_failed"] += 1
                # 记录历史
                self.contributor_history.append({
                    "sub_id": sub.sub_id,
                    "parent_id": sub.parent_id,
                    "task_type": sub.task_type,
                    "requester": requester_addr,
                    "status": sub.status,
                    "duration_sec": sub.duration_sec,
                    "tops_measured": sub.tops_measured,
                    "result_hash": sub.result_hash[:16] if sub.result_hash else "",
                    "error": sub.error,
                    "timestamp": time.time(),
                })
                # 保留最近 200 条
                if len(self.contributor_history) > 200:
                    self.contributor_history = self.contributor_history[-200:]
            # 触发结算（自动结算收益）
            self._create_remote_settlement(sub, requester_addr)
            # 记录节点任务完成
            if self.p2p:
                self.p2p.record_task_completed(
                    self.identity.address,
                    success=(sub.status == SubTaskStatus.COMPLETED.value)
                )
        except Exception as e:
            logger.error(f"远程子任务执行失败: {e}")
            with self._contributor_history_lock:
                self.contributor_stats["tasks_failed"] += 1
                self.contributor_history.append({
                    "sub_id": sub.sub_id,
                    "parent_id": sub.parent_id,
                    "task_type": sub.task_type,
                    "requester": requester_addr,
                    "status": SubTaskStatus.FAILED.value,
                    "error": str(e),
                    "timestamp": time.time(),
                })

    def _create_remote_settlement(self, sub: SubTask, requester_addr: str):
        """创建远程任务结算（自动结算收益）"""
        if not self.settlement_engine or not self.identity:
            return
        duration_min = max(0.001, sub.duration_sec / 60.0)  # 防止 0
        vram_rate = 1.0  # 简化
        settlement = self.settlement_engine.create_settlement(
            task_id=sub.parent_id,
            requester=requester_addr,
            contributor=self.identity.address,
            task_type=sub.task_type,
            measured_tops=sub.tops_measured,
            vram_usage_rate=vram_rate,
            duration_min=duration_min,
            result_hash=sub.result_hash,
        )
        # 更新累计收益统计
        with self._contributor_history_lock:
            self.contributor_stats["total_tpt_earned"] += settlement.contributor_reward
        logger.info(f"远程结算已创建: {settlement.settlement_id}, "
                    f"收益 {settlement.contributor_reward:.6f} TPT, "
                    f"挑战期至 {time.strftime('%H:%M:%S', time.localtime(settlement.challenge_until))}")

    def _handle_result_request(self, msg, addr):
        """处理远程任务结果返回（转发给 SmartScheduler 结果回收器）"""
        sub_dict = msg.payload.get("subtask")
        if not sub_dict:
            return
        try:
            sub = SubTask.from_dict(sub_dict)
            contributor = msg.payload.get("contributor", "")
            # 记录节点健康状态
            if self.p2p and contributor:
                self.p2p.record_task_completed(contributor, success=(sub.status == SubTaskStatus.COMPLETED.value))
            # 优先交给智能调度器回收
            if self.smart_scheduler:
                self.smart_scheduler.collect_result(sub, contributor)
            # 兼容旧调度器
            elif self.task_scheduler:
                for task in self.task_scheduler.list_tasks():
                    if task.task_id == sub.parent_id:
                        for i, sd in enumerate(task.subtasks):
                            if sd["sub_id"] == sub.sub_id:
                                task.subtasks[i] = sub.to_dict()
                                break
                        all_done = all(
                            s.get("status") == SubTaskStatus.COMPLETED.value
                            for s in task.subtasks
                        )
                        if all_done:
                            results = {s["sub_id"]: base64.b64decode(s["result_data"])
                                       for s in task.subtasks}
                            task.result_data = self.task_scheduler.aggregate_results(task, results)
                            task.status = TaskStatus.COMPLETED.value
                            task.completed_at = time.time()
                            logger.info(f"远程任务 {task.task_id} 全部完成")
                        break
        except Exception as e:
            logger.error(f"处理结果返回失败: {e}")

    # ---------- 远程计算请求处理（贡献端） ----------

    def _handle_compute_req(self, msg, addr):
        """处理远程计算请求（贡献端：收到其他节点的推理/张量计算请求）"""
        if not self.settings.get("sharing_enabled") or not self.compute_proxy:
            # 未开启共享，拒绝
            if self.p2p:
                self.p2p.send_to_peer(msg.sender_addr, "COMPUTE_RESULT", {
                    "request_id": msg.payload.get("request_id"),
                    "result": {"error": "节点未开启算力共享"},
                    "success": False,
                })
            return

        request_type = msg.payload.get("request_type")
        payload = msg.payload.get("payload", {})
        requester = msg.payload.get("requester", "")

        # 在后台执行
        threading.Thread(
            target=self._execute_remote_compute,
            args=(msg.payload.get("request_id", ""), request_type, payload, requester),
            daemon=True,
        ).start()

    def _execute_remote_compute(self, request_id: str, request_type: str,
                                 payload: dict, requester: str):
        """执行远程计算请求并返回结果"""
        try:
            if request_type == "inference":
                result = self.compute_proxy._execute_local_inference(
                    ComputeRequest(
                        request_id=request_id,
                        request_type="inference",
                        source=f"remote:{requester[:12]}",
                        payload=payload,
                    )
                )
            elif request_type == "tensor":
                result = self.compute_proxy._execute_local_tensor(
                    ComputeRequest(
                        request_id=request_id,
                        request_type="tensor",
                        source=f"remote:{requester[:12]}",
                        payload=payload,
                    )
                )
            else:
                result = {"error": f"未知请求类型: {request_type}"}

            if self.p2p:
                self.p2p.send_to_peer(requester, "COMPUTE_RESULT", {
                    "request_id": request_id,
                    "result": result,
                    "success": True,
                })
        except Exception as e:
            logger.error(f"远程计算执行失败: {e}")
            if self.p2p:
                self.p2p.send_to_peer(requester, "COMPUTE_RESULT", {
                    "request_id": request_id,
                    "result": {"error": str(e)},
                    "success": False,
                })

    def _handle_compute_result(self, msg, addr):
        """处理远程计算结果返回"""
        if not self.compute_proxy:
            return
        request_id = msg.payload.get("request_id")
        result = msg.payload.get("result", {})
        self.compute_proxy.receive_remote_result(request_id, result)

    # ---------- 计算代理 API（供外部软件调用） ----------

    def compute_inference(self, model: str, messages: list[dict],
                          source: str = "external", **kwargs) -> dict:
        """推理计算 API（供外部软件调用）"""
        if not self.compute_proxy:
            raise RuntimeError("计算代理未初始化")
        self.compute_proxy.set_smart_scheduler(
            self.smart_scheduler, self.p2p, self.identity.address
        )
        return self.compute_proxy.inference(model, messages, source, **kwargs)

    def compute_tensor(self, operation: str, data: dict,
                       source: str = "external") -> dict:
        """张量计算 API（供外部软件调用）"""
        if not self.compute_proxy:
            raise RuntimeError("计算代理未初始化")
        self.compute_proxy.set_smart_scheduler(
            self.smart_scheduler, self.p2p, self.identity.address
        )
        return self.compute_proxy.tensor_compute(operation, data, source)

    def get_compute_stats(self) -> dict:
        """获取计算代理统计"""
        if not self.compute_proxy:
            return {}
        return self.compute_proxy.get_stats()

    def get_compute_logs(self, limit: int = 50) -> list[dict]:
        """获取调度日志"""
        if not self.compute_proxy:
            return []
        return self.compute_proxy.get_schedule_logs(limit)

    def get_compute_requests(self, limit: int = 20) -> list[dict]:
        """获取最近计算请求"""
        if not self.compute_proxy:
            return []
        return self.compute_proxy.get_recent_requests(limit)

    # ---------- 见证 ----------

    def _handle_witness_req(self, msg, addr):
        """处理见证请求"""
        if not self.witness_network:
            return
        target_id = msg.payload.get("target_id")
        target_type = msg.payload.get("target_type")
        payload = msg.payload.get("payload", {})
        # 验证（简化：通过即批准）
        result = "approved"
        if target_type == "settlement":
            result = self.witness_network.verify_settlement(
                payload, payload.get("result_hash", "")
            )
        att = self.witness_network.attest(target_id, target_type, result)
        if att and self.p2p:
            self.p2p.send_to_peer(msg.sender_addr, "WITNESS_RESP", att.to_dict())

    # ---------- 节点信息同步 ----------

    def _handle_node_info(self, msg, addr):
        """处理节点信息更新"""
        if not self.p2p:
            return
        address = msg.payload.get("address")
        info = msg.payload.get("info", {})
        with self.p2p._peers_lock:
            if address in self.p2p.peers:
                peer = self.p2p.peers[address]
                peer.nickname = info.get("nickname", peer.nickname)
                peer.gpu_name = info.get("gpu_name", peer.gpu_name)
                peer.gpu_tops = info.get("gpu_tops", peer.gpu_tops)
                peer.gpu_available = info.get("gpu_available", peer.gpu_available)
                peer.is_sharing = info.get("is_sharing", peer.is_sharing)
                peer.utilization_limit = info.get("utilization_limit", peer.utilization_limit)

    def _handle_tx_broadcast(self, msg, addr):
        """处理交易广播"""
        tx_dict = msg.payload
        if self.ledger:
            from .ledger import Transaction
            tx = Transaction(
                tx_id=tx_dict["tx_id"], type=tx_dict["type"],
                from_addr=tx_dict["from_addr"], to_addr=tx_dict["to_addr"],
                amount=tx_dict["amount"], fee=tx_dict.get("fee", 0),
                timestamp=tx_dict.get("timestamp", int(time.time())),
                payload=tx_dict.get("payload", {}),
                signature=tx_dict.get("signature", ""),
                signer=tx_dict.get("signer", ""),
            )
            self.ledger.add_pending_tx(tx)

    def _handle_discover(self, msg, addr):
        """处理发现请求，回应已知节点"""
        if self.p2p:
            self.p2p.send_peers_to(msg.sender_addr if hasattr(msg, 'sender_addr') else addr[0], addr[1])

    # ---------- 共享开关（贡献端）与远程开关（需求端）互斥 ----------

    def start_sharing(self) -> dict:
        """开启算力共享（贡献端）

        不需要质押 TPT。
        与"使用远程算力"互斥：开启共享时自动关闭远程。
        """
        if not self.identity:
            raise RuntimeError("未登录")
        # 互斥：关闭远程算力
        if self.settings["remote_enabled"]:
            self.settings["remote_enabled"] = False
            logger.info("开启共享，已自动关闭远程算力")
        self.settings["sharing_enabled"] = True
        self.contributor_stats["sharing_started_at"] = time.time()
        # 广播节点信息
        gpu_info = self.gpu_monitor.get_info()
        if self.p2p:
            self.p2p.update_my_info({
                "nickname": self.identity.nickname,
                "encryption_pub": self.identity.encryption_pub,
                "gpu_name": gpu_info.device_name,
                "gpu_tops": gpu_info.estimated_tops,
                "gpu_available": gpu_info.available,
                "utilization_limit": self.settings["sharing_utilization_limit"],
                "is_sharing": True,
                "is_bootstrap": False,
            })
        logger.info("算力共享已开启（无需质押）")
        return {"status": "sharing", "remote_enabled": self.settings["remote_enabled"]}

    def stop_sharing(self) -> dict:
        """关闭算力共享"""
        self.settings["sharing_enabled"] = False
        if self.p2p:
            self.p2p.update_my_info({
                "is_sharing": False,
            })
        logger.info("算力共享已关闭")
        return {"status": "stopped"}

    def enable_remote(self) -> dict:
        """开启远程算力（需求端）

        与"共享算力"互斥：开启远程时自动关闭共享。
        """
        if not self.identity:
            raise RuntimeError("未登录")
        # 互斥：关闭共享
        if self.settings["sharing_enabled"]:
            self.stop_sharing()
            logger.info("开启远程算力，已自动关闭共享")
        self.settings["remote_enabled"] = True
        logger.info("远程算力已开启")
        return {"status": "remote_enabled", "sharing_enabled": self.settings["sharing_enabled"]}

    def disable_remote(self) -> dict:
        """关闭远程算力"""
        self.settings["remote_enabled"] = False
        logger.info("远程算力已关闭")
        return {"status": "remote_disabled"}

    # ---------- 贡献端数据查询 ----------

    def get_contributor_stats(self) -> dict:
        """获取贡献端统计"""
        with self._contributor_history_lock:
            stats = dict(self.contributor_stats)
            history = list(self.contributor_history)
        # 计算共享时长
        if stats.get("sharing_started_at") and self.settings["sharing_enabled"]:
            stats["sharing_duration"] = time.time() - stats["sharing_started_at"]
        else:
            stats["sharing_duration"] = 0
        # 当前正在执行的任务
        stats["active_tasks"] = sum(
            1 for s in self.received_subtasks.values()
            if s.status == SubTaskStatus.RUNNING.value
        )
        stats["sharing_enabled"] = self.settings["sharing_enabled"]
        stats["sharing_utilization_limit"] = self.settings["sharing_utilization_limit"]
        # 成功率
        total = stats["tasks_received"]
        stats["success_rate"] = (stats["tasks_completed"] / total * 100) if total > 0 else 100.0
        # 待确认收益（挑战期中的结算）
        pending_reward = 0.0
        confirmed_reward = 0.0
        if self.settlement_engine:
            for s in self.settlement_engine.list_settlements(self.identity.address):
                if s.status == "confirmed":
                    confirmed_reward += s.contributor_reward
                elif s.status in ("challenge", "pending"):
                    pending_reward += s.contributor_reward
        stats["pending_reward"] = pending_reward
        stats["confirmed_reward"] = confirmed_reward
        stats["total_tpt_earned"] = confirmed_reward  # 用已确认的收益覆盖
        return stats

    def get_contributor_history(self, limit: int = 50) -> list[dict]:
        """获取贡献端任务历史"""
        with self._contributor_history_lock:
            return list(reversed(self.contributor_history[-limit:]))

    # ---------- 设置 ----------

    def update_settings(self, **kwargs) -> dict:
        """更新设置"""
        for k, v in kwargs.items():
            if k in self.settings:
                self.settings[k] = v
        # 同步到 P2P
        if self.p2p and self.identity:
            gpu_info = self.gpu_monitor.get_info()
            self.p2p.update_my_info({
                "nickname": self.identity.nickname,
                "gpu_name": gpu_info.device_name,
                "gpu_tops": gpu_info.estimated_tops,
                "gpu_available": gpu_info.available,
                "utilization_limit": self.settings["sharing_utilization_limit"],
                "is_sharing": self.settings["sharing_enabled"],
            })
        return self.settings

    # ---------- 节点列表 ----------

    def get_peers(self) -> list[dict]:
        if not self.p2p:
            return []
        return [p.to_dict() for p in self.p2p.get_online_peers()]

    def get_sharing_peers(self) -> list[dict]:
        if not self.p2p:
            return []
        return [p.to_dict() for p in self.p2p.get_sharing_peers()]

    # ---------- C2C 交易 ----------

    def place_order(self, order_type: str, price: float, amount: float) -> dict:
        """挂单"""
        if not self.identity or not self.ledger:
            raise RuntimeError("未登录")
        if order_type == "sell":
            bal, _ = self.ledger.get_my_balance()
            if bal < amount:
                raise RuntimeError("余额不足")
        order_id = crypto.sha256_hex(f"order{time.time()}{crypto.random_nonce_hex()}")
        sig = self.identity.signing_keypair.sign_json({
            "order_id": order_id, "type": order_type, "price": price,
            "amount": amount, "maker": self.identity.address,
        }).hex()
        self.ledger.place_order(order_id, order_type, self.identity.address,
                                price, amount, sig, self.identity.signing_pub)
        # 广播
        if self.p2p:
            self.p2p.broadcast("TX_BROADCAST", {
                "type": "order", "order_id": order_id, "order_type": order_type,
                "price": price, "amount": amount, "maker": self.identity.address,
            })
        return {"order_id": order_id, "status": "open"}

    def get_orders(self) -> list[dict]:
        if not self.ledger:
            return []
        return self.ledger.get_orders("open")

    def cancel_order(self, order_id: str) -> dict:
        if not self.ledger:
            raise RuntimeError("未登录")
        self.ledger.cancel_order(order_id)
        return {"order_id": order_id, "status": "cancelled"}

    # ---------- 统计 ----------

    def get_stats(self) -> dict:
        uptime = time.time() - self.stats["uptime_start"]
        return {
            **self.stats,
            "uptime_sec": uptime,
            "peer_count": len(self.p2p.get_online_peers()) if self.p2p else 0,
            "sharing_peer_count": len(self.p2p.get_sharing_peers()) if self.p2p else 0,
        }

    def get_settlements(self) -> list[dict]:
        if not self.settlement_engine:
            return []
        return [s.to_dict() for s in self.settlement_engine.list_settlements()]

    # ---------- 系统信息 ----------

    def get_system_info(self) -> dict:
        return {
            "version": "1.0.0",
            "p2p_port": self.listen_port,
            "web_port": config.WEB_CONFIG["port"],
            "data_dir": str(config.DATA_DIR),
            "economics": config.ECONOMICS,
            "task_types": config.TASK_TYPES,
        }


# 全局节点实例
_node: Optional[Node] = None


def get_node() -> Node:
    global _node
    if _node is None:
        _node = Node()
    return _node


def init_node(listen_port: int = 9000,
              bootstrap_nodes: list[tuple[str, int]] | None = None) -> Node:
    global _node
    _node = Node(listen_port=listen_port, bootstrap_nodes=bootstrap_nodes)
    return _node
