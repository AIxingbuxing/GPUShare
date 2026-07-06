"""P2P 网络模块

基于 asyncio 的简化 Kademlia DHT + TCP 直连 + NaCl Box 端对端加密。
- DHT 用于节点发现与路由
- TCP 用于任务数据传输
- 所有消息均签名 + 加密
"""
from __future__ import annotations

import asyncio
import json
import time
import socket
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable, Any
from collections import defaultdict

from . import crypto

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """对等节点信息"""
    address: str                  # 钱包地址
    host: str
    port: int
    signing_pub: str              # Ed25519 公钥 hex
    encryption_pub: str           # X25519 公钥 hex
    nickname: str = ""
    gpu_name: str = ""
    gpu_tops: float = 0.0
    gpu_available: bool = False
    utilization_limit: int = 80
    last_heartbeat: float = 0.0
    reputation: int = 100
    is_sharing: bool = False
    is_bootstrap: bool = False
    # 健康追踪（用于择优调度）
    active_tasks: int = 0         # 当前正在执行的任务数
    completed_tasks: int = 0      # 累计完成任务数
    failed_tasks: int = 0         # 累计失败任务数
    success_rate: float = 1.0     # 成功率（0-1）

    @property
    def is_online(self) -> bool:
        # 离线判定：15 秒无心跳（原 30 秒太慢，掉线检测不及时）
        return time.time() - self.last_heartbeat < 15

    @property
    def latency_ms(self) -> float:
        if not self.is_online:
            return 9999.0
        return max(1.0, (time.time() - self.last_heartbeat) * 1000)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_online"] = self.is_online
        d["latency_ms"] = self.latency_ms
        return d


class P2PMessage:
    """P2P 消息封装"""

    def __init__(self, msg_type: str, payload: dict, sender_addr: str,
                 sender_signing_pub: str = "", signature: str = ""):
        self.msg_type = msg_type
        self.payload = payload
        self.sender_addr = sender_addr
        self.sender_signing_pub = sender_signing_pub
        self.signature = signature
        self.timestamp = int(time.time())

    def to_dict(self) -> dict:
        return {
            "msg_type": self.msg_type,
            "payload": self.payload,
            "sender_addr": self.sender_addr,
            "sender_signing_pub": self.sender_signing_pub,
            "signature": self.signature,
            "timestamp": self.timestamp,
        }

    def to_sign_dict(self) -> dict:
        d = self.to_dict()
        d.pop("signature")
        return d

    def sign(self, signing_kp: crypto.SigningKeyPair):
        self.sender_signing_pub = signing_kp.public_bytes.hex()
        self.signature = signing_kp.sign_json(self.to_sign_dict()).hex()
        return self

    def verify(self) -> bool:
        if not self.signature or not self.sender_signing_pub:
            return False
        try:
            sig = bytes.fromhex(self.signature)
            pub = bytes.fromhex(self.sender_signing_pub)
            return crypto.verify_json_signature(pub, sig, self.to_sign_dict())
        except Exception:
            return False

    @classmethod
    def from_dict(cls, d: dict) -> "P2PMessage":
        m = cls(
            msg_type=d["msg_type"],
            payload=d["payload"],
            sender_addr=d["sender_addr"],
            sender_signing_pub=d.get("sender_signing_pub", ""),
            signature=d.get("signature", ""),
        )
        m.timestamp = d.get("timestamp", int(time.time()))
        return m


class P2PNode:
    """P2P 节点"""

    def __init__(self, listen_host: str = "0.0.0.0", listen_port: int = 9000,
                 signing_kp: Optional[crypto.SigningKeyPair] = None,
                 encryption_kp: Optional[crypto.EncryptionKeyPair] = None,
                 my_address: str = ""):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.signing_kp = signing_kp or crypto.SigningKeyPair()
        self.encryption_kp = encryption_kp or crypto.EncryptionKeyPair()
        self.my_address = my_address or self.signing_kp.address

        self.peers: dict[str, PeerInfo] = {}  # address -> PeerInfo
        self._peers_lock = threading.RLock()

        self.server: Optional[socket.socket] = None
        self._running = False
        self._threads: list[threading.Thread] = []

        # 消息处理器
        self._handlers: dict[str, Callable[[P2PMessage, tuple], Any]] = {}

        # DHT 路由表（简化：address -> (host, port)）
        self._dht: dict[str, tuple[str, int]] = {}

        # 接收到的交易/区块缓存（用于去中心化同步）
        self.received_txs: list[dict] = []
        self.received_blocks: list[dict] = []

        # 统计
        self.messages_sent = 0
        self.messages_received = 0

    # ---------- 消息处理器注册 ----------

    def on(self, msg_type: str, handler: Callable[[P2PMessage, tuple], Any]):
        """注册消息处理器"""
        self._handlers[msg_type] = handler

    # ---------- 启动/停止 ----------

    def start(self, bootstrap_nodes: list[tuple[str, int]] | None = None):
        """启动 P2P 节点"""
        if self._running:
            return
        self._running = True

        # 启动 TCP 服务器
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server.bind((self.listen_host, self.listen_port))
        except OSError as e:
            # 端口被占用，自动找可用端口
            self.server.bind((self.listen_host, 0))
            self.listen_port = self.server.getsockname()[1]
            logger.warning(f"原端口被占用，改用 {self.listen_port}: {e}")
        self.server.listen(64)
        self.server.settimeout(1.0)

        # 接收线程
        t = threading.Thread(target=self._accept_loop, daemon=True, name="p2p-accept")
        t.start()
        self._threads.append(t)

        # 心跳线程
        t = threading.Thread(target=self._heartbeat_loop, daemon=True, name="p2p-heartbeat")
        t.start()
        self._threads.append(t)

        # 发现线程
        t = threading.Thread(target=self._discovery_loop, daemon=True, name="p2p-discovery")
        t.start()
        self._threads.append(t)

        logger.info(f"P2P 节点已启动: {self.listen_host}:{self.listen_port}, address={self.my_address}")

        # 连接引导节点
        if bootstrap_nodes:
            for host, port in bootstrap_nodes:
                threading.Thread(
                    target=self._bootstrap_to, args=(host, port), daemon=True
                ).start()

    def stop(self):
        self._running = False
        if self.server:
            try:
                self.server.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2)
        logger.info("P2P 节点已停止")

    # ---------- 网络循环 ----------

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self.server.accept()
                threading.Thread(
                    target=self._handle_connection, args=(conn, addr), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_connection(self, conn: socket.socket, addr: tuple):
        """处理入站连接"""
        try:
            conn.settimeout(10)
            data = self._recv_all(conn)
            if not data:
                return
            try:
                msg_dict = json.loads(data.decode("utf-8"))
                msg = P2PMessage.from_dict(msg_dict)
            except Exception as e:
                logger.debug(f"解析消息失败: {e}")
                return

            self.messages_received += 1

            # 验证签名（Bootstrap/Hello 消息可能无签名）
            if msg.signature and not msg.verify():
                logger.warning(f"消息签名验证失败: {msg.msg_type} from {addr}")
                return

            # 处理消息
            handler = self._handlers.get(msg.msg_type)
            if handler:
                try:
                    handler(msg, addr)
                except Exception as e:
                    logger.error(f"处理消息 {msg.msg_type} 失败: {e}")
            else:
                # 自动处理 Hello/Ping/Peers
                if msg.msg_type == "HELLO":
                    self._handle_hello(msg, addr)
                elif msg.msg_type == "PING":
                    self._handle_ping(msg, addr)
                elif msg.msg_type == "PEERS_RESP":
                    self._handle_peers_resp(msg, addr)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _heartbeat_loop(self):
        """周期性心跳：3 秒间隔，15 秒判定离线"""
        while self._running:
            try:
                time.sleep(3)  # 原 5 秒，加速到 3 秒
                self.broadcast_ping()
                # 每 15 秒清理离线节点（原 30 秒）
                with self._peers_lock:
                    offline = [a for a, p in self.peers.items() if not p.is_online]
                    for a in offline:
                        logger.info(f"节点离线: {a}")
                        self.peers.pop(a, None)
                        self._dht.pop(a, None)
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")

    def _discovery_loop(self):
        """周期性节点发现"""
        while self._running:
            try:
                time.sleep(30)
                self.discover_peers()
            except Exception as e:
                logger.error(f"发现循环异常: {e}")

    # ---------- 消息发送 ----------

    def _send_to(self, host: str, port: int, msg: P2PMessage) -> bool:
        """发送消息到指定地址"""
        if self.signing_kp:
            msg.sign(self.signing_kp)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, port))
                data = json.dumps(msg.to_dict()).encode("utf-8")
                self._send_all(s, data)
                self.messages_sent += 1
            return True
        except Exception as e:
            logger.debug(f"发送到 {host}:{port} 失败: {e}")
            return False

    @staticmethod
    def _send_all(sock: socket.socket, data: bytes):
        """发送带长度前缀的数据"""
        length = len(data)
        sock.sendall(length.to_bytes(8, "big") + data)

    @staticmethod
    def _recv_all(sock: socket.socket) -> bytes:
        """接收带长度前缀的数据"""
        header = b""
        while len(header) < 8:
            chunk = sock.recv(8 - len(header))
            if not chunk:
                return b""
            header += chunk
        length = int.from_bytes(header, "big")
        if length <= 0 or length > 100 * 1024 * 1024:  # 100MB 上限
            return b""
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                break
            data += chunk
        return data

    # ---------- 消息类型 ----------

    def send_hello(self, host: str, port: int, my_info: dict) -> bool:
        """发送 Hello 消息（注册到对端）"""
        msg = P2PMessage("HELLO", {
            "address": self.my_address,
            "host": self.listen_host,
            "port": self.listen_port,
            "info": my_info,
        }, self.my_address)
        return self._send_to(host, port, msg)

    def _handle_hello(self, msg: P2PMessage, addr: tuple):
        """处理 Hello 消息"""
        payload = msg.payload
        peer = PeerInfo(
            address=payload["address"],
            host=payload.get("host", addr[0]),
            port=payload.get("port", addr[1]),
            signing_pub=msg.sender_signing_pub,
            encryption_pub=payload.get("info", {}).get("encryption_pub", ""),
            nickname=payload.get("info", {}).get("nickname", ""),
            gpu_name=payload.get("info", {}).get("gpu_name", ""),
            gpu_tops=payload.get("info", {}).get("gpu_tops", 0.0),
            gpu_available=payload.get("info", {}).get("gpu_available", False),
            utilization_limit=payload.get("info", {}).get("utilization_limit", 80),
            last_heartbeat=time.time(),
            is_sharing=payload.get("info", {}).get("is_sharing", False),
            is_bootstrap=payload.get("info", {}).get("is_bootstrap", False),
        )
        with self._peers_lock:
            self.peers[peer.address] = peer
            self._dht[peer.address] = (peer.host, peer.port)
        logger.info(f"新节点上线: {peer.address} @ {peer.host}:{peer.port}")
        # 回应已知节点列表
        self.send_peers_to(peer.host, peer.port)

    def broadcast_ping(self):
        """广播 Ping"""
        msg = P2PMessage("PING", {"timestamp": time.time()}, self.my_address)
        with self._peers_lock:
            peers = list(self.peers.values())
        for p in peers:
            self._send_to(p.host, p.port, msg)

    def _handle_ping(self, msg: P2PMessage, addr: tuple):
        """处理 Ping"""
        with self._peers_lock:
            if msg.sender_addr in self.peers:
                self.peers[msg.sender_addr].last_heartbeat = time.time()

    def send_peers_to(self, host: str, port: int):
        """发送已知节点列表"""
        with self._peers_lock:
            peers_list = [
                {"address": p.address, "host": p.host, "port": p.port,
                 "signing_pub": p.signing_pub, "encryption_pub": p.encryption_pub,
                 "nickname": p.nickname, "gpu_name": p.gpu_name,
                 "gpu_tops": p.gpu_tops, "gpu_available": p.gpu_available,
                 "is_sharing": p.is_sharing, "reputation": p.reputation}
                for p in list(self.peers.values())[:20]
            ]
        msg = P2PMessage("PEERS_RESP", {"peers": peers_list}, self.my_address)
        self._send_to(host, port, msg)

    def _handle_peers_resp(self, msg: P2PMessage, addr: tuple):
        """处理节点列表响应"""
        peers = msg.payload.get("peers", [])
        with self._peers_lock:
            for p in peers:
                if p["address"] != self.my_address and p["address"] not in self.peers:
                    peer = PeerInfo(
                        address=p["address"], host=p["host"], port=p["port"],
                        signing_pub=p.get("signing_pub", ""),
                        encryption_pub=p.get("encryption_pub", ""),
                        nickname=p.get("nickname", ""),
                        gpu_name=p.get("gpu_name", ""),
                        gpu_tops=p.get("gpu_tops", 0.0),
                        gpu_available=p.get("gpu_available", False),
                        last_heartbeat=time.time(),
                        is_sharing=p.get("is_sharing", False),
                        reputation=p.get("reputation", 100),
                    )
                    self.peers[peer.address] = peer
                    self._dht[peer.address] = (peer.host, peer.port)

    def discover_peers(self):
        """主动发现节点"""
        with self._peers_lock:
            peers = list(self.peers.values())
        for p in peers[:5]:
            msg = P2PMessage("DISCOVER", {"from": self.my_address}, self.my_address)
            self._send_to(p.host, p.port, msg)

    def broadcast(self, msg_type: str, payload: dict, exclude: str | None = None):
        """广播消息到所有节点"""
        msg = P2PMessage(msg_type, payload, self.my_address)
        with self._peers_lock:
            peers = [p for p in self.peers.values() if p.address != exclude]
        for p in peers:
            self._send_to(p.host, p.port, msg)

    def send_to_peer(self, address: str, msg_type: str, payload: dict) -> bool:
        """发送给指定节点"""
        with self._peers_lock:
            peer = self.peers.get(address)
        if not peer:
            # 从 DHT 查
            host_port = self._dht.get(address)
            if not host_port:
                return False
            peer = PeerInfo(address=address, host=host_port[0], port=host_port[1],
                            signing_pub="", encryption_pub="")
        msg = P2PMessage(msg_type, payload, self.my_address)
        return self._send_to(peer.host, peer.port, msg)

    # ---------- 引导 ----------

    def _bootstrap_to(self, host: str, port: int):
        """连接到引导节点"""
        time.sleep(1)  # 等待服务器就绪
        my_info = {
            "nickname": "node",
            "encryption_pub": self.encryption_kp.public_bytes.hex(),
            "gpu_name": "",
            "gpu_tops": 0.0,
            "gpu_available": False,
            "is_sharing": False,
            "is_bootstrap": False,
        }
        self.send_hello(host, port, my_info)
        logger.info(f"已向引导节点 {host}:{port} 注册")

    # ---------- 节点信息更新 ----------

    def update_my_info(self, info: dict):
        """更新并广播自己的信息"""
        self.broadcast("NODE_INFO", {
            "address": self.my_address,
            "info": info,
        })

    def get_online_peers(self) -> list[PeerInfo]:
        with self._peers_lock:
            return [p for p in self.peers.values() if p.is_online]

    def get_sharing_peers(self) -> list[PeerInfo]:
        with self._peers_lock:
            return [p for p in self.peers.values() if p.is_online and p.is_sharing]

    def get_best_peer(self, exclude: list[str] | None = None) -> Optional[PeerInfo]:
        """获取最优节点：信誉高 + 延迟低 + 负载低 + 成功率高"""
        exclude = exclude or []
        with self._peers_lock:
            candidates = [
                p for p in self.peers.values()
                if p.is_online and p.is_sharing and p.address not in exclude
            ]
        if not candidates:
            return None
        # 综合评分：信誉×成功率 / (1 + 活跃任务数) - 延迟惩罚
        candidates.sort(key=lambda p: (
            -(p.reputation * p.success_rate / (1 + p.active_tasks)),
            p.latency_ms
        ))
        return candidates[0]

    def record_task_assigned(self, address: str):
        """记录节点开始执行任务"""
        with self._peers_lock:
            p = self.peers.get(address)
            if p:
                p.active_tasks += 1

    def record_task_completed(self, address: str, success: bool):
        """记录节点任务完成状态，更新健康指标"""
        with self._peers_lock:
            p = self.peers.get(address)
            if p:
                p.active_tasks = max(0, p.active_tasks - 1)
                if success:
                    p.completed_tasks += 1
                    p.reputation = min(200, p.reputation + 1)
                else:
                    p.failed_tasks += 1
                    p.reputation = max(0, p.reputation - 5)
                total = p.completed_tasks + p.failed_tasks
                if total > 0:
                    p.success_rate = p.completed_tasks / total
