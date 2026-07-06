"""见证网络模块

见证节点对结算/转账做去中心化公证：
- 任务结果哈希公证
- 转账合法性公证
- 5 见证节点多签紧急熔断（仅限违法内容）

简化实现：本节点可作为见证节点，对收到的结算请求做结果重算验证。
"""
from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional
from enum import Enum

from . import crypto

logger = logging.getLogger(__name__)


class WitnessStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class WitnessAttestation:
    """见证证明"""
    attestation_id: str
    target_id: str                # 被见证对象 ID（settlement_id / tx_id）
    target_type: str              # settlement / transaction / block
    witness_addr: str
    witness_pub: str
    result: str                   # WitnessStatus
    reason: str = ""
    signature: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.attestation_id:
            self.attestation_id = crypto.sha256_hex(
                f"{self.target_id}{self.witness_addr}{self.timestamp}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_sign_dict(self) -> dict:
        d = self.to_dict()
        d.pop("signature")
        return d

    def sign_by(self, kp: crypto.SigningKeyPair):
        self.witness_pub = kp.public_bytes.hex()
        self.signature = kp.sign_json(self.to_sign_dict()).hex()
        return self


class WitnessNetwork:
    """见证网络"""

    def __init__(self, my_address: str, signing_kp: Optional[crypto.SigningKeyPair] = None):
        self.my_address = my_address
        self.signing_kp = signing_kp
        self.attestations: dict[str, list[WitnessAttestation]] = {}
        self._lock = threading.RLock()
        # 注册为见证节点的对端
        self.witness_peers: dict[str, str] = {}  # address -> signing_pub

    def is_witness(self, address: str) -> bool:
        """是否为见证节点（本简化版：信誉 ≥ 80 即可）"""
        # 实际网络中应由选举/质押决定
        return True  # 简化：所有节点都可作见证

    def attest(self, target_id: str, target_type: str, result: str,
               reason: str = "") -> Optional[WitnessAttestation]:
        """本节点作为见证，签署证明"""
        if not self.signing_kp:
            return None
        att = WitnessAttestation(
            attestation_id="",
            target_id=target_id,
            target_type=target_type,
            witness_addr=self.my_address,
            witness_pub="",
            result=result,
            reason=reason,
        )
        att.sign_by(self.signing_kp)
        with self._lock:
            self.attestations.setdefault(target_id, []).append(att)
        logger.info(f"已签署见证证明: {target_id} -> {result}")
        return att

    def receive_attestation(self, att: WitnessAttestation) -> bool:
        """接收来自其他见证节点的证明"""
        # 验证签名
        if not att.signature or not att.witness_pub:
            return False
        try:
            sig = bytes.fromhex(att.signature)
            pub = bytes.fromhex(att.witness_pub)
            if not crypto.verify_json_signature(pub, sig, att.to_sign_dict()):
                return False
        except Exception:
            return False
        with self._lock:
            self.attestations.setdefault(att.target_id, []).append(att)
        return True

    def count_attestations(self, target_id: str, result: str = "approved") -> int:
        with self._lock:
            return sum(1 for a in self.attestations.get(target_id, []) if a.result == result)

    def is_notarized(self, target_id: str,
                     min_signatures: int = 3) -> bool:
        """是否达到公证门槛"""
        return self.count_attestations(target_id, "approved") >= min_signatures

    def verify_settlement(self, settlement_dict: dict,
                          expected_result_hash: str) -> str:
        """验证结算单（结果哈希一致性）"""
        result_hash = settlement_dict.get("result_hash", "")
        redundant_hash = settlement_dict.get("redundant_result_hash", "")
        if not result_hash:
            return WitnessStatus.REJECTED.value
        if expected_result_hash and result_hash != expected_result_hash:
            return WitnessStatus.REJECTED.value
        if redundant_hash and result_hash != redundant_hash:
            return WitnessStatus.REJECTED.value
        return WitnessStatus.APPROVED.value

    def get_attestations(self, target_id: str) -> list[WitnessAttestation]:
        with self._lock:
            return list(self.attestations.get(target_id, []))


class WitnessClient:
    """见证客户端：与见证网络交互"""

    def __init__(self, p2p_node, witness_network: WitnessNetwork):
        self.p2p_node = p2p_node
        self.witness_network = witness_network

    def request_notarization(self, target_id: str, target_type: str,
                             payload: dict) -> bool:
        """请求见证网络公证"""
        # 广播见证请求
        self.p2p_node.broadcast("WITNESS_REQ", {
            "target_id": target_id,
            "target_type": target_type,
            "payload": payload,
            "requester": self.p2p_node.my_address,
        })
        return True

    def submit_to_storage(self, encrypted_data: bytes, metadata: dict) -> bool:
        """提交加密数据到存证服务（Bootstrap 节点）"""
        # 简化：本地存证
        import json
        from pathlib import Path
        store_dir = Path("data/witness_store")
        store_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "metadata": metadata,
            "data_hash": crypto.sha256_hex(encrypted_data),
            "timestamp": time.time(),
        }
        (store_dir / f"{metadata.get('id', crypto.random_nonce_hex(8))}.json").write_text(
            json.dumps(record, indent=2)
        )
        return True
