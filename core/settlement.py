"""结算模块

任务完成后：
1. 核算总 TPT 消耗 = 实测 TOPS × 显存占用率 × 有效时长(分钟)
2. 拆分 2% 手续费（1% 见证池 + 0.5% 运维 + 0.5% 收益）
3. 98% 收益点对点结算给贡献者
4. 进入 1 小时挑战期，期满最终到账
"""
from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional
from enum import Enum

from . import crypto
from .ledger import Transaction, TxType, Ledger
import config

logger = logging.getLogger(__name__)


class SettlementStatus(str, Enum):
    PENDING = "pending"             # 待结算
    CHALLENGE = "challenge"         # 挑战期中
    CONFIRMED = "confirmed"         # 已确认（到账）
    SLASHED = "slashed"             # 被扣除
    DISPUTED = "disputed"           # 争议中
    CANCELLED = "cancelled"         # 取消


@dataclass
class Settlement:
    """结算单"""
    settlement_id: str
    task_id: str
    requester: str                  # 需求方 address
    contributor: str                # 贡献方 address
    redundant_contributor: str = "" # 冗余贡献方 address
    task_type: str = ""
    measured_tops: float = 0.0
    vram_usage_rate: float = 1.0    # 显存占用率
    duration_min: float = 0.0       # 有效时长（分钟）
    total_tpt: float = 0.0          # 总 TPT 消耗
    contributor_reward: float = 0.0 # 贡献者收益 (98%)
    witness_pool: float = 0.0       # 见证池 (1%)
    ops_pool: float = 0.0           # 运维池 (0.5%)
    revenue_pool: float = 0.0       # 收益池 (0.5%)
    status: str = SettlementStatus.PENDING.value
    created_at: float = 0.0
    challenge_until: float = 0.0    # 挑战期截止
    confirmed_at: float = 0.0
    result_hash: str = ""
    redundant_result_hash: str = ""
    slash_amount: float = 0.0
    error: str = ""

    def __post_init__(self):
        if not self.settlement_id:
            self.settlement_id = crypto.sha256_hex(
                f"{self.task_id}{self.contributor}{time.time()}"
            )
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


class SettlementEngine:
    """结算引擎"""

    def __init__(self, ledger: Ledger, signing_kp: Optional[crypto.SigningKeyPair] = None):
        self.ledger = ledger
        self.signing_kp = signing_kp
        self.settlements: dict[str, Settlement] = {}
        self._lock = threading.RLock()
        self._challenge_checker: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """启动挑战期检查器"""
        if self._running:
            return
        self._running = True
        self._challenge_checker = threading.Thread(
            target=self._challenge_loop, daemon=True, name="settlement-challenge"
        )
        self._challenge_checker.start()

    def stop(self):
        self._running = False

    def _challenge_loop(self):
        while self._running:
            try:
                time.sleep(30)
                self._check_challenges()
            except Exception as e:
                logger.error(f"挑战期检查异常: {e}")

    def _check_challenges(self):
        """检查挑战期到期的结算"""
        now = time.time()
        with self._lock:
            to_confirm = [
                s for s in self.settlements.values()
                if s.status == SettlementStatus.CHALLENGE.value
                and s.challenge_until <= now
            ]
        for s in to_confirm:
            self._confirm_settlement(s)

    def calculate_tpt(self, measured_tops: float, vram_usage_rate: float,
                      duration_min: float) -> float:
        """计算 TPT 消耗"""
        # TPT = 实测TOPS × 显存占用率 × 有效时长(分钟)
        return max(0.0, measured_tops * vram_usage_rate * duration_min)

    def create_settlement(self, task_id: str, requester: str, contributor: str,
                          task_type: str, measured_tops: float,
                          vram_usage_rate: float, duration_min: float,
                          result_hash: str = "",
                          redundant_contributor: str = "",
                          redundant_result_hash: str = "") -> Settlement:
        """创建结算单"""
        total_tpt = self.calculate_tpt(measured_tops, vram_usage_rate, duration_min)
        econ = config.ECONOMICS

        # 拆分
        contributor_reward = total_tpt * econ["contributor_rate"]
        witness_pool = total_tpt * econ["witness_pool_rate"]
        ops_pool = total_tpt * econ["ops_pool_rate"]
        revenue_pool = total_tpt * econ["revenue_pool_rate"]

        # 冗余结果一致性校验
        hash_consistent = True
        if redundant_result_hash and result_hash:
            hash_consistent = (result_hash == redundant_result_hash)

        settlement = Settlement(
            settlement_id="",
            task_id=task_id,
            requester=requester,
            contributor=contributor,
            redundant_contributor=redundant_contributor,
            task_type=task_type,
            measured_tops=measured_tops,
            vram_usage_rate=vram_usage_rate,
            duration_min=duration_min,
            total_tpt=total_tpt,
            contributor_reward=contributor_reward,
            witness_pool=witness_pool,
            ops_pool=ops_pool,
            revenue_pool=revenue_pool,
            result_hash=result_hash,
            redundant_result_hash=redundant_result_hash,
            status=SettlementStatus.PENDING.value,
            challenge_until=time.time() + econ["challenge_period"],
        )

        if not hash_consistent:
            # 哈希不一致，进入争议
            settlement.status = SettlementStatus.DISPUTED.value
            settlement.error = "冗余结果哈希不一致"
            logger.warning(f"结算 {settlement.settlement_id} 进入争议：结果哈希不一致")

        with self._lock:
            self.settlements[settlement.settlement_id] = settlement

        # 若哈希一致，进入挑战期
        if hash_consistent:
            settlement.status = SettlementStatus.CHALLENGE.value
            self._freeze_and_settle(settlement)

        return settlement

    def _freeze_and_settle(self, settlement: Settlement):
        """冻结额度并生成交易"""
        if not self.signing_kp:
            logger.warning("无私钥，跳过链上结算")
            return

        # 1. 需求方消费交易（SYSTEM -> ops/revenue/witness 池）
        consume_tx = Transaction(
            tx_id="",
            type=TxType.CONSUME.value,
            from_addr=settlement.requester,
            to_addr="SYSTEM_TPT_POOL",
            amount=settlement.total_tpt - settlement.contributor_reward,
            fee=0.0,
            payload={
                "task_id": settlement.task_id,
                "settlement_id": settlement.settlement_id,
                "witness_pool": settlement.witness_pool,
                "ops_pool": settlement.ops_pool,
                "revenue_pool": settlement.revenue_pool,
            },
        )
        consume_tx.sign_by(self.signing_kp)
        self.ledger.add_pending_tx(consume_tx)

        # 2. 贡献者收益交易（SYSTEM -> 贡献者），挑战期满确认
        reward_tx = Transaction(
            tx_id="",
            type=TxType.REWARD.value,
            from_addr="SYSTEM",
            to_addr=settlement.contributor,
            amount=settlement.contributor_reward,
            fee=0.0,
            payload={
                "task_id": settlement.task_id,
                "settlement_id": settlement.settlement_id,
                "challenge_until": settlement.challenge_until,
            },
        )
        reward_tx.sign_by(self.signing_kp)
        self.ledger.add_pending_tx(reward_tx)

        logger.info(f"结算单已创建: {settlement.settlement_id}, "
                    f"总消耗 {settlement.total_tpt:.4f} TPT, "
                    f"贡献者收益 {settlement.contributor_reward:.4f} TPT")

    def _confirm_settlement(self, settlement: Settlement):
        """挑战期满确认结算"""
        settlement.status = SettlementStatus.CONFIRMED.value
        settlement.confirmed_at = time.time()
        # 立即出块确认交易
        if self.signing_kp:
            try:
                block = self.ledger.produce_block(
                    settlement.requester, self.signing_kp
                )
                if block:
                    logger.info(f"结算 {settlement.settlement_id} 已确认，"
                                f"区块 #{block.index}")
            except Exception as e:
                logger.error(f"出块失败: {e}")
        logger.info(f"结算 {settlement.settlement_id} 挑战期满，已确认到账")

    def dispute_settlement(self, settlement_id: str, witness_addr: str,
                           reason: str) -> bool:
        """见证节点发起争议"""
        with self._lock:
            s = self.settlements.get(settlement_id)
            if not s or s.status != SettlementStatus.CHALLENGE.value:
                return False
            s.status = SettlementStatus.DISPUTED.value
            s.error = f"见证节点 {witness_addr} 争议: {reason}"
            # Slashing 贡献者
            s.slash_amount = config.ECONOMICS["stake_min"] * config.ECONOMICS["stake_slash_rate"]
            logger.warning(f"结算 {settlement_id} 被见证节点争议，扣除质押 {s.slash_amount}")
            # 生成 slashing 交易
            if self.signing_kp:
                slash_tx = Transaction(
                    tx_id="",
                    type=TxType.SLASH.value,
                    from_addr=s.contributor,
                    to_addr="SYSTEM_SLASH_POOL",
                    amount=s.slash_amount,
                    payload={"settlement_id": settlement_id, "reason": reason,
                             "witness": witness_addr},
                )
                slash_tx.sign_by(self.signing_kp)
                self.ledger.add_pending_tx(slash_tx)
            return True

    def get_settlement(self, settlement_id: str) -> Optional[Settlement]:
        with self._lock:
            return self.settlements.get(settlement_id)

    def list_settlements(self, address: str | None = None) -> list[Settlement]:
        with self._lock:
            if address:
                return [s for s in self.settlements.values()
                        if s.requester == address or s.contributor == address]
            return list(self.settlements.values())
