"""去中心化账本模块

本地维护区块链结构账本，记录 TPT 资产的充值、消费、收益、转账、质押、C2C 交易。
所有交易需 Ed25519 签名，区块需见证网络多签公证。
"""
from __future__ import annotations

import json
import time
import sqlite3
import threading
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional
from enum import Enum

from . import crypto

logger = logging.getLogger(__name__)


class TxType(str, Enum):
    DEPOSIT = "deposit"           # 充值
    CONSUME = "consume"           # 算力消费
    REWARD = "reward"             # 算力收益
    TRANSFER = "transfer"         # 转账
    STAKE = "stake"               # 质押
    UNSTAKE = "unstake"           # 解除质押
    SLASH = "slash"               # 扣除质押
    FEE = "fee"                   # 手续费
    TRADE = "trade"               # C2C 交易
    NEW_USER_BONUS = "new_user_bonus"  # 新用户福利


@dataclass
class Transaction:
    """交易"""
    tx_id: str                    # 交易 ID（哈希）
    type: str                     # TxType
    from_addr: str                # 发送方地址（系统交易为 "SYSTEM"）
    to_addr: str                  # 接收方地址
    amount: float                 # TPT 数量
    fee: float = 0.0              # 手续费
    timestamp: int = 0
    payload: dict = field(default_factory=dict)  # 附加数据（任务 ID 等）
    signature: str = ""           # 发送方签名 hex
    signer: str = ""              # 发送方公钥 hex
    witness_sigs: list = field(default_factory=list)  # 见证签名列表

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time())
        if not self.tx_id:
            self.tx_id = self._compute_id()

    def _compute_id(self) -> str:
        data = {
            "type": self.type,
            "from_addr": self.from_addr,
            "to_addr": self.to_addr,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
        return crypto.sha256_hex(crypto.canonical_json(data))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_sign_dict(self) -> dict:
        """待签名内容（不含 signature/witness_sigs）"""
        d = self.to_dict()
        d.pop("signature", None)
        d.pop("signer", None)
        d.pop("witness_sigs", None)
        return d

    def sign_by(self, signing_kp: crypto.SigningKeyPair):
        """签名"""
        sig = signing_kp.sign_json(self.to_sign_dict())
        self.signature = sig.hex()
        self.signer = signing_kp.public_bytes.hex()
        # 重新计算 tx_id（含签名后内容保持稳定，实际中 tx_id 应基于无签名内容）
        return self

    def verify_signature(self) -> bool:
        """验证发送方签名"""
        if not self.signature or not self.signer:
            return False
        try:
            sig = bytes.fromhex(self.signature)
            pub = bytes.fromhex(self.signer)
            return crypto.verify_json_signature(pub, sig, self.to_sign_dict())
        except Exception:
            return False


@dataclass
class Block:
    """区块"""
    index: int
    prev_hash: str
    transactions: list[dict]
    timestamp: int
    merkle_root: str
    producer: str                  # 出块者地址
    producer_sig: str = ""
    witness_sigs: list = field(default_factory=list)  # 见证签名
    hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time())
        if not self.merkle_root:
            self.merkle_root = crypto.merkle_root([tx["tx_id"] for tx in self.transactions])
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "producer": self.producer,
            "tx_count": len(self.transactions),
        }
        return crypto.sha256_hex(crypto.canonical_json(data))

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    """去中心化账本"""

    def __init__(self, db_path, my_address: str = "SYSTEM"):
        self.db_path = str(db_path)
        self.my_address = my_address
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS blocks (
                        block_index INTEGER PRIMARY KEY,
                        hash TEXT NOT NULL,
                        prev_hash TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        producer TEXT NOT NULL,
                        merkle_root TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        tx_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        from_addr TEXT NOT NULL,
                        to_addr TEXT NOT NULL,
                        amount REAL NOT NULL,
                        fee REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        signature TEXT NOT NULL,
                        signer TEXT NOT NULL,
                        block_index INTEGER,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS balances (
                        address TEXT PRIMARY KEY,
                        balance REAL NOT NULL DEFAULT 0,
                        staked REAL NOT NULL DEFAULT 0,
                        updated_at INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,           -- buy / sell
                        maker TEXT NOT NULL,
                        price REAL NOT NULL,          -- TPT 单价（法币）
                        amount REAL NOT NULL,         -- 数量
                        filled REAL NOT NULL DEFAULT 0,
                        status TEXT DEFAULT 'open',   -- open / filled / cancelled
                        timestamp INTEGER NOT NULL,
                        signature TEXT NOT NULL,
                        signer TEXT NOT NULL
                    )
                """)
                conn.commit()
                # 创世块
                count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
                if count == 0:
                    genesis = Block(
                        index=0,
                        prev_hash="0" * 64,
                        transactions=[],
                        timestamp=int(time.time()),
                        merkle_root=crypto.sha256_hex(b"genesis"),
                        producer="GENESIS",
                    )
                    conn.execute(
                        "INSERT INTO blocks VALUES (?,?,?,?,?,?,?)",
                        (genesis.index, genesis.hash, genesis.prev_hash,
                         genesis.timestamp, genesis.producer, genesis.merkle_root,
                         json.dumps(genesis.to_dict()))
                    )
                    conn.commit()
                    logger.info("创世块已创建")
            finally:
                conn.close()

    # ---------- 区块操作 ----------

    def get_latest_block(self) -> Optional[Block]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT data FROM blocks ORDER BY block_index DESC LIMIT 1"
                ).fetchone()
                if row:
                    return Block(**json.loads(row[0]))
                return None
            finally:
                conn.close()

    def get_block(self, index: int) -> Optional[Block]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT data FROM blocks WHERE block_index = ?", (index,)
                ).fetchone()
                if row:
                    return Block(**json.loads(row[0]))
                return None
            finally:
                conn.close()

    def get_block_count(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                return conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            finally:
                conn.close()

    def append_block(self, block: Block) -> bool:
        """追加区块"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                latest = self.get_latest_block()
                if latest and block.index != latest.index + 1:
                    logger.error(f"区块高度不连续: 期望 {latest.index + 1}, 实际 {block.index}")
                    return False
                if latest and block.prev_hash != latest.hash:
                    logger.error("prev_hash 不匹配")
                    return False
                conn.execute(
                    "INSERT INTO blocks VALUES (?,?,?,?,?,?,?)",
                    (block.index, block.hash, block.prev_hash,
                     block.timestamp, block.producer, block.merkle_root,
                     json.dumps(block.to_dict()))
                )
                # 更新交易状态
                for tx in block.transactions:
                    conn.execute(
                        "UPDATE transactions SET block_index = ?, status = 'confirmed' WHERE tx_id = ?",
                        (block.index, tx["tx_id"])
                    )
                # 更新余额
                for tx in block.transactions:
                    self._apply_tx_to_balance(conn, tx)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"追加区块失败: {e}")
                return False
            finally:
                conn.close()

    def _apply_tx_to_balance(self, conn, tx: dict):
        """应用交易到余额（仅确认后调用）"""
        from_addr = tx["from_addr"]
        to_addr = tx["to_addr"]
        amount = tx["amount"]
        fee = tx.get("fee", 0)

        # 扣发送方
        if from_addr != "SYSTEM":
            row = conn.execute(
                "SELECT balance, staked FROM balances WHERE address = ?", (from_addr,)
            ).fetchone()
            if row:
                bal, staked = row
                conn.execute(
                    "UPDATE balances SET balance = ?, updated_at = ? WHERE address = ?",
                    (bal - amount - fee, int(time.time()), from_addr)
                )
            else:
                # 余额不足或不存在，仍记录为负（应在前置校验拦截）
                conn.execute(
                    "INSERT OR REPLACE INTO balances VALUES (?,?,?,?)",
                    (from_addr, -amount - fee, 0, int(time.time()))
                )
        # 加接收方
        row = conn.execute(
            "SELECT balance, staked FROM balances WHERE address = ?", (to_addr,)
        ).fetchone()
        if row:
            bal, staked = row
            conn.execute(
                "UPDATE balances SET balance = ?, updated_at = ? WHERE address = ?",
                (bal + amount, int(time.time()), to_addr)
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO balances VALUES (?,?,?,?)",
                (to_addr, amount, 0, int(time.time()))
            )

    # ---------- 交易操作 ----------

    def add_pending_tx(self, tx: Transaction) -> bool:
        """添加待确认交易"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO transactions
                    (tx_id, type, from_addr, to_addr, amount, fee, timestamp,
                     payload, signature, signer, block_index, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,NULL,'pending')""",
                    (tx.tx_id, tx.type, tx.from_addr, tx.to_addr, tx.amount,
                     tx.fee, tx.timestamp, json.dumps(tx.payload),
                     tx.signature, tx.signer)
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"添加交易失败: {e}")
                return False
            finally:
                conn.close()

    def produce_block(self, producer_addr: str, producer_kp: crypto.SigningKeyPair) -> Optional[Block]:
        """出块（打包待确认交易）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    "SELECT tx_id, type, from_addr, to_addr, amount, fee, timestamp, payload, signature, signer FROM transactions WHERE status = 'pending' LIMIT 100"
                ).fetchall()
                if not rows:
                    return None
                txs = []
                for r in rows:
                    txs.append({
                        "tx_id": r[0], "type": r[1], "from_addr": r[2],
                        "to_addr": r[3], "amount": r[4], "fee": r[5],
                        "timestamp": r[6], "payload": json.loads(r[7]),
                        "signature": r[8], "signer": r[9],
                    })
                latest = self.get_latest_block()
                new_index = (latest.index + 1) if latest else 1
                prev_hash = latest.hash if latest else "0" * 64
                block = Block(
                    index=new_index,
                    prev_hash=prev_hash,
                    transactions=txs,
                    timestamp=int(time.time()),
                    merkle_root=crypto.merkle_root([t["tx_id"] for t in txs]),
                    producer=producer_addr,
                )
                # 出块者签名
                block.producer_sig = producer_kp.sign_json({
                    "index": block.index, "prev_hash": block.prev_hash,
                    "merkle_root": block.merkle_root, "timestamp": block.timestamp,
                }).hex()
                conn.close()
                if self.append_block(block):
                    logger.info(f"出块成功: #{block.index}, {len(txs)} 笔交易")
                    return block
                return None
            finally:
                if conn:
                    conn.close()

    # ---------- 余额查询 ----------

    def get_balance(self, address: str) -> tuple[float, float]:
        """返回 (余额, 质押)"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT balance, staked FROM balances WHERE address = ?", (address,)
                ).fetchone()
                if row:
                    return (row[0], row[1])
                return (0.0, 0.0)
            finally:
                conn.close()

    def get_my_balance(self) -> tuple[float, float]:
        return self.get_balance(self.my_address)

    def set_balance(self, address: str, balance: float, staked: float = 0):
        """直接设置余额（仅用于初始化/恢复）"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO balances VALUES (?,?,?,?)",
                    (address, balance, staked, int(time.time()))
                )
                conn.commit()
            finally:
                conn.close()

    def update_staked(self, address: str, delta: float):
        """仅更新质押额（delta 可正可负），不影响 balance"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT balance, staked FROM balances WHERE address = ?", (address,)
                ).fetchone()
                if row:
                    bal, staked = row
                    conn.execute(
                        "UPDATE balances SET staked = ?, updated_at = ? WHERE address = ?",
                        (staked + delta, int(time.time()), address)
                    )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO balances VALUES (?,?,?,?)",
                        (address, 0.0, delta, int(time.time()))
                    )
                conn.commit()
            finally:
                conn.close()

    # ---------- 交易历史 ----------

    def get_tx_history(self, address: str, limit: int = 50) -> list[dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    """SELECT tx_id, type, from_addr, to_addr, amount, fee,
                              timestamp, payload, status, block_index
                       FROM transactions
                       WHERE from_addr = ? OR to_addr = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (address, address, limit)
                ).fetchall()
                result = []
                for r in rows:
                    result.append({
                        "tx_id": r[0], "type": r[1], "from_addr": r[2],
                        "to_addr": r[3], "amount": r[4], "fee": r[5],
                        "timestamp": r[6], "payload": json.loads(r[7]),
                        "status": r[8], "block_index": r[9],
                    })
                return result
            finally:
                conn.close()

    # ---------- C2C 订单 ----------

    def place_order(self, order_id: str, order_type: str, maker: str,
                    price: float, amount: float, signature: str, signer: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO orders
                    (order_id, type, maker, price, amount, filled, status, timestamp, signature, signer)
                    VALUES (?,?,?,?,?,0,'open',?,?,?)""",
                    (order_id, order_type, maker, price, amount,
                     int(time.time()), signature, signer)
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"挂单失败: {e}")
                return False
            finally:
                conn.close()

    def get_orders(self, status: str = "open") -> list[dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    "SELECT order_id, type, maker, price, amount, filled, status, timestamp FROM orders WHERE status = ? ORDER BY timestamp DESC",
                    (status,)
                ).fetchall()
                return [{
                    "order_id": r[0], "type": r[1], "maker": r[2], "price": r[3],
                    "amount": r[4], "filled": r[5], "status": r[6], "timestamp": r[7],
                } for r in rows]
            finally:
                conn.close()

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE orders SET status = 'cancelled' WHERE order_id = ?",
                    (order_id,)
                )
                conn.commit()
                return conn.total_changes > 0
            finally:
                conn.close()

    def fill_order(self, order_id: str, filled_amount: float) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT amount, filled FROM orders WHERE order_id = ?",
                    (order_id,)
                ).fetchone()
                if not row:
                    return False
                total, already = row
                new_filled = already + filled_amount
                status = "filled" if new_filled >= total else "open"
                conn.execute(
                    "UPDATE orders SET filled = ?, status = ? WHERE order_id = ?",
                    (new_filled, status, order_id)
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def close(self):
        pass  # SQLite 每次连接独立关闭
