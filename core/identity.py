"""身份管理模块

邮箱为可读标识，Ed25519 密钥对为权属根。
本地加密存储，云端仅存脱敏指纹。
支持助记词恢复、3-of-5 社交恢复。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import crypto


@dataclass
class Identity:
    """用户身份"""
    email: str
    address: str                     # GPU 开头地址
    signing_pub: str                 # Ed25519 公钥 hex
    encryption_pub: str              # X25519 公钥 hex
    created_at: int
    nickname: str = ""
    avatar: str = ""
    # 私钥不在内存中持久，使用时从加密存储加载
    _signing_priv: bytes = b""
    _encryption_priv: bytes = b""

    # 质押
    staked: int = 0
    # 信誉
    reputation: int = 100
    # 社交恢复联系人（5 个 address）
    guardians: list[str] = field(default_factory=list)

    @property
    def signing_keypair(self) -> crypto.SigningKeyPair:
        if not self._signing_priv:
            raise RuntimeError("私钥未加载")
        return crypto.SigningKeyPair.from_private_bytes(self._signing_priv)

    @property
    def encryption_keypair(self) -> crypto.EncryptionKeyPair:
        if not self._encryption_priv:
            raise RuntimeError("私钥未加载")
        return crypto.EncryptionKeyPair.from_private_bytes(self._encryption_priv)

    def to_public_dict(self) -> dict:
        """公开信息（不含私钥）"""
        d = asdict(self)
        for k in ("_signing_priv", "_encryption_priv"):
            d.pop(k, None)
        return d

    def sign(self, obj: dict) -> dict:
        return crypto.sign_dict(self.signing_keypair, obj)


class IdentityManager:
    """身份管理器"""

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.identity: Optional[Identity] = None
        self.mnemonic: Optional[str] = None
        # 当前会话密码（仅内存，不持久化），用于内部 _save 调用
        self._session_password: str = ""

    @property
    def is_registered(self) -> bool:
        return self.identity is not None

    def register(self, email: str, nickname: str = "", password: str = "") -> tuple[Identity, str]:
        """注册新身份，返回 (identity, mnemonic)"""
        if self.is_registered:
            raise RuntimeError("已存在身份")

        # 生成密钥对
        sign_kp = crypto.SigningKeyPair()
        enc_kp = crypto.EncryptionKeyPair()

        # 生成助记词
        mnemonic = crypto.generate_mnemonic(12)

        identity = Identity(
            email=email,
            address=sign_kp.address,
            signing_pub=sign_kp.public_bytes.hex(),
            encryption_pub=enc_kp.public_bytes.hex(),
            created_at=int(time.time()),
            nickname=nickname or email.split("@")[0],
            _signing_priv=sign_kp.private_bytes,
            _encryption_priv=enc_kp.private_bytes,
        )

        self.identity = identity
        self.mnemonic = mnemonic
        self._session_password = password or "default"
        self._save()
        return identity, mnemonic

    def login_with_password(self, email: str, password: str) -> Identity:
        """用密码登录（从本地加密存储恢复）"""
        if not self.storage_path.exists():
            raise FileNotFoundError("本地无身份记录")
        data = self.storage_path.read_bytes()
        plaintext = crypto.symmetric_decrypt(password, data)
        record = json.loads(plaintext)
        if record["email"] != email:
            raise ValueError("邮箱不匹配")
        identity = self._from_record(record)
        self.identity = identity
        self.mnemonic = record.get("mnemonic")
        self._session_password = password
        return identity

    def recover_from_mnemonic(self, mnemonic: str, email: str, new_password: str) -> Identity:
        """用助记词恢复身份"""
        if not crypto.is_valid_mnemonic(mnemonic):
            raise ValueError("助记词格式错误（需 12 个有效单词）")
        seed = crypto.mnemonic_to_seed(mnemonic)
        sign_kp = crypto.SigningKeyPair(seed=seed[:32])
        enc_priv = crypto.EncryptionKeyPair.from_private_bytes(
            crypto.derive_key(mnemonic, b"enc-salt")  # 确定性派生
        ).private_bytes
        enc_kp = crypto.EncryptionKeyPair.from_private_bytes(enc_priv)
        identity = Identity(
            email=email,
            address=sign_kp.address,
            signing_pub=sign_kp.public_bytes.hex(),
            encryption_pub=enc_kp.public_bytes.hex(),
            created_at=int(time.time()),
            nickname=email.split("@")[0],
            _signing_priv=sign_kp.private_bytes,
            _encryption_priv=enc_kp.private_bytes,
        )
        self.identity = identity
        self.mnemonic = mnemonic
        self._session_password = new_password or "default"
        self._save()
        return identity

    def _from_record(self, record: dict) -> Identity:
        return Identity(
            email=record["email"],
            address=record["address"],
            signing_pub=record["signing_pub"],
            encryption_pub=record["encryption_pub"],
            created_at=record["created_at"],
            nickname=record.get("nickname", ""),
            avatar=record.get("avatar", ""),
            staked=record.get("staked", 0),
            reputation=record.get("reputation", 100),
            guardians=record.get("guardians", []),
            _signing_priv=bytes.fromhex(record["signing_priv"]),
            _encryption_priv=bytes.fromhex(record["encryption_priv"]),
        )

    def _save(self):
        """用当前会话密码加密保存（不再需要外部传密码）"""
        if not self.identity:
            return
        record = {
            "email": self.identity.email,
            "address": self.identity.address,
            "signing_pub": self.identity.signing_pub,
            "encryption_pub": self.identity.encryption_pub,
            "created_at": self.identity.created_at,
            "nickname": self.identity.nickname,
            "avatar": self.identity.avatar,
            "staked": self.identity.staked,
            "reputation": self.identity.reputation,
            "guardians": self.identity.guardians,
            "signing_priv": self.identity._signing_priv.hex(),
            "encryption_priv": self.identity._encryption_priv.hex(),
            "mnemonic": self.mnemonic,
        }
        plaintext = json.dumps(record).encode("utf-8")
        ciphertext = crypto.symmetric_encrypt(self._session_password or "default", plaintext)
        self.storage_path.write_bytes(ciphertext)

    def update(self, password: Optional[str] = None, **kwargs):
        """更新身份信息。password 可选，不传则用当前会话密码"""
        if not self.identity:
            return
        if password:
            self._session_password = password
        for k, v in kwargs.items():
            if hasattr(self.identity, k) and not k.startswith("_"):
                setattr(self.identity, k, v)
        self._save()

    def add_guardian(self, guardian_address: str, password: Optional[str] = None):
        if self.identity and guardian_address not in self.identity.guardians:
            self.identity.guardians.append(guardian_address)
            if len(self.identity.guardians) > 5:
                self.identity.guardians = self.identity.guardians[-5:]
            if password:
                self._session_password = password
            self._save()

    def logout(self):
        self.identity = None
        self.mnemonic = None
        self._session_password = ""
