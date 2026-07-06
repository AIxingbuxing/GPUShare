"""加密工具模块

提供 Ed25519 签名、X25519 密钥交换、NaCl Box 端对端加密、SHA256 哈希、
BIP-39 风格助记词生成（简化版）、AES-256-GCM 对称加密。
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, Dict, Tuple

import nacl.encoding
import nacl.hash
import nacl.secret
import nacl.signing
import nacl.utils
from nacl.public import PrivateKey, PublicKey, SealedBox


# ---------- 哈希 ----------

def sha256_hex(data: bytes | str) -> str:
    """SHA256 哈希，返回 hex 字符串"""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def blake2b_hex(data: bytes | str, digest_size: int = 32) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


# ---------- Ed25519 签名密钥对 ----------

class SigningKeyPair:
    """Ed25519 签名密钥对"""

    # Ed25519 种子固定 32 字节
    SEED_SIZE = 32

    def __init__(self, seed: bytes | None = None):
        if seed is None:
            seed = nacl.utils.random(self.SEED_SIZE)
        self.signing_key = nacl.signing.SigningKey(seed)
        self.verify_key = self.signing_key.verify_key

    @property
    def private_bytes(self) -> bytes:
        return bytes(self.signing_key)

    @property
    def public_bytes(self) -> bytes:
        return bytes(self.verify_key)

    @property
    def address(self) -> str:
        """基于公钥生成的地址（前缀 + 公钥哈希前 20 字节）"""
        pk_hash = sha256_hex(self.public_bytes)
        return "GPU" + pk_hash[:32]

    def sign(self, data: bytes | str) -> bytes:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.signing_key.sign(data).signature

    def sign_json(self, obj: Any) -> bytes:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.sign(data)

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> "SigningKeyPair":
        return cls(seed=private_bytes)


def verify_signature(public_bytes: bytes, signature: bytes, data: bytes | str) -> bool:
    """验证 Ed25519 签名"""
    try:
        if isinstance(data, str):
            data = data.encode("utf-8")
        vk = nacl.signing.VerifyKey(public_bytes)
        vk.verify(data, signature)
        return True
    except Exception:
        return False


def verify_json_signature(public_bytes: bytes, signature: bytes, obj: Any) -> bool:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return verify_signature(public_bytes, signature, data)


# ---------- X25519 加密密钥对 + NaCl Box ----------

class EncryptionKeyPair:
    """X25519 加密密钥对，用于端对端加密"""

    def __init__(self, private_key: PrivateKey | None = None):
        if private_key is None:
            private_key = PrivateKey.generate()
        self.private_key = private_key
        self.public_key = private_key.public_key

    @property
    def private_bytes(self) -> bytes:
        return bytes(self.private_key)

    @property
    def public_bytes(self) -> bytes:
        return bytes(self.public_key)

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> "EncryptionKeyPair":
        return cls(private_key=PrivateKey(private_bytes))

    def encrypt_to(self, recipient_public_bytes: bytes, plaintext: bytes) -> bytes:
        """加密给指定接收方（带发送方公钥前缀）"""
        from nacl.public import Box
        recipient_pk = PublicKey(recipient_public_bytes)
        box = Box(self.private_key, recipient_pk)
        nonce = nacl.utils.random(Box.NONCE_SIZE)
        encrypted = box.encrypt(plaintext, nonce)
        return self.public_bytes + encrypted  # 前 32 字节为发送方公钥

    def decrypt_from(self, sender_public_bytes: bytes, ciphertext: bytes) -> bytes:
        """解密来自指定发送方的密文"""
        from nacl.public import Box
        sender_pk = PublicKey(sender_public_bytes)
        box = Box(self.private_key, sender_pk)
        # ciphertext 前 32 字节是发送方公钥（已在 encrypt_to 中拼接），但调用方需先剥离
        return box.decrypt(ciphertext)

    def seal(self, plaintext: bytes) -> bytes:
        """匿名密封加密（无需发送方身份）"""
        sealed_box = SealedBox(self.public_key)
        return sealed_box.encrypt(plaintext)

    def open_seal(self, ciphertext: bytes) -> bytes:
        """解密封匿名密文"""
        sealed_box = SealedBox(self.private_key)
        return sealed_box.decrypt(ciphertext)


def encrypt_box(sender: EncryptionKeyPair, recipient_pub: bytes, plaintext: bytes) -> bytes:
    """便捷封装：发送方加密给接收方，返回 [发送方公钥(32) | nonce(24) | 密文]"""
    return sender.encrypt_to(recipient_pub, plaintext)


def decrypt_box(recipient: EncryptionKeyPair, ciphertext: bytes) -> bytes:
    """便捷封装：接收方解密，自动剥离前 32 字节发送方公钥"""
    if len(ciphertext) < 32:
        raise ValueError("密文过短")
    sender_pub = ciphertext[:32]
    payload = ciphertext[32:]
    return recipient.decrypt_from(sender_pub, payload)


# ---------- 对称加密（本地存储） ----------

def derive_key(passphrase: str, salt: bytes) -> bytes:
    """从口令派生密钥"""
    import hashlib
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 100000, 32)


def symmetric_encrypt(passphrase: str, plaintext: bytes) -> bytes:
    """AES-256 对称加密（NaCl SecretBox，用 PBKDF2 派生密钥）"""
    salt = nacl.utils.random(16)
    key = derive_key(passphrase, salt)
    box = nacl.secret.SecretBox(key)
    encrypted = box.encrypt(plaintext)
    return salt + encrypted  # 前 16 字节 salt


def symmetric_decrypt(passphrase: str, ciphertext: bytes) -> bytes:
    """对称解密"""
    if len(ciphertext) < 16:
        raise ValueError("密文过短")
    salt = ciphertext[:16]
    payload = ciphertext[16:]
    key = derive_key(passphrase, salt)
    box = nacl.secret.SecretBox(key)
    return box.decrypt(payload)


# ---------- 助记词（BIP-39 简化版，2048 词表子集） ----------

# 简化助记词表（实际 BIP-39 是 2048 词，这里取 256 词做演示，扩展为完整词表需加载 bip39.txt）
_MNEMONIC_WORDS = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
    "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
    "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
    "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
    "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
    "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
    "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
    "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
    "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
    "army", "around", "arrange", "arrest", "arrive", "arrow", "art", "artefact",
    "artist", "artwork", "ask", "aspect", "assault", "asset", "assist", "assume",
    "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
    "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado",
    "avoid", "awake", "aware", "away", "awesome", "awful", "awkward", "axis",
    "baby", "bachelor", "bacon", "badge", "bag", "balance", "balcony", "ball",
    "bamboo", "banana", "banner", "bar", "barely", "bargain", "barrel", "base",
    "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
    "beef", "before", "begin", "behave", "behind", "believe", "below", "belt",
    "bench", "benefit", "best", "betray", "better", "between", "beyond", "bicycle",
    "bid", "bike", "bind", "biology", "bird", "birth", "bitter", "black",
    "blade", "blame", "blanket", "blast", "bleak", "bless", "blind", "blood",
    "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
    "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring",
    "borrow", "boss", "bottom", "bounce", "box", "boy", "bracket", "brain",
    "brand", "brass", "brave", "bread", "breeze", "brick", "bridge", "brief",
    "bright", "bring", "brisk", "broccoli", "broken", "bronze", "broom", "brother",
    "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
    "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus",
    "business", "busy", "butter", "buyer", "buzz", "cabbage", "cabin", "cable",
]


def generate_mnemonic(word_count: int = 12) -> str:
    """生成助记词"""
    words = []
    for _ in range(word_count):
        idx = secrets.randbelow(len(_MNEMONIC_WORDS))
        words.append(_MNEMONIC_WORDS[idx])
    return " ".join(words)


def mnemonic_to_seed(mnemonic: str) -> bytes:
    """助记词转种子（PBKDF2）"""
    salt = b"gpu-share-mnemonic-salt"
    return hashlib.pbkdf2_hmac("sha256", mnemonic.encode("utf-8"), salt, 2048, 32)


def is_valid_mnemonic(mnemonic: str) -> bool:
    """校验助记词格式"""
    words = mnemonic.strip().split()
    if len(words) != 12:
        return False
    word_set = set(_MNEMONIC_WORDS)
    return all(w in word_set for w in words)


# ---------- Merkle 树 ----------

def merkle_root(hashes: list[str]) -> str:
    """计算 Merkle 根"""
    if not hashes:
        return sha256_hex(b"")
    if len(hashes) == 1:
        return hashes[0]
    # 补齐偶数
    layer = list(hashes)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer = []
        for i in range(0, len(layer), 2):
            next_layer.append(sha256_hex(layer[i] + layer[i + 1]))
        layer = next_layer
    return layer[0]


# ---------- 通用工具 ----------

def random_nonce_hex(n: int = 16) -> str:
    return secrets.token_hex(n)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sign_dict(signing_kp: SigningKeyPair, obj: Dict) -> Dict:
    """对字典签名，返回带 signature 字段的新字典"""
    sig = signing_kp.sign_json(obj)
    return {**obj, "signature": sig.hex(), "signer": signing_kp.public_bytes.hex()}


def verify_signed_dict(signed_obj: Dict, required_fields: list[str] | None = None) -> bool:
    """验证带 signature/signer 字段的字典"""
    if "signature" not in signed_obj or "signer" not in signed_obj:
        return False
    try:
        signature = bytes.fromhex(signed_obj["signature"])
        signer_pub = bytes.fromhex(signed_obj["signer"])
        obj_to_verify = {k: v for k, v in signed_obj.items() if k not in ("signature", "signer")}
        return verify_json_signature(signer_pub, signature, obj_to_verify)
    except Exception:
        return False
