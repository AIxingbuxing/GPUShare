"""全局配置"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 任务样本目录
TASKS_DIR = BASE_DIR / "tasks"

# 用户数据目录（每个节点本地）
USER_DATA_DIR = DATA_DIR / "user"
USER_DATA_DIR.mkdir(exist_ok=True)

# 账本数据库
LEDGER_DB = USER_DATA_DIR / "ledger.db"

# 身份密钥文件
IDENTITY_FILE = USER_DATA_DIR / "identity.json"

# P2P 网络配置
P2P_CONFIG = {
    "listen_host": "0.0.0.0",
    "default_port": 9000,
    "bootstrap_nodes": [
        # ("127.0.0.1", 9000),  # 默认引导节点
    ],
    "k_bucket_size": 16,
    "heartbeat_interval": 5,        # 心跳间隔（秒）
    "node_timeout": 15,             # 节点超时（秒）
    "discovery_interval": 30,       # 发现间隔（秒）
}

# Web API 配置
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": False,
}

# TPT 经济模型
ECONOMICS = {
    "tpt_unit": "1 TPT = 1 TOPS * 1 Min",
    "fee_rate": 0.02,               # 总手续费 2%
    "witness_pool_rate": 0.01,      # 见证网络激励池 1%
    "ops_pool_rate": 0.005,         # 运维池 0.5%
    "revenue_pool_rate": 0.005,     # 收益池 0.5%
    "contributor_rate": 0.98,       # 贡献者收益 98%
    "stake_min": 100,               # 贡献者最低质押 TPT
    "stake_slash_rate": 0.5,        # 单次作恶扣除质押比例
    "freeze_buffer": 1.1,           # 任务前冗余冻结倍数
    "challenge_period": 3600,       # 挑战期 1 小时（秒）
    "new_user_bonus": 200,          # 新用户赠送 TPT
    "benchmark_duration": 30,       # 基准测试时长（秒）
    "checkpoint_interval": 10,      # 断点续算间隔（秒）
}

# 任务类型
TASK_TYPES = {
    "image_blur": {
        "name": "图像模糊处理",
        "description": "对图像执行高斯模糊，GPU 加速",
        "input_type": "image",
        "estimated_tops": 5.0,
    },
    "matrix_mult": {
        "name": "矩阵乘法",
        "description": "大规模矩阵乘法运算",
        "input_type": "matrix_spec",
        "estimated_tops": 10.0,
    },
    "hash_benchmark": {
        "name": "哈希基准测试",
        "description": "SHA256 哈希基准，验证算力真实性",
        "input_type": "hash_spec",
        "estimated_tops": 3.0,
    },
    "ml_inference": {
        "name": "ML 推理",
        "description": "简单神经网络前向推理",
        "input_type": "ml_spec",
        "estimated_tops": 8.0,
    },
}

# GPU 配置
GPU_CONFIG = {
    "default_utilization_limit": 80,  # 默认 GPU 使用率上限 %
    "default_vram_reserve_mb": 512,   # 默认显存预留 MB
    "temp_threshold": 85,             # 温度熔断阈值 ℃
    "utilization_sample_interval": 1, # 采样间隔（秒）
}

# 见证网络配置
WITNESS_CONFIG = {
    "witness_count": 5,              # 见证节点数量
    "notarization_timeout": 30,      # 公证超时（秒）
    "min_witness_signatures": 3,     # 最少见证签名
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file": DATA_DIR / "node.log",
}
