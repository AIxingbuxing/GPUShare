"""去中心化 GPU 算力共享平台 - 主入口

用法:
    # 启动引导节点（第一个节点）
    python main.py --port 9000 --bootstrap --web-port 5000

    # 启动第二个节点
    python main.py --port 9001 --bootstrap-host 127.0.0.1 --bootstrap-port 9000 --web-port 5001
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

# 项目根目录加入 path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from api.server import run_server


def setup_logging(level: str = "INFO"):
    """配置日志"""
    log_format = config.LOG_CONFIG["format"]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(config.LOG_CONFIG["file"]), encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="去中心化 GPU 算力共享平台")
    parser.add_argument("--port", type=int, default=config.P2P_CONFIG["default_port"],
                        help="P2P 监听端口")
    parser.add_argument("--web-port", type=int, default=config.WEB_CONFIG["port"],
                        help="Web 服务端口")
    parser.add_argument("--bootstrap", action="store_true",
                        help="作为引导节点启动")
    parser.add_argument("--bootstrap-host", type=str, default="",
                        help="引导节点地址")
    parser.add_argument("--bootstrap-port", type=int, default=9000,
                        help="引导节点端口")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("去中心化 GPU 算力共享平台 v1.0.0")
    logger.info("=" * 60)
    logger.info(f"P2P 端口: {args.port}")
    logger.info(f"Web 端口: {args.web_port}")
    logger.info(f"数据目录: {config.DATA_DIR}")

    # 解析引导节点
    bootstrap_nodes = []
    if not args.bootstrap and args.bootstrap_host:
        bootstrap_nodes.append((args.bootstrap_host, args.bootstrap_port))
        logger.info(f"引导节点: {args.bootstrap_host}:{args.bootstrap_port}")
    else:
        logger.info("作为引导节点启动")

    # 启动 Web 服务器（内含 P2P 节点初始化）
    try:
        run_server(
            listen_port=args.port,
            bootstrap_nodes=bootstrap_nodes,
            web_port=args.web_port,
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
    except Exception as e:
        logger.exception(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
