"""GPU 监控模块

使用 NVML 监测 NVIDIA GPU，无 GPU 时回退到 CPU 模拟。
提供 TOPS 估算、实时利用率采样、温度监测、显存监测。
"""
from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# NVIDIA GPU 型号到估算 TOPS 的映射（FP32 理论值，简化版）
GPU_TOPS_TABLE = {
    # RTX 40 系列
    "rtx 4090": 82.6,
    "rtx 4080": 48.7,
    "rtx 4070 ti": 40.1,
    "rtx 4070": 29.1,
    "rtx 4060 ti": 22.1,
    "rtx 4060": 15.1,
    # RTX 30 系列
    "rtx 3090": 35.6,
    "rtx 3080": 29.8,
    "rtx 3070": 20.3,
    "rtx 3060": 12.7,
    "rtx 3050": 8.1,
    # RTX 20 系列
    "rtx 2080 ti": 13.4,
    "rtx 2080": 10.1,
    "rtx 2070": 7.5,
    "rtx 2060": 6.5,
    # GTX 16 系列
    "gtx 1660 ti": 5.4,
    "gtx 1660": 5.0,
    "gtx 1650": 3.0,
    # 专业卡
    "a100": 19.5,
    "v100": 15.7,
    "t4": 8.1,
    "p40": 11.8,
    "p4": 5.5,
}

# CPU 回退 TOPS 估算（基于核心数）
CPU_TOPS_PER_CORE = 0.05  # 每核心约 0.05 TOPS（非常粗略）


@dataclass
class GPUInfo:
    """GPU 信息"""
    available: bool
    device_name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    utilization: float          # GPU 利用率 %
    temperature: float          # 温度 ℃
    power_usage: float          # 功耗 W
    power_limit: float          # 功耗上限 W
    estimated_tops: float       # 估算 TOPS
    is_cuda: bool               # 是否真 CUDA 设备
    index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class GPUMonitor:
    """GPU 监控器（单例）"""

    _instance: Optional["GPUMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._nvml_initialized = False
        self._cpu_mode = False
        self._cpu_cores = 0
        self._sim_utilization = 0.0
        self._sim_temp = 45.0
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._pynvml = pynvml
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML 初始化成功，检测到 {device_count} 个 NVIDIA GPU")
            self._device_count = device_count
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)
            ]
        except Exception as e:
            logger.warning(f"NVML 不可用，切换到 CPU 模拟模式: {e}")
            self._cpu_mode = True
            import os
            self._cpu_cores = os.cpu_count() or 4
            self._device_count = 1  # 模拟一个虚拟设备
            self._handles = []

    def get_info(self, index: int = 0) -> GPUInfo:
        """获取 GPU 信息"""
        if self._cpu_mode:
            return self._get_cpu_info(index)
        try:
            return self._get_gpu_info(index)
        except Exception as e:
            logger.error(f"获取 GPU 信息失败: {e}")
            return self._get_cpu_info(index)

    def _get_gpu_info(self, index: int) -> GPUInfo:
        pynvml = self._pynvml
        if index >= len(self._handles):
            index = 0
        handle = self._handles[index]
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0.0
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            power = 0.0
            power_limit = 0.0

        # 估算 TOPS
        estimated_tops = self._estimate_tops(name)

        return GPUInfo(
            available=True,
            device_name=name,
            total_vram_mb=int(mem.total / 1024 / 1024),
            used_vram_mb=int(mem.used / 1024 / 1024),
            free_vram_mb=int(mem.free / 1024 / 1024),
            utilization=float(util.gpu),
            temperature=float(temp),
            power_usage=float(power),
            power_limit=float(power_limit),
            estimated_tops=estimated_tops,
            is_cuda=True,
            index=index,
        )

    def _get_cpu_info(self, index: int) -> GPUInfo:
        """CPU 模拟模式"""
        import random
        # 模拟负载变化
        self._sim_utilization = max(0.0, min(100.0, self._sim_utilization + random.uniform(-5, 5)))
        self._sim_temp = max(40.0, min(70.0, self._sim_temp + random.uniform(-1, 1)))
        estimated_tops = CPU_TOPS_PER_CORE * self._cpu_cores
        return GPUInfo(
            available=True,
            device_name=f"CPU Mode ({self._cpu_cores} cores)",
            total_vram_mb=8192,  # 模拟 8GB
            used_vram_mb=2048,
            free_vram_mb=6144,
            utilization=self._sim_utilization,
            temperature=self._sim_temp,
            power_usage=65.0,
            power_limit=125.0,
            estimated_tops=estimated_tops,
            is_cuda=False,
            index=index,
        )

    def _estimate_tops(self, name: str) -> float:
        """根据 GPU 名称估算 TOPS"""
        name_lower = name.lower()
        # 精确匹配
        for key, tops in GPU_TOPS_TABLE.items():
            if key in name_lower:
                return tops
        # 默认估算
        if "rtx" in name_lower:
            return 10.0
        if "gtx" in name_lower:
            return 5.0
        if "a100" in name_lower or "v100" in name_lower:
            return 15.0
        return 5.0

    def list_devices(self) -> list[GPUInfo]:
        """列出所有设备"""
        return [self.get_info(i) for i in range(self._device_count)]

    def benchmark(self, duration_sec: int = 5) -> float:
        """基准测试，返回实测 TOPS"""
        import numpy as np
        logger.info(f"开始基准测试，时长 {duration_sec} 秒")
        # 矩阵乘法基准
        size = 1024
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)

        operations = 0
        start = time.time()
        while time.time() - start < duration_sec:
            c = np.dot(a, b)
            operations += 2 * size * size * size  # 乘加算 2 次运算

        elapsed = time.time() - start
        # 转换为 TOPS (1e12 ops/s)
        measured_tops = (operations / elapsed) / 1e12
        # 矩阵乘法的实际 TOPS 通常远低于理论值，做适当放大以贴近 GPU 表现
        # 这里取基准结果，让用户看到真实测量值
        logger.info(f"基准测试完成，实测 {measured_tops:.4f} TOPS")
        return max(measured_tops, 0.001)

    def shutdown(self):
        if self._nvml_initialized:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def get_monitor() -> GPUMonitor:
    """获取 GPU 监控器单例"""
    return GPUMonitor()
