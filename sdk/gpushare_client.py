"""GPUShare Python SDK

供本地其他软件（大模型、科学计算等）一行代码接入 GPUShare 算力调度。

使用示例：

    # 推理请求
    from gpushare import ComputeClient
    client = ComputeClient()  # 默认连接本地 GPUShare
    result = client.inference("llama2", [{"role": "user", "content": "你好"}])

    # 张量计算
    result = client.tensor_compute("matmul", {"shape": [1024, 1024]})

    # OpenAI 兼容模式（配合 openai 库使用）
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="gpushare")
    response = client.chat.completions.create(
        model="llama2",
        messages=[{"role": "user", "content": "你好"}],
    )
"""
import json
import urllib.request
import urllib.error


class ComputeClient:
    """GPUShare 计算客户端

    本地其他软件通过此客户端提交计算需求，GPUShare 自动调度本地/远程 GPU。

    Args:
        base_url: GPUShare 服务地址，默认 http://127.0.0.1:5000
        source: 来源标识，用于调度日志追踪
    """

    def __init__(self, base_url: str = "http://127.0.0.1:5000",
                 source: str = "sdk"):
        self.base_url = base_url.rstrip("/")
        self.source = source

    def inference(self, model: str, messages: list[dict],
                  **kwargs) -> dict:
        """推理请求

        Args:
            model: 模型名称（如 "llama2", "qwen2"）
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数（temperature, max_tokens 等）

        Returns:
            {"request_id": "...", "status": "completed", "schedule": "local/remote/mixed",
             "result": {...}, "latency_ms": 123}

        调度说明：
            - 本地 GPU 空闲 → 本地执行（零延迟）
            - 本地 GPU 满载 → 自动分发到远程 GPU 节点
            - 远程不可用 → 本地兜底
        """
        payload = {
            "model": model,
            "messages": messages,
            "source": self.source,
            **kwargs,
        }
        return self._post("/api/compute/inference", payload)

    def tensor_compute(self, operation: str, data: dict) -> dict:
        """张量计算请求

        Args:
            operation: 运算类型（"matmul", "vector_add", "reduce"）
            data: 运算参数 {"shape": [m, n], "seed": 42}

        Returns:
            {"request_id": "...", "status": "completed", "result": {...}}
        """
        payload = {"operation": operation, "source": self.source, **data}
        return self._post("/api/compute/tensor", payload)

    def get_stats(self) -> dict:
        """获取计算代理统计"""
        return self._get("/api/compute/stats")

    def get_logs(self, limit: int = 50) -> list[dict]:
        """获取调度日志"""
        return self._get(f"/api/compute/logs?limit={limit}")

    def get_requests(self, limit: int = 20) -> list[dict]:
        """获取最近计算请求"""
        return self._get(f"/api/compute/requests?limit={limit}")

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": e.read().decode("utf-8")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}


# 便捷函数
def quick_inference(prompt: str, model: str = "llama2",
                    base_url: str = "http://127.0.0.1:5000") -> str:
    """快速推理（一行代码）

    >>> from gpushare import quick_inference
    >>> answer = quick_inference("什么是去中心化？")
    """
    client = ComputeClient(base_url)
    result = client.inference(model, [{"role": "user", "content": prompt}])
    if result.get("ok"):
        inner = result["data"].get("result", {})
        return inner.get("message", {}).get("content", "")
    return result.get("error", "推理失败")
