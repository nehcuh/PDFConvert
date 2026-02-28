import subprocess
import time
import requests
from ..config import LMS_CLI, LMS_BASE_URL, TARGET_MODEL


def is_server_running() -> bool:
    try:
        resp = requests.get(f"{LMS_BASE_URL}/v1/models", timeout=5)
        return resp.status_code in (200, 503)
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_loaded_models() -> list[str]:
    resp = requests.get(f"{LMS_BASE_URL}/v1/models", timeout=10)
    if resp.status_code == 503:
        return []
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def ensure_model_loaded() -> str:
    """确保目标模型已加载，返回模型 ID"""
    if not is_server_running():
        raise RuntimeError(
            f"LM Studio 服务未运行。请先启动服务：\n  {LMS_CLI} server start"
        )

    loaded = get_loaded_models()
    match = next((m for m in loaded if TARGET_MODEL in m.lower()), None)
    if match:
        return match

    raise RuntimeError(
        f"模型 {TARGET_MODEL} 未加载。请先加载模型：\n  {LMS_CLI} load {TARGET_MODEL}"
    )
