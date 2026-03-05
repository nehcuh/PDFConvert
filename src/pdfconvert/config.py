from pathlib import Path

LMS_CLI = str(Path.home() / ".lmstudio/bin/lms")
LMS_BASE_URL = "http://localhost:1234"
TARGET_MODEL = "qwen3.5-27b"

DEFAULT_DPI = 300
MAX_LONG_EDGE = 2048
MAX_TOKENS = 4096
TEMPERATURE = 0.1
DEFAULT_CONCURRENCY = 2

# 图片保存配置
SAVE_IMAGES_DEFAULT = True
IMAGE_FORMAT = "PNG"
IMAGE_OPTIMIZE = True
