# PDFConvert 系统架构设计文档

## 1. 项目概述

PDFConvert 是一个 PDF 转 Markdown 工具，使用本地视觉大模型（通过 LM Studio）替代传统 OCR，实现对手写内容、公式、图表的高质量识别与结构化输出。

**技术栈：** Python 3.12 / uv / LM Studio (OpenAI 兼容 API) / GLM-4.6V-Flash (9B)

---

## 2. 模块划分

```
pdfconvert/
├── core/
│   ├── pdf_parser.py        # PDF 解析层：PDF -> PIL Image 列表
│   ├── image_processor.py   # 图片处理层：图片预处理、base64 编码
│   ├── vision_analyzer.py   # 内容分析层：调用视觉模型识别内容
│   └── markdown_writer.py   # Markdown 生成层：组装最终输出
├── lmstudio/
│   ├── client.py            # LM Studio REST API 客户端
│   └── manager.py           # 模型检测、下载、加载管理
├── config.py                # 全局配置
└── cli.py                   # CLI 入口
```

### 各模块职责

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `pdf_parser` | 将 PDF 按页转为图片 | PDF 文件路径 | `list[PIL.Image]` |
| `image_processor` | 图片预处理与 base64 编码 | PIL.Image | base64 字符串 |
| `vision_analyzer` | 调用视觉模型分析页面内容 | base64 图片 + prompt | 页面 Markdown 文本 |
| `markdown_writer` | 合并各页结果，输出文件 | 页面文本列表 | .md 文件 |
| `lmstudio/client` | 封装 LM Studio OpenAI 兼容 API | 请求参数 | API 响应 |
| `lmstudio/manager` | 模型生命周期管理 | 模型标识符 | 模型就绪状态 |

---

## 3. 数据流

```
┌─────────────┐
│  PDF 文件    │
└──────┬──────┘
       │ pdf_parser: pdf2image 按页转图片 (DPI=300)
       ▼
┌─────────────────┐
│ list[PIL.Image]  │  (每页一张图片)
└──────┬──────────┘
       │ image_processor: 缩放 + base64 编码
       ▼
┌──────────────────┐
│ list[base64 str]  │
└──────┬───────────┘
       │ vision_analyzer: 并发调用 LM Studio API
       │ (asyncio.Semaphore 控制并发数)
       ▼
┌─────────────────────┐
│ list[page_markdown]  │  (按页序排列)
└──────┬──────────────┘
       │ markdown_writer: 合并 + 写文件
       ▼
┌─────────────┐
│  .md 文件    │
└─────────────┘
```

**LM Studio 管理流（在主流程之前执行）：**

```
启动 → 检测 LM Studio 服务 (GET /v1/models)
         │
         ├─ 服务未启动 → 通过 lms CLI 启动 (lms server start)
         │
         ▼
       检查目标模型是否已下载 (lms ls)
         │
         ├─ 未下载 → lms get <model-id> 下载模型
         │
         ▼
       检查模型是否已加载 (GET /v1/models)
         │
         ├─ 未加载 → POST /v1/models/load 或 lms load <model>
         │
         ▼
       模型就绪，开始处理 PDF
```

---

## 4. LM Studio 模型检测与加载流程

### 4.1 配置常量

```python
LMS_CLI = "/Users/huchen/.lmstudio/bin/lms"
LMS_BASE_URL = "http://localhost:1234"
TARGET_MODEL = "glm-4.6v-flash"  # lms get 使用的模型标识符
```

### 4.2 管理流程详细步骤

**Step 1 - 检测 LM Studio 服务：**
- `GET http://localhost:1234/v1/models`
- 超时 3 秒，连接失败则执行 `lms server start`，等待服务就绪（轮询，最多 30 秒）

**Step 2 - 检测模型是否已下载：**
- 执行 `lms ls` 解析输出，检查是否包含 `glm-4.6v-flash` 相关条目
- 若未找到，执行 `lms get glm-4.6v-flash` 下载（此为阻塞操作，需流式输出进度）

**Step 3 - 检测模型是否已加载：**
- `GET /v1/models` 返回的 `data` 列表中查找目标模型
- 若未加载，执行 `lms load glm-4.6v-flash` 或通过 API 加载
- 加载后再次确认模型出现在 `/v1/models` 响应中

### 4.3 错误处理

| 场景 | 处理方式 |
|------|----------|
| LM Studio 未安装 | 报错退出，提示用户安装 |
| 模型下载失败 | 重试 1 次，仍失败则报错退出 |
| 模型加载超时（>120s） | 报错退出，提示检查内存 |
| API 调用返回 500 | 重试 2 次，间隔 2 秒 |

---

## 5. 并发处理策略

### 设计原则

本地视觉模型推理是 I/O 密集型操作（等待 LM Studio 响应），适合使用 `asyncio` 并发。但本地模型同一时刻只能处理有限请求，需要控制并发度。

### 实现方案

```python
# 使用 asyncio + httpx 异步客户端
# Semaphore 限制并发请求数，避免 LM Studio 过载

MAX_CONCURRENT_REQUESTS = 2  # 本地 9B 模型建议并发数

async def process_all_pages(pages: list[PageImage]) -> list[str]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
        analyze_page_with_semaphore(semaphore, page, idx)
        for idx, page in enumerate(pages)
    ]
    results = await asyncio.gather(*tasks)
    return results  # 按原始页序返回
```

### 并发度选择依据

| 并发数 | 适用场景 |
|--------|----------|
| 1 | 内存 < 16GB，或模型 > 15B |
| 2 | 16GB+ 内存，9B 模型（推荐默认值） |
| 3-4 | 32GB+ 内存，可通过配置调整 |

用户可通过 CLI 参数 `--concurrency N` 覆盖默认值。

---

## 6. 关键接口定义

### 6.1 PDF 解析层

```python
# core/pdf_parser.py
from PIL import Image

def parse_pdf(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """将 PDF 文件按页转换为 PIL Image 列表。"""
    ...
```

### 6.2 图片处理层

```python
# core/image_processor.py
from PIL import Image

def prepare_image(image: Image.Image, max_long_edge: int = 2048) -> str:
    """缩放图片并转为 base64 编码字符串（PNG 格式）。

    Args:
        image: 原始页面图片
        max_long_edge: 长边最大像素数，超过则等比缩放
    Returns:
        base64 编码的 PNG 图片字符串
    """
    ...
```

### 6.3 内容分析层

```python
# core/vision_analyzer.py

async def analyze_page(
    client: "LMStudioClient",
    image_base64: str,
    page_number: int,
    prompt: str | None = None,
) -> str:
    """调用视觉模型分析单页内容，返回 Markdown 文本。

    Args:
        client: LM Studio API 客户端
        image_base64: base64 编码的页面图片
        page_number: 页码（用于日志和错误信息）
        prompt: 自定义 prompt，None 则使用默认 prompt
    Returns:
        该页的 Markdown 文本
    """
    ...
```

### 6.4 Markdown 生成层

```python
# core/markdown_writer.py
from pathlib import Path

def write_markdown(
    pages: list[str],
    output_path: Path,
    add_page_breaks: bool = True,
) -> Path:
    """将各页 Markdown 文本合并写入文件。

    Args:
        pages: 按页序排列的 Markdown 文本列表
        output_path: 输出文件路径
        add_page_breaks: 是否在页间添加分隔符
    Returns:
        实际写入的文件路径
    """
    ...
```

### 6.5 LM Studio 客户端

```python
# lmstudio/client.py
import httpx

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def chat_with_vision(
        self,
        image_base64: str,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """发送包含图片的 chat completion 请求。"""
        ...

    async def list_models(self) -> list[dict]:
        """获取当前已加载的模型列表。"""
        ...

    async def close(self):
        """关闭 HTTP 客户端连接。"""
        ...
```

### 6.6 LM Studio 管理器

```python
# lmstudio/manager.py

class LMStudioManager:
    def __init__(
        self,
        lms_cli: str = "/Users/huchen/.lmstudio/bin/lms",
        base_url: str = "http://localhost:1234",
        target_model: str = "glm-4.6v-flash",
    ):
        ...

    async def ensure_ready(self) -> str:
        """确保 LM Studio 服务运行且目标模型已加载。

        Returns:
            已加载模型的完整标识符
        Raises:
            RuntimeError: 服务启动失败或模型加载失败
        """
        ...

    async def _check_server(self) -> bool:
        """检测 LM Studio 服务是否在线。"""
        ...

    async def _start_server(self) -> None:
        """通过 lms CLI 启动服务。"""
        ...

    async def _ensure_model_downloaded(self) -> None:
        """检测并下载目标模型。"""
        ...

    async def _load_model(self) -> str:
        """加载目标模型到内存，返回模型标识符。"""
        ...
```

---

## 7. 目录结构设计

```
PDFConvert/
├── pdfconvert/                  # 主包
│   ├── __init__.py
│   ├── cli.py                   # CLI 入口 (argparse)
│   ├── config.py                # 配置常量与默认值
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py        # PDF -> 图片
│   │   ├── image_processor.py   # 图片预处理
│   │   ├── vision_analyzer.py   # 视觉模型调用
│   │   └── markdown_writer.py   # Markdown 输出
│   └── lmstudio/
│       ├── __init__.py
│       ├── client.py            # API 客户端
│       └── manager.py           # 模型管理
├── tests/                       # 测试
│   ├── __init__.py
│   ├── test_pdf_parser.py
│   ├── test_image_processor.py
│   ├── test_vision_analyzer.py
│   ├── test_markdown_writer.py
│   └── test_lmstudio.py
├── docs/
│   └── ARCHITECTURE.md          # 本文档
├── pyproject.toml               # 项目配置与依赖
├── README.md
└── .gitignore
```

---

## 8. 依赖清单

```toml
# pyproject.toml dependencies（替换现有 PaddleOCR 依赖）
dependencies = [
    "httpx>=0.28",          # 异步 HTTP 客户端（调用 LM Studio API）
    "pdf2image>=1.17.0",    # PDF 转图片
    "pillow>=12.1.0",       # 图片处理
]
```

移除的依赖：`paddleocr`、`paddlepaddle`（不再需要传统 OCR）。

---

## 9. CLI 使用方式

```bash
# 基本用法
uv run python -m pdfconvert input.pdf

# 指定输出路径
uv run python -m pdfconvert input.pdf -o output.md

# 调整并发数
uv run python -m pdfconvert input.pdf --concurrency 3

# 指定模型
uv run python -m pdfconvert input.pdf --model glm-4.6v-flash
```

---

## 10. 设计决策记录

| 决策 | 理由 |
|------|------|
| 使用 asyncio 而非多线程 | LM Studio API 调用是 I/O 密集型，asyncio 更轻量 |
| 默认并发数为 2 | 9B 模型在 Apple Silicon 上单次推理已占用大量内存 |
| 图片长边限制 2048px | 平衡识别精度与推理速度，GLM-4.6V 支持的合理上限 |
| 使用 httpx 替代 openai SDK | 更轻量，避免引入不必要的依赖，且 API 调用简单 |
| lms CLI + REST API 双通道管理 | CLI 用于模型下载/加载等管理操作，REST API 用于推理和状态查询 |
