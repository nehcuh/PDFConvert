# PDFConvert 后端技术设计文档

## 1. 项目目录结构

```
PDFConvert/
├── main.py                    # CLI 入口
├── pyproject.toml             # 项目配置与依赖
├── uv.lock
├── README.md
├── docs/
│   └── BACKEND_DESIGN.md      # 本文档
├── src/
│   └── pdfconvert/
│       ├── __init__.py
│       ├── lmstudio_manager.py    # LM Studio 进程与模型管理
│       ├── pdf_processor.py       # PDF → 图片转换
│       ├── vision_analyzer.py     # 视觉模型 API 调用
│       ├── content_classifier.py  # 内容类型分类
│       ├── markdown_builder.py    # Markdown 组装输出
│       └── prompts.py             # Prompt 模板集中管理
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
└── output/                        # 默认输出目录（gitignore）
```

---

## 2. 核心模块设计

### 2.1 lmstudio_manager.py — LM Studio 进程与模型管理

负责检测 LM Studio 服务状态、确认模型是否已加载、必要时通过 CLI 自动加载模型。

```python
import subprocess
import httpx

LMS_CLI = "/Users/huchen/.lmstudio/bin/lms"
LMS_BASE_URL = "http://localhost:1234"
TARGET_MODEL = "glm-4v-flash-9b"  # 模型标识符，以实际 lms ls 输出为准


def is_server_running() -> bool:
    """检测 LM Studio HTTP 服务是否可达"""
    try:
        resp = httpx.get(f"{LMS_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False


def get_loaded_models() -> list[str]:
    """返回当前已加载的模型 ID 列表"""
    resp = httpx.get(f"{LMS_BASE_URL}/v1/models", timeout=10)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def ensure_model_loaded() -> str:
    """确保目标模型已加载，返回可用的模型 ID。
    如果服务未启动则先启动；如果模型未加载则通过 CLI 加载。
    """
    if not is_server_running():
        subprocess.run([LMS_CLI, "server", "start"], check=True)

    loaded = get_loaded_models()
    # 模糊匹配：模型 ID 可能包含量化后缀
    match = next((m for m in loaded if TARGET_MODEL in m.lower()), None)
    if match:
        return match

    # 未加载 → 通过 CLI 加载
    subprocess.run([LMS_CLI, "load", TARGET_MODEL], check=True)
    loaded = get_loaded_models()
    match = next((m for m in loaded if TARGET_MODEL in m.lower()), None)
    if not match:
        raise RuntimeError(f"模型 {TARGET_MODEL} 加载失败")
    return match
```

### 2.2 pdf_processor.py — PDF 转图片

将 PDF 按页转为 PIL Image，支持 DPI 配置和分页迭代。

```python
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

DEFAULT_DPI = 300


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> list[Image.Image]:
    """将 PDF 文件转换为 PIL Image 列表，每页一张。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    return convert_from_path(str(pdf_path), dpi=dpi)


def pdf_page_count(pdf_path: str | Path) -> int:
    """快速获取 PDF 页数（不渲染图片）。"""
    from pdf2image.pdf2image import pdfinfo_from_path
    info = pdfinfo_from_path(str(pdf_path))
    return info["Pages"]
```

### 2.3 vision_analyzer.py — 视觉模型 API 调用

通过 OpenAI 兼容 API 将页面图片发送给 GLM-4V-Flash 进行分析。

```python
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

LMS_BASE_URL = "http://localhost:1234/v1"


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """将 PIL Image 编码为 base64 data URI。"""
    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def analyze_page(
    image: Image.Image,
    prompt: str,
    model_id: str,
) -> str:
    """调用 LM Studio 视觉 API 分析单页图片，返回模型文本输出。"""
    client = OpenAI(base_url=LMS_BASE_URL, api_key="lm-studio")
    data_uri = image_to_base64(image)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.1,  # 低温度保证输出稳定
    )
    return response.choices[0].message.content
```

### 2.4 content_classifier.py — 内容类型分类

对页面进行预分类，决定使用哪套 Prompt 进行精细提取。

```python
from enum import Enum


class ContentType(Enum):
    TEXT = "text"           # 纯文字段落
    FORMULA = "formula"     # 数学/物理公式
    DIAGRAM = "diagram"     # 图表、示意图
    MIXED = "mixed"         # 混合内容（默认）


def classify_page(analysis_text: str) -> ContentType:
    """根据视觉模型初步分析结果判断页面主要内容类型。
    analysis_text: 由 vision_analyzer.analyze_page 使用分类 prompt 返回的文本。
    """
    lower = analysis_text.lower()
    if "diagram" in lower or "graph" in lower or "figure" in lower:
        return ContentType.DIAGRAM
    if "equation" in lower or "formula" in lower or "latex" in lower:
        return ContentType.FORMULA
    if "text" in lower or "paragraph" in lower:
        return ContentType.TEXT
    return ContentType.MIXED
```

### 2.5 markdown_builder.py — Markdown 组装输出

将各页分析结果组装为完整的 Markdown 文档。

```python
from pathlib import Path


def build_markdown(
    pages: list[dict],
    title: str | None = None,
) -> str:
    """将各页分析结果组装为 Markdown 字符串。

    Args:
        pages: 每个元素为 {"page_num": int, "content": str, "content_type": str}
        title: 可选文档标题
    Returns:
        完整 Markdown 文本
    """
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    for page in pages:
        num = page["page_num"]
        parts.append(f"\n---\n\n## Page {num}\n")
        parts.append(page["content"])

    return "\n".join(parts)


def save_markdown(content: str, output_path: str | Path) -> Path:
    """将 Markdown 内容写入文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
```

---

## 3. LM Studio API 调用示例

LM Studio 提供 OpenAI 兼容的 `/v1/chat/completions` 端点，视觉模型通过 `image_url` 字段接收 base64 编码图片。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

response = client.chat.completions.create(
    model="glm-4v-flash-9b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请提取这张图片中的所有文字内容。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo..."
                    },
                },
            ],
        }
    ],
    max_tokens=4096,
    temperature=0.1,
)

print(response.choices[0].message.content)
```

关键点：
- `api_key` 可以是任意非空字符串，LM Studio 不校验
- `model` 字段需与 `/v1/models` 返回的 ID 匹配
- 图片通过 `data:image/png;base64,{b64_string}` 格式传入
- `temperature=0.1` 保证输出稳定性，适合文档提取场景

---

## 4. Prompt 设计（prompts.py）

集中管理三套 Prompt，供 `vision_analyzer` 按内容类型选用。

### 4.1 分类 Prompt（首轮调用，判断页面内容类型）

```python
CLASSIFY_PROMPT = """Analyze this page image and classify its content.
Reply with ONLY one or more of these labels, comma-separated:
- TEXT: contains paragraphs of readable text
- FORMULA: contains mathematical or physics equations/formulas
- DIAGRAM: contains charts, graphs, figures, or diagrams

Example reply: TEXT, FORMULA"""
```

### 4.2 文字提取 Prompt

```python
TEXT_EXTRACT_PROMPT = """You are a precise document OCR assistant.
Extract ALL text content from this page image exactly as it appears.
Rules:
- Preserve the original paragraph structure and line breaks.
- Keep headings, bullet points, and numbered lists in Markdown format.
- If there are tables, convert them to Markdown table syntax.
- Do NOT add any commentary or explanation. Output only the extracted text.
- For any text you cannot read clearly, mark it as [unclear].
- Preserve both English and Chinese text as-is."""
```

### 4.3 公式 → LaTeX Prompt

```python
FORMULA_EXTRACT_PROMPT = """You are a LaTeX transcription assistant.
Convert ALL mathematical and physics formulas in this image to LaTeX.
Rules:
- Wrap inline formulas with single dollar signs: $...$
- Wrap display/block formulas with double dollar signs: $$...$$
- Preserve surrounding text context in plain text.
- Use standard LaTeX packages (amsmath, amssymb) notation.
- For vectors use \\vec{}, for matrices use \\begin{pmatrix}.
- If a formula is unclear, provide your best interpretation with a
  comment: %% [uncertain]
- Output the formulas in the order they appear on the page."""
```

### 4.4 图表描述 Prompt

```python
DIAGRAM_DESCRIBE_PROMPT = """You are a technical diagram analyst.
Describe the diagram/chart/figure in this image for a Markdown document.
Rules:
- Start with a brief one-line caption: **Figure: [description]**
- Describe the diagram structure, labels, axes, and data trends.
- If it contains numerical data, extract key values into a Markdown table.
- If it is a circuit diagram or physics diagram, describe components
  and their connections.
- Keep the description concise but complete enough to understand
  without seeing the image."""
```

---

## 5. 更新后的 pyproject.toml

移除 PaddleOCR/PaddlePaddle 依赖，引入 OpenAI SDK 和 httpx。

```toml
[project]
name = "pdfconvert"
version = "0.2.0"
description = "PDF to Markdown converter using LM Studio vision models"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.40.0",       # OpenAI 兼容 API 客户端
    "httpx>=0.27.0",        # HTTP 客户端（模型状态检测）
    "pdf2image>=1.17.0",    # PDF → 图片转换
    "pillow>=12.1.1",       # 图片处理
]

[project.scripts]
pdfconvert = "pdfconvert.main:main"
```

变更说明：
| 操作 | 包名 | 原因 |
|------|------|------|
| 移除 | `paddleocr` | 不再使用 PaddleOCR |
| 移除 | `paddlepaddle` | PaddleOCR 的运行时依赖 |
| 新增 | `openai` | 调用 LM Studio OpenAI 兼容 API |
| 新增 | `httpx` | 轻量 HTTP 客户端，用于服务健康检查 |
| 保留 | `pdf2image` | PDF 转图片核心功能 |
| 保留 | `pillow` | 图片处理 |

---

## 6. 关键函数签名与类型注解汇总

```python
# --- lmstudio_manager.py ---
def is_server_running() -> bool: ...
def get_loaded_models() -> list[str]: ...
def ensure_model_loaded() -> str: ...

# --- pdf_processor.py ---
def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]: ...
def pdf_page_count(pdf_path: str | Path) -> int: ...

# --- vision_analyzer.py ---
def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str: ...
def analyze_page(image: Image.Image, prompt: str, model_id: str) -> str: ...

# --- content_classifier.py ---
class ContentType(Enum):
    TEXT = "text"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    MIXED = "mixed"

def classify_page(analysis_text: str) -> ContentType: ...

# --- markdown_builder.py ---
def build_markdown(pages: list[dict], title: str | None = None) -> str: ...
def save_markdown(content: str, output_path: str | Path) -> Path: ...
```

---

## 7. 主流程（main.py 入口逻辑）

```python
def convert(pdf_path: str, output_path: str | None = None) -> Path:
    """完整转换流程：PDF → 图片 → 视觉分析 → Markdown"""
    # 1. 确保模型就绪
    model_id = lmstudio_manager.ensure_model_loaded()

    # 2. PDF 转图片
    images = pdf_processor.pdf_to_images(pdf_path)

    pages = []
    for i, image in enumerate(images, 1):
        # 3. 分类：判断页面内容类型
        classify_result = vision_analyzer.analyze_page(
            image, prompts.CLASSIFY_PROMPT, model_id
        )
        content_type = content_classifier.classify_page(classify_result)

        # 4. 按类型选择对应 Prompt 进行精细提取
        prompt_map = {
            ContentType.TEXT: prompts.TEXT_EXTRACT_PROMPT,
            ContentType.FORMULA: prompts.FORMULA_EXTRACT_PROMPT,
            ContentType.DIAGRAM: prompts.DIAGRAM_DESCRIBE_PROMPT,
            ContentType.MIXED: prompts.TEXT_EXTRACT_PROMPT,  # 混合默认用文字提取
        }
        detail_prompt = prompt_map[content_type]
        content = vision_analyzer.analyze_page(image, detail_prompt, model_id)

        pages.append({
            "page_num": i,
            "content": content,
            "content_type": content_type.value,
        })

    # 5. 组装并保存 Markdown
    md_text = markdown_builder.build_markdown(pages)
    out = markdown_builder.save_markdown(md_text, output_path or f"{Path(pdf_path).stem}.md")
    return out
```

---

## 8. 设计要点与注意事项

1. **两轮调用策略**：每页先用分类 Prompt 判断内容类型，再用对应的专用 Prompt 精细提取。虽然增加了一次 API 调用，但显著提升公式和图表的提取质量。
2. **模型自动管理**：`lmstudio_manager` 封装了服务启动和模型加载逻辑，用户无需手动操作 LM Studio。
3. **低温度设置**：`temperature=0.1` 确保文档提取的确定性和一致性。
4. **错误处理边界**：API 调用失败、模型加载超时等异常在各模块内抛出，由 `main.py` 统一捕获并给出用户友好提示。
5. **可扩展性**：新增内容类型只需在 `ContentType` 枚举和 `prompts.py` 中添加对应项，无需修改核心流程。
