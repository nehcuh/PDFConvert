# PDFConvert 测试方案

## 1. 概述

本文档定义 PDFConvert（PDF 转 Markdown 工具）的完整测试策略。项目基于 Python 3.12 + uv 包管理，使用 LM Studio（端口 1234，OpenAI 兼容 API）运行 GLM-4.6V-Flash 视觉模型，实现 PDF 页面图片化后通过视觉模型提取文字、数学公式（LaTeX）和图表描述，输出结构化 Markdown。

### 1.1 测试范围

| 模块 | 说明 | 测试类型 |
|------|------|----------|
| LM Studio 模型检测 | 检测并加载 GLM-4.6V-Flash 模型 | 单元测试（mock） |
| PDF 转图片 | 每页 PDF 转为 PIL Image | 单元测试（mock） |
| 视觉模型 API 调用 | 通过 OpenAI 兼容 API 发送图片并获取识别结果 | 单元测试（mock） |
| Markdown 生成 | 将识别结果组装为结构化 Markdown | 单元测试 |
| 端到端流程 | 完整 PDF -> Markdown 转换 | 集成测试 |

### 1.2 技术栈

- 测试框架：pytest >= 8.0
- Mock 库：unittest.mock（标准库）
- 覆盖率：pytest-cov
- 异步测试：pytest-asyncio（如需）
- 标记管理：pytest.mark 自定义标记

---

## 2. pytest 目录结构

```
PDFConvert/
├── pyproject.toml
├── pdf_to_markdown.py
├── main.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # 全局 fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_lmstudio_detect.py  # LM Studio 模型检测
│   │   ├── test_pdf_to_image.py     # PDF 转图片
│   │   ├── test_vision_api.py       # 视觉模型 API 调用
│   │   └── test_markdown_gen.py     # Markdown 生成逻辑
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_e2e_convert.py      # 端到端转换测试
│   └── fixtures/
│       ├── sample_text_only.pdf     # 纯文字 PDF
│       ├── sample_with_formulas.pdf # 含数学公式 PDF
│       ├── sample_with_charts.pdf   # 含图表 PDF
│       ├── sample_mixed.pdf         # 混合内容 PDF
│       └── expected/               # 期望输出 Markdown
│           ├── text_only.md
│           ├── with_formulas.md
│           ├── with_charts.md
│           └── mixed.md
```

---

## 3. conftest.py 设计

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_DIR = FIXTURES_DIR / "expected"


# ---- pytest 自定义标记 ----

def pytest_configure(config):
    config.addinivalue_line("markers", "lmstudio: 需要真实 LM Studio 运行时的测试")
    config.addinivalue_line("markers", "slow: 执行时间较长的测试")
    config.addinivalue_line("markers", "gpu: 需要 GPU 的测试")


# ---- 通用 Fixtures ----

@pytest.fixture
def sample_pdf_path():
    """返回测试用 PDF 文件路径"""
    return FIXTURES_DIR / "sample_text_only.pdf"


@pytest.fixture
def fake_pil_image():
    """生成一个假的 PIL Image 用于测试"""
    return Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))


@pytest.fixture
def fake_page_images(fake_pil_image):
    """模拟多页 PDF 转换后的图片列表"""
    return [fake_pil_image] * 3


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI 兼容客户端"""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        "# Sample Title\n\n"
        "This is extracted text.\n\n"
        "$$E = mc^2$$\n\n"
        "[Figure: A bar chart showing data distribution]"
    )
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_lms_cli():
    """Mock lms CLI 命令调用"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"id": "glm-4v-flash-9b", "type": "llm"}]',
            stderr=""
        )
        yield mock_run
```

---

## 4. 单元测试用例

### 4.1 LM Studio 模型检测（test_lmstudio_detect.py）

```python
# tests/unit/test_lmstudio_detect.py
import pytest
from unittest.mock import patch, MagicMock


class TestLMStudioDetect:
    """LM Studio 模型检测与加载测试"""

    @patch("subprocess.run")
    def test_detect_lms_installed(self, mock_run):
        """验证能正确检测 lms CLI 是否已安装"""
        mock_run.return_value = MagicMock(returncode=0, stdout="lms version 0.1.0")
        # 调用检测函数，断言返回 True
        # result = detect_lms_installed()
        # assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_detect_lms_not_installed(self, mock_run):
        """lms CLI 未安装时应抛出明确错误"""
        mock_run.side_effect = FileNotFoundError("lms not found")
        # with pytest.raises(EnvironmentError, match="LM Studio"):
        #     detect_lms_installed()

    @patch("subprocess.run")
    def test_list_loaded_models(self, mock_run):
        """验证能正确列出已加载模型"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"id": "glm-4v-flash-9b", "type": "llm"}]'
        )
        # models = list_loaded_models()
        # assert "glm-4v-flash-9b" in [m["id"] for m in models]

    @patch("subprocess.run")
    def test_load_model_success(self, mock_run):
        """验证成功加载 GLM-4.6V-Flash 模型"""
        mock_run.return_value = MagicMock(returncode=0, stdout="Model loaded")
        # result = load_model("glm-4v-flash-9b")
        # assert result is True

    @patch("subprocess.run")
    def test_load_model_failure(self, mock_run):
        """模型加载失败时应返回错误信息"""
        mock_run.return_value = MagicMock(returncode=1, stderr="Model not found")
        # with pytest.raises(RuntimeError, match="Model not found"):
        #     load_model("nonexistent-model")

    @patch("subprocess.run")
    def test_model_already_loaded_skip(self, mock_run):
        """模型已加载时应跳过重复加载"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"id": "glm-4v-flash-9b", "type": "llm"}]'
        )
        # result = ensure_model_loaded("glm-4v-flash-9b")
        # assert result == "already_loaded"
```

### 4.2 PDF 转图片（test_pdf_to_image.py）

```python
# tests/unit/test_pdf_to_image.py
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


class TestPdfToImage:
    """PDF 转图片模块测试"""

    @patch("pdf2image.convert_from_path")
    def test_convert_single_page(self, mock_convert):
        """单页 PDF 应返回包含一张图片的列表"""
        fake_img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [fake_img]
        # images = pdf_to_images("test.pdf")
        # assert len(images) == 1
        # assert isinstance(images[0], Image.Image)

    @patch("pdf2image.convert_from_path")
    def test_convert_multi_page(self, mock_convert):
        """多页 PDF 应返回对应数量的图片"""
        fake_imgs = [Image.new("RGB", (100, 100)) for _ in range(5)]
        mock_convert.return_value = fake_imgs
        # images = pdf_to_images("test.pdf")
        # assert len(images) == 5

    @patch("pdf2image.convert_from_path")
    def test_convert_with_dpi_setting(self, mock_convert):
        """验证 DPI 参数正确传递"""
        mock_convert.return_value = [Image.new("RGB", (100, 100))]
        # pdf_to_images("test.pdf", dpi=300)
        # mock_convert.assert_called_with("test.pdf", dpi=300)

    @patch("pdf2image.convert_from_path")
    def test_convert_invalid_pdf(self, mock_convert):
        """无效 PDF 文件应抛出异常"""
        mock_convert.side_effect = Exception("Unable to read PDF")
        # with pytest.raises(Exception, match="Unable to read PDF"):
        #     pdf_to_images("invalid.pdf")

    def test_convert_nonexistent_file(self):
        """不存在的文件应抛出 FileNotFoundError"""
        # with pytest.raises(FileNotFoundError):
        #     pdf_to_images("nonexistent.pdf")

    @patch("pdf2image.convert_from_path")
    def test_convert_empty_pdf(self, mock_convert):
        """空 PDF（0 页）应返回空列表"""
        mock_convert.return_value = []
        # images = pdf_to_images("empty.pdf")
        # assert images == []
```

### 4.3 视觉模型 API 调用（test_vision_api.py）

```python
# tests/unit/test_vision_api.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import base64


class TestVisionAPI:
    """视觉模型 API 调用测试"""

    def test_image_to_base64(self, fake_pil_image):
        """PIL Image 应正确编码为 base64 字符串"""
        # b64 = image_to_base64(fake_pil_image)
        # assert isinstance(b64, str)
        # decoded = base64.b64decode(b64)
        # assert len(decoded) > 0

    def test_call_vision_api_success(self, mock_openai_client):
        """正常调用视觉 API 应返回 Markdown 文本"""
        # result = call_vision_api(mock_openai_client, "base64_image_data")
        # assert "Sample Title" in result
        # assert "$$E = mc^2$$" in result

    def test_call_vision_api_with_prompt(self, mock_openai_client):
        """验证 system prompt 正确传递给 API"""
        # call_vision_api(mock_openai_client, "base64_data", prompt="Extract text")
        # call_args = mock_openai_client.chat.completions.create.call_args
        # messages = call_args.kwargs["messages"]
        # assert any("Extract text" in str(m) for m in messages)

    def test_call_vision_api_timeout(self, mock_openai_client):
        """API 超时应抛出可重试异常"""
        mock_openai_client.chat.completions.create.side_effect = TimeoutError
        # with pytest.raises(TimeoutError):
        #     call_vision_api(mock_openai_client, "base64_data")

    def test_call_vision_api_rate_limit(self, mock_openai_client):
        """API 限流应正确处理"""
        mock_openai_client.chat.completions.create.side_effect = Exception("Rate limit")
        # with pytest.raises(Exception, match="Rate limit"):
        #     call_vision_api(mock_openai_client, "base64_data")

    def test_call_vision_api_empty_response(self, mock_openai_client):
        """API 返回空内容时应返回空字符串或抛出异常"""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = ""
        # result = call_vision_api(mock_openai_client, "base64_data")
        # assert result == "" or result is None

    def test_api_base_url_config(self):
        """验证 API base_url 默认指向 localhost:1234"""
        # client = create_vision_client()
        # assert "localhost:1234" in str(client.base_url)
```

### 4.4 Markdown 生成逻辑（test_markdown_gen.py）

```python
# tests/unit/test_markdown_gen.py
import pytest


class TestMarkdownGeneration:
    """Markdown 生成与格式化测试"""

    def test_page_separator(self):
        """每页之间应有分隔符"""
        pages = ["Page 1 content", "Page 2 content"]
        # md = assemble_markdown(pages)
        # assert md.count("---") >= 1

    def test_heading_extraction(self):
        """标题应正确转为 Markdown heading"""
        raw = "# Chapter 1\nSome text"
        # md = format_page_content(raw)
        # assert md.startswith("# Chapter 1")

    def test_latex_formula_inline(self):
        """行内公式应保留 $...$ 格式"""
        raw = "The formula $E=mc^2$ is famous."
        # md = format_page_content(raw)
        # assert "$E=mc^2$" in md

    def test_latex_formula_block(self):
        """块级公式应保留 $$...$$ 格式"""
        raw = "$$\\int_0^1 x^2 dx = \\frac{1}{3}$$"
        # md = format_page_content(raw)
        # assert "$$" in md
        # assert "\\int" in md

    def test_figure_description(self):
        """图表描述应转为 Markdown 格式"""
        raw = "[Figure: Bar chart of sales data]"
        # md = format_page_content(raw)
        # assert "Figure" in md

    def test_empty_page_handling(self):
        """空页面应生成空内容而非报错"""
        # md = format_page_content("")
        # assert md == "" or md.strip() == ""

    def test_output_file_creation(self, tmp_path):
        """输出文件应正确创建并写入内容"""
        output = tmp_path / "output.md"
        content = "# Test\nHello world"
        # save_markdown(content, str(output))
        # assert output.exists()
        # assert output.read_text() == content

    def test_default_output_path(self):
        """未指定输出路径时应使用 PDF 同名 .md"""
        # path = get_default_output_path("/path/to/input.pdf")
        # assert path == "/path/to/input.md"

    def test_multi_page_assembly_order(self):
        """多页内容应按页码顺序组装"""
        pages = {1: "First", 3: "Third", 2: "Second"}
        # md = assemble_markdown(pages)
        # assert md.index("First") < md.index("Second") < md.index("Third")

    def test_unicode_content_preserved(self):
        """中文和特殊字符应正确保留"""
        raw = "这是中文内容，包含特殊符号：α β γ"
        # md = format_page_content(raw)
        # assert "中文" in md
        # assert "α" in md
```

---

## 5. 集成测试策略

### 5.1 端到端转换测试（test_e2e_convert.py）

集成测试需要真实 LM Studio 运行时，使用 `@pytest.mark.lmstudio` 标记以便 CI 中跳过。

```python
# tests/integration/test_e2e_convert.py
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.lmstudio
@pytest.mark.slow
class TestEndToEnd:
    """端到端集成测试（需要 LM Studio 运行）"""

    def test_text_only_pdf(self, tmp_path):
        """纯文字 PDF 转换"""
        pdf = FIXTURES_DIR / "sample_text_only.pdf"
        output = tmp_path / "output.md"
        # result = convert_pdf_to_markdown(str(pdf), str(output))
        # assert output.exists()
        # content = output.read_text()
        # assert len(content) > 0
        # assert "---" in content  # 页面分隔符

    def test_formula_pdf(self, tmp_path):
        """含数学公式 PDF 转换"""
        pdf = FIXTURES_DIR / "sample_with_formulas.pdf"
        output = tmp_path / "output.md"
        # result = convert_pdf_to_markdown(str(pdf), str(output))
        # content = output.read_text()
        # assert "$$" in content or "$" in content  # 包含 LaTeX

    def test_chart_pdf(self, tmp_path):
        """含图表 PDF 转换"""
        pdf = FIXTURES_DIR / "sample_with_charts.pdf"
        output = tmp_path / "output.md"
        # result = convert_pdf_to_markdown(str(pdf), str(output))
        # content = output.read_text()
        # assert "Figure" in content or "图" in content

    def test_mixed_content_pdf(self, tmp_path):
        """混合内容 PDF 转换"""
        pdf = FIXTURES_DIR / "sample_mixed.pdf"
        output = tmp_path / "output.md"
        # result = convert_pdf_to_markdown(str(pdf), str(output))
        # content = output.read_text()
        # assert len(content) > 100

    def test_large_pdf_performance(self, tmp_path):
        """大文件性能测试（>20 页）"""
        pdf = FIXTURES_DIR / "sample_mixed.pdf"
        output = tmp_path / "output.md"
        import time
        # start = time.time()
        # convert_pdf_to_markdown(str(pdf), str(output))
        # elapsed = time.time() - start
        # assert elapsed < 300  # 5 分钟超时
```

### 5.2 LM Studio 连通性前置检查

```python
# tests/integration/conftest.py
import pytest
import requests


@pytest.fixture(autouse=True, scope="session")
def check_lmstudio_available():
    """集成测试前检查 LM Studio 是否可用"""
    try:
        resp = requests.get("http://localhost:1234/v1/models", timeout=5)
        if resp.status_code != 200:
            pytest.skip("LM Studio API 不可用")
    except requests.ConnectionError:
        pytest.skip("LM Studio 未运行（localhost:1234 无法连接）")
```

---

## 6. 测试数据准备建议

### 6.1 PDF 测试样本分类

| 类别 | 文件名 | 内容描述 | 测试重点 |
|------|--------|----------|----------|
| 纯文字 | `sample_text_only.pdf` | 英文/中文段落，标题层级 | OCR 准确率、段落识别 |
| 数学公式 | `sample_with_formulas.pdf` | 行内公式、块级公式、矩阵 | LaTeX 转换准确率 |
| 图表 | `sample_with_charts.pdf` | 柱状图、折线图、表格 | 图表描述生成质量 |
| 混合内容 | `sample_mixed.pdf` | 文字+公式+图表+代码块 | 综合识别与布局还原 |
| 手写体 | `sample_handwritten.pdf` | 手写笔记 | 手写识别鲁棒性 |
| 扫描件 | `sample_scanned.pdf` | 低质量扫描 | 低分辨率容错 |

### 6.2 期望输出（Golden Files）

每个测试 PDF 对应一个 `expected/*.md` 文件，作为回归测试的基准。首次运行时人工审核并确认，后续通过 diff 比较检测回归。

### 6.3 测试数据生成建议

- 使用 LaTeX 编译生成标准化 PDF（公式场景）
- 使用 matplotlib/seaborn 生成图表后导出 PDF（图表场景）
- 从公开学术论文中截取页面（混合场景）
- 手写样本可使用平板手写后导出

---

## 7. 质量评估指标

### 7.1 指标定义

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| OCR 准确率 | `正确字符数 / 总字符数` | >= 95% | 基于字符级编辑距离 |
| LaTeX 准确率 | `正确公式数 / 总公式数` | >= 85% | 公式可编译且语义正确 |
| 布局还原度 | `正确段落/标题数 / 总结构数` | >= 90% | 标题层级、段落分割 |
| 图表描述质量 | 人工评分 1-5 分 | >= 3.5 | 描述是否准确反映图表内容 |
| 页面完整率 | `成功处理页数 / 总页数` | 100% | 不应丢失任何页面 |

### 7.2 自动化评估脚本框架

```python
# tests/quality/evaluate.py
from difflib import SequenceMatcher


def ocr_accuracy(expected: str, actual: str) -> float:
    """基于 SequenceMatcher 计算字符级准确率"""
    return SequenceMatcher(None, expected, actual).ratio()


def latex_accuracy(expected_formulas: list[str], actual_formulas: list[str]) -> float:
    """计算 LaTeX 公式匹配率"""
    if not expected_formulas:
        return 1.0
    matched = sum(1 for e in expected_formulas if e in actual_formulas)
    return matched / len(expected_formulas)


def layout_score(expected_headings: list[str], actual_headings: list[str]) -> float:
    """计算标题/段落结构还原度"""
    if not expected_headings:
        return 1.0
    matched = sum(1 for e in expected_headings if e in actual_headings)
    return matched / len(expected_headings)
```

---

## 8. CI 友好的测试配置

### 8.1 pyproject.toml 测试配置

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "lmstudio: 需要真实 LM Studio 运行时（CI 中跳过）",
    "slow: 执行时间较长的测试",
    "gpu: 需要 GPU 的测试",
]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["."]
omit = ["tests/*", ".venv/*"]

[tool.coverage.report]
fail_under = 70
show_missing = true
```

### 8.2 CI 运行命令

```bash
# 仅运行单元测试（CI 默认）
uv run pytest tests/unit/ -v --cov --cov-report=term-missing

# 跳过需要 LM Studio 的测试
uv run pytest -m "not lmstudio" -v

# 跳过慢速测试
uv run pytest -m "not slow" -v

# 本地完整测试（需要 LM Studio 运行）
uv run pytest -v --cov --cov-report=html

# 仅运行集成测试
uv run pytest tests/integration/ -v -m lmstudio
```

### 8.3 GitHub Actions 示例

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run pytest tests/unit/ -v --cov --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

### 8.4 Makefile 快捷命令

```makefile
.PHONY: test test-unit test-integration test-cov

test:
	uv run pytest -m "not lmstudio" -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m lmstudio

test-cov:
	uv run pytest --cov --cov-report=html -m "not lmstudio"
	open htmlcov/index.html
```

---

## 9. 测试执行优先级

| 优先级 | 测试类别 | 触发时机 | 预计耗时 |
|--------|----------|----------|----------|
| P0 | 单元测试（全部） | 每次提交 | < 30s |
| P1 | 集成测试（LM Studio） | 本地手动 / 发版前 | 2-5 min |
| P2 | 质量评估（Golden File 对比） | 模型或 prompt 变更时 | 5-10 min |
| P3 | 性能测试（大文件） | 发版前 | 5-10 min |
