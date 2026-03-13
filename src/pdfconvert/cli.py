import argparse
import sys
from pathlib import Path

from .lmstudio.manager import ensure_model_loaded
from .core.pdf_parser import pdf_to_images
from .core.vision_analyzer import analyze_page
from .core.content_classifier import classify_page, ContentType
from .core.markdown_builder import build_markdown, save_markdown
from . import prompts
from .config import DEFAULT_DPI, DEFAULT_CONCURRENCY

PROMPT_MAP = {
    ContentType.TEXT: prompts.TEXT_EXTRACT_PROMPT,
    ContentType.FORMULA: prompts.FORMULA_EXTRACT_PROMPT,
    ContentType.DIAGRAM: prompts.DIAGRAM_DESCRIBE_PROMPT,
    ContentType.MIXED: prompts.MIXED_EXTRACT_PROMPT,
}


def convert_pdf(pdf_path: str, output_path: str | None = None, dpi: int = DEFAULT_DPI,
                max_pages: int | None = None, save_images: bool = True,
                progress_callback=None) -> dict:
    """
    将 PDF 转换为 Markdown（纯函数，无副作用）

    Args:
        pdf_path: PDF 文件路径
        output_path: 输出 Markdown 文件路径（可选）
        dpi: 渲染 DPI
        max_pages: 最多处理的页数（可选）
        save_images: 是否保存页面图片
        progress_callback: 进度回调函数 callback(stage, current, total, message)

    Returns:
        {
            "output_path": Path,
            "markdown": str,
            "pages": list[dict],
            "model_id": str
        }
    """
    def notify(stage, current=0, total=0, message=""):
        if progress_callback:
            progress_callback(stage, current, total, message)

    notify("init", message="检查 LM Studio 模型状态")
    model_id = ensure_model_loaded()

    notify("parse", message=f"解析 PDF: {pdf_path}")
    images = pdf_to_images(pdf_path, dpi=dpi)
    if max_pages:
        images = images[:max_pages]
    total = len(images)

    pages = []
    for i, image in enumerate(images, 1):
        notify("classify", i, total, "分类页面内容")
        classify_result = analyze_page(image, prompts.CLASSIFY_PROMPT, model_id)
        content_type = classify_page(classify_result)

        notify("extract", i, total, f"提取内容 ({content_type.value})")
        content = analyze_page(image, PROMPT_MAP[content_type], model_id)

        pages.append({
            "page_num": i,
            "content": content,
            "content_type": content_type.value,
            "image": image,
        })

    stem = Path(pdf_path).stem
    out_path = output_path or f"{stem}.md"

    notify("build", message="构建 Markdown")
    md_text = build_markdown(pages, title=stem, output_path=out_path, save_images=save_images)

    notify("save", message="保存文件")
    result_path = save_markdown(md_text, out_path)

    return {
        "output_path": result_path,
        "markdown": md_text,
        "pages": pages,
        "model_id": model_id
    }


def convert(pdf_path: str, output_path: str | None = None, dpi: int = DEFAULT_DPI,
            max_pages: int | None = None, save_images: bool = True) -> Path:
    """CLI 版本的转换函数（带进度输出）"""
    def print_progress(stage, current, total, message):
        if stage == "init":
            print(f"{message}...")
        elif stage == "parse":
            print(f"{message}")
        elif stage == "classify":
            print(f"[{current}/{total}] {message}...", end=" ", flush=True)
        elif stage == "extract":
            print(f"{message}")
            print(f"[{current}/{total}] 提取内容...", end=" ", flush=True)
            print("完成")
        elif stage == "save":
            pass

    result = convert_pdf(pdf_path, output_path, dpi, max_pages, save_images, print_progress)
    print(f"\n输出文件: {result['output_path']}")
    return result["output_path"]


def main():
    parser = argparse.ArgumentParser(description="PDF 转 Markdown（使用本地视觉模型）")
    parser.add_argument("pdf", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出 Markdown 文件路径")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"渲染 DPI（默认 {DEFAULT_DPI}）")
    parser.add_argument("--max-pages", type=int, help="最多处理的页数（用于测试）")
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="保存页面图片到输出目录（默认启用）")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false",
                        help="不保存页面图片")
    args = parser.parse_args()

    try:
        convert(args.pdf, args.output, args.dpi, args.max_pages, args.save_images)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
