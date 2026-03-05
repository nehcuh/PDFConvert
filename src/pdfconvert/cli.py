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


def convert(pdf_path: str, output_path: str | None = None, dpi: int = DEFAULT_DPI,
            max_pages: int | None = None, save_images: bool = True) -> Path:
    print("检查 LM Studio 模型状态...")
    model_id = ensure_model_loaded()
    print(f"使用模型: {model_id}")

    print(f"解析 PDF: {pdf_path}")
    images = pdf_to_images(pdf_path, dpi=dpi)
    if max_pages:
        images = images[:max_pages]
    total = len(images)
    print(f"共 {total} 页")

    pages = []
    for i, image in enumerate(images, 1):
        print(f"[{i}/{total}] 分类页面内容...", end=" ", flush=True)
        classify_result = analyze_page(image, prompts.CLASSIFY_PROMPT, model_id)
        content_type = classify_page(classify_result)
        print(f"{content_type.value}")

        print(f"[{i}/{total}] 提取内容...", end=" ", flush=True)
        content = analyze_page(image, PROMPT_MAP[content_type], model_id)
        print("完成")

        pages.append({
            "page_num": i,
            "content": content,
            "content_type": content_type.value,
            "image": image,  # 保存原始图片对象
        })

    stem = Path(pdf_path).stem
    out_path = output_path or f"{stem}.md"
    md_text = build_markdown(pages, title=stem, output_path=out_path, save_images=save_images)
    result = save_markdown(md_text, out_path)
    print(f"\n输出文件: {result}")
    return result


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
