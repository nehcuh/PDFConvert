from pathlib import Path
import sys
import re
from PIL import Image


def detect_and_save_diagrams(pages: list[dict], output_path: Path) -> dict[int, list[dict]]:
    """
    使用智谱 API 检测图示区域并保存裁剪后的图片

    Args:
        pages: 页面数据列表，包含 image 对象
        output_path: Markdown 输出路径

    Returns:
        {page_num: [{"path": str, "description": str}]} 映射
    """
    from ..zhipu.client import detect_diagram_regions

    images_dir = output_path.parent / f"{output_path.stem}_images"

    # 清空旧图片或创建目录
    if images_dir.exists():
        for old_image in images_dir.glob("page_*.png"):
            old_image.unlink()
    else:
        images_dir.mkdir(parents=True, exist_ok=True)

    page_diagrams = {}

    for page in pages:
        if "image" not in page:
            continue

        page_num = page["page_num"]
        image = page["image"]

        print(f"[{page_num}] 检测图示区域...", end=" ", flush=True)

        try:
            # 调用智谱 API 检测图示区域
            regions = detect_diagram_regions(image)
            print(f"发现 {len(regions)} 个图示")

            if not regions:
                continue

            diagrams = []
            width, height = image.size

            for idx, region in enumerate(regions, 1):
                description = region.get("description", "图示")
                bbox = region.get("bbox", [])

                if len(bbox) != 2:
                    print(f"  警告: 图示 {idx} 的 bbox 格式错误: {bbox}", file=sys.stderr)
                    continue

                y1_percent, y2_percent = bbox
                y1 = int(height * y1_percent / 100)
                y2 = int(height * y2_percent / 100)

                # 确保坐标有效
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))

                if y2 <= y1:
                    print(f"  警告: 图示 {idx} 的坐标无效: y1={y1}, y2={y2}", file=sys.stderr)
                    continue

                # 裁剪图片
                cropped = image.crop((0, y1, width, y2))

                # 保存图片
                image_filename = f"page_{page_num:02d}_{idx}.png"
                image_path = images_dir / image_filename
                cropped.save(image_path, format="PNG", optimize=True)

                # 保存相对路径
                relative_path = f"{output_path.stem}_images/{image_filename}"
                diagrams.append({
                    "path": relative_path,
                    "description": description
                })

            if diagrams:
                page_diagrams[page_num] = diagrams

        except Exception as e:
            print(f"失败: {e}", file=sys.stderr)

    return page_diagrams


def crop_image_by_position(image: Image.Image, position: str) -> Image.Image:
    """
    根据位置信息裁剪图片

    Args:
        image: PIL Image 对象
        position: 位置标识，格式为 "bbox: y1,y2" 或 "y1,y2" (百分比坐标) 或 "top/middle/bottom/full"

    Returns:
        裁剪后的图片
    """
    width, height = image.size

    # 移除 "bbox:" 前缀（如果存在）
    position = position.replace('bbox:', '').strip()

    # 尝试解析百分比坐标
    if ',' in position:
        try:
            parts = position.split(',')
            y1_percent = float(parts[0].strip())
            y2_percent = float(parts[1].strip())

            # 转换为像素坐标
            y1 = int(height * y1_percent / 100)
            y2 = int(height * y2_percent / 100)

            # 确保坐标在有效范围内
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if y2 > y1:
                return image.crop((0, y1, width, y2))
        except (ValueError, IndexError) as e:
            print(f"警告: 无法解析坐标 '{position}': {e}，使用完整图片", file=sys.stderr)
            return image

    # 兼容旧的位置标识
    if position == "top":
        return image.crop((0, 0, width, int(height * 0.4)))
    elif position == "middle":
        return image.crop((0, int(height * 0.3), width, int(height * 0.7)))
    elif position == "bottom":
        return image.crop((0, int(height * 0.6), width, height))
    else:  # full or unknown
        return image


def extract_image_placeholders(content: str) -> list[dict]:
    """
    从内容中提取所有图片标记信息

    Args:
        content: 页面内容

    Returns:
        [{"description": str, "position": str, "match": Match}] 列表
    """
    # 匹配 [IMAGE_PLACEHOLDER: description | position]
    # position 可以是 "bbox: y1,y2" 或 "y1,y2" 或 "top/middle/bottom/full"
    # 注意：模型可能会输出多余的 ']'，所以使用 [^\]]+ 来匹配到第一个 ]
    pattern = r'\[IMAGE_PLACEHOLDER:\s*([^\|]+?)\s*\|\s*([^\]]+?)\s*\]+'
    placeholders = []

    for match in re.finditer(pattern, content):
        placeholders.append({
            "description": match.group(1).strip(),
            "position": match.group(2).strip().lower(),
            "match": match
        })

    return placeholders


def save_page_images(pages: list[dict], output_path: Path) -> dict[int, list[str]]:
    """
    保存包含图片标记的页面图片到输出目录，根据位置信息裁剪

    Args:
        pages: 页面数据列表，包含 image 对象
        output_path: Markdown 输出路径

    Returns:
        {page_num: [relative_image_paths]} 映射（一个页面可能有多个图片）
    """
    images_dir = output_path.parent / f"{output_path.stem}_images"

    # 清空旧图片或创建目录
    if images_dir.exists():
        for old_image in images_dir.glob("page_*.png"):
            old_image.unlink()
    else:
        images_dir.mkdir(parents=True, exist_ok=True)

    page_image_paths = {}

    for page in pages:
        content = page["content"]
        placeholders = extract_image_placeholders(content)

        if placeholders and "image" in page:
            page_num = page["page_num"]
            image_paths = []

            for idx, placeholder in enumerate(placeholders, 1):
                position = placeholder["position"]
                image_filename = f"page_{page_num:02d}_{idx}.png"
                image_path = images_dir / image_filename

                try:
                    # 根据位置裁剪图片
                    cropped_image = crop_image_by_position(page["image"], position)
                    cropped_image.save(image_path, format="PNG", optimize=True)

                    # 保存相对路径
                    relative_path = f"{output_path.stem}_images/{image_filename}"
                    image_paths.append(relative_path)
                except Exception as e:
                    print(f"警告: 第 {page_num} 页图片 {idx} 保存失败: {e}", file=sys.stderr)
                    image_paths.append(None)

            page_image_paths[page_num] = image_paths

    return page_image_paths


def replace_image_placeholders(content: str, image_paths: list[str] | None) -> str:
    """
    将内容中的 [IMAGE_PLACEHOLDER: description | position] 替换为实际的图片引用

    Args:
        content: 页面内容
        image_paths: 图片相对路径列表（如果为 None，则移除标记）

    Returns:
        替换后的内容
    """
    # 匹配 [IMAGE_PLACEHOLDER: description | position]
    # position 可以是 "bbox: y1,y2" 或 "y1,y2" 或 "top/middle/bottom/full"
    # 注意：模型可能会输出多余的 ']'，所以使用 ]+ 来匹配一个或多个 ]
    pattern = r'\[IMAGE_PLACEHOLDER:\s*([^\|]+?)\s*\|\s*([^\]]+?)\s*\]+'

    if image_paths:
        # 使用计数器来跟踪当前替换到第几个图片
        counter = {"index": 0}

        def replace_func(match):
            description = match.group(1).strip()
            idx = counter["index"]

            if idx < len(image_paths) and image_paths[idx]:
                result = f"![{description}]({image_paths[idx]})"
            else:
                # 如果图片路径不足或为 None，保留描述
                result = f"**[图示: {description}]**"

            counter["index"] += 1
            return result

        return re.sub(pattern, replace_func, content)
    else:
        # 如果没有图片路径，移除标记但保留描述
        def replace_func(match):
            description = match.group(1).strip()
            return f"**[图示: {description}]**"

        return re.sub(pattern, replace_func, content)


def build_markdown(pages: list[dict], title: str | None = None,
                   output_path: str | Path | None = None,
                   save_images: bool = True) -> str:
    """
    将页面数据组装为 Markdown 文本

    Args:
        pages: 页面数据列表
        title: 文档标题
        output_path: 输出路径（用于保存图片）
        save_images: 是否保存图片

    Returns:
        Markdown 文本
    """
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    # 保存包含图片标记的页面图片
    image_paths = {}
    if save_images and output_path:
        image_paths = save_page_images(pages, Path(output_path))

    for page in pages:
        page_num = page["page_num"]
        content_type = page["content_type"]
        content = page["content"]

        parts.append(f"\n---\n\n<!-- Page {page_num} ({content_type}) -->\n")

        # 替换图片标记为实际的图片引用
        image_path = image_paths.get(page_num)
        processed_content = replace_image_placeholders(content, image_path)
        parts.append(processed_content)

    return "\n".join(parts)


def save_markdown(content: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)
    return path
