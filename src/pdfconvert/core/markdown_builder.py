from pathlib import Path


def build_markdown(pages: list[dict], title: str | None = None) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
    for page in pages:
        parts.append(f"\n---\n\n<!-- Page {page['page_num']} ({page['content_type']}) -->\n")
        parts.append(page["content"])
    return "\n".join(parts)


def save_markdown(content: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)
    return path
